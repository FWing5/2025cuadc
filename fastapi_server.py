import io
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import cv2
import numpy as np
import threading
import asyncio

app = FastAPI()

class FrameCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.version = 0
        self.last_updated = 0.0

    def update(self, frame: np.ndarray):
        with self.lock:
            self.frame = frame.copy()
            self.version += 1
            self.last_updated = time.time()

    def get_frame_and_version(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy(), self.version)

    def get_version(self):
        with self.lock:
            return self.version
            
async def wait_for_update_poll(cache: FrameCache, client_version: int, timeout: float = 30.0, interval: float = 0.2):
    # 轮询版本变化，避免跨事件循环对象
    start = time.time()
    while True:
        current_version = cache.get_version()
        if client_version < current_version:
            return True, current_version
        if time.time() - start >= timeout:
            return False, current_version
        await asyncio.sleep(interval)

frame_cache = FrameCache()

@app.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    # 接收客户端上传的JPEG图片，解码为numpy数组保存
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码上传的图片")

    # 压缩图片尺寸以减小带宽
    scale_factor = 0.7  # 调整为0.3-0.8之间平衡质量和大小
    if scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    
    frame_cache.update(img)
    return {"status": "frame updated", "version": frame_cache.get_version()}

@app.get("/latest_frame")
def latest_frame(version: int = -1):
    current_version = frame_cache.get_version()
    # 如果客户端的版本等于当前版本，则返回 304，减少流量
    if version == current_version and current_version > 0:
        return Response(status_code=304, headers={
            "X-Frame-Version": str(current_version),
            "Cache-Control": "no-cache, no-store, must-revalidate",
        })

    frame, current_version = frame_cache.get_frame_and_version()
    headers = {
        "X-Frame-Version": str(current_version),
        "Cache-Control": "no-cache, no-store, must-revalidate",
    }

    if frame is None:
        # 没有帧，返回 204，避免报错
        return Response(status_code=204, headers=headers)

    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ret:
        raise HTTPException(status_code=500, detail="编码图片失败")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg", headers=headers)

@app.get("/check_update")
async def check_update(version: int = -1, timeout: int = 30):
    # 如果客户端从未拿过版本(-1)，不强制立即 update。使用轮询等待，保证没有输入时不报错
    updated, new_version = await wait_for_update_poll(frame_cache, version, float(timeout))
    return JSONResponse({"update": updated, "new_version": new_version})

@app.get("/", response_class=HTMLResponse)
def video_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>低带宽实时监控</title>
        <style>
            body { margin: 0; background: #000; }
            img { width: 100%; height: 100vh; object-fit: contain; }
        </style>
    </head>
    <body>
        <img id="liveImage" src="/latest_frame">
        
        <script>
            let currentVersion = -1;
            let retryCount = 0;
            const MAX_RETRIES = 5;

            async function updateImage() {
                try {
                    const url = `/latest_frame` + (currentVersion >= 0 ? `?version=${currentVersion}` : '') + `&t=${Date.now()}`;
                    const response = await fetch(url);
                    
                    if (response.status === 200) {
                        const versionHeader = response.headers.get('X-Frame-Version');
                        if (versionHeader) {
                            currentVersion = parseInt(versionHeader);
                            retryCount = 0;
                        }
                        const blob = await response.blob();
                        document.getElementById('liveImage').src = URL.createObjectURL(blob);
                    } 
                } catch (error) {
                    console.error('图片更新失败:', error);
                    if (++retryCount > MAX_RETRIES) {
                        console.error('达到最大重试次数，重新加载页面');
                        setTimeout(() => location.reload(), 2000);
                    }
                }
            }

            async function checkForUpdates() {
                try {
                    const response = await fetch(`/check_update?version=${currentVersion}&timeout=300`);
                    const data = await response.json();
                    
                    if (data.update) {
                        currentVersion = data.new_version;
                        updateImage();
                    }
                    checkForUpdates();
                } catch (error) {
                    console.error('更新检查失败:', error);
                    setTimeout(checkForUpdates, 2000);
                }
            }

            // 定时刷新，每秒一次
            setInterval(updateImage, 1000);

            // 初始加载
            updateImage();
            checkForUpdates();
        </script>

    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)