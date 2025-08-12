import io
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import cv2
import numpy as np
import threading
import asyncio

app = FastAPI()

class FrameCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None  # 存储最新的BGR numpy数组
        self.version = 0    # 帧版本号
        self.last_updated = 0  # 最后更新时间戳
        self.event = asyncio.Event()  # 新帧到达事件

    def update(self, frame: np.ndarray):
        with self.lock:
            self.frame = frame.copy()
            self.version += 1
            self.last_updated = time.time()
            self.event.set()  # 触发新帧事件
            self.event.clear()  # 重置事件

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def get_version(self):
        with self.lock:
            return self.version
            
    async def wait_for_update(self, client_version: int, timeout: float = 30.0):
        """等待新帧到达或超时"""
        # 如果客户端版本已是最新，立即返回
        if client_version < self.version:
            return True
            
        try:
            # 等待新帧事件触发或超时
            await asyncio.wait_for(self.event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

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
    """获取最新帧图像，可选版本号参数"""
    # 如果请求指定版本且是最新版本，返回304减少传输
    current_version = frame_cache.get_version()
    if version == current_version:
        return Response(status_code=304)
    
    frame = frame_cache.get_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="暂无图片数据")

    # 使用较低质量压缩
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    if not ret:
        raise HTTPException(status_code=500, detail="编码图片失败")

    # 添加缓存控制头
    headers = {
        "X-Frame-Version": str(current_version),
        "Cache-Control": "no-cache, no-store, must-revalidate"
    }
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()), 
        media_type="image/jpeg",
        headers=headers
    )

@app.get("/check_update")
async def check_update(version: int, timeout: int = 30):
    """长轮询接口检查是否有新帧"""
    if await frame_cache.wait_for_update(version, timeout):
        return JSONResponse({"update": True, "new_version": frame_cache.get_version()})
    return JSONResponse({"update": False, "current_version": frame_cache.get_version()})

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
            
            function updateImage() {
                const img = document.getElementById('liveImage');
                // 添加时间戳防止缓存
                const url = `/latest_frame?t=${Date.now()}` + 
                            (currentVersion >= 0 ? `&version=${currentVersion}` : '');
                img.src = url;
                
                // 从响应头获取版本号
                fetch(url, { method: 'HEAD' })
                    .then(response => {
                        const versionHeader = response.headers.get('X-Frame-Version');
                        if (versionHeader) {
                            currentVersion = parseInt(versionHeader);
                            retryCount = 0; // 重置重试计数器
                        }
                    })
                    .catch(error => {
                        console.error('获取版本号失败:', error);
                        if (++retryCount > MAX_RETRIES) {
                            console.error('达到最大重试次数，重新加载页面');
                            setTimeout(() => location.reload(), 2000);
                        }
                    });
            }
            
            async function checkForUpdates() {
                try {
                    const response = await fetch(`/check_update?version=${currentVersion}&timeout=300`);
                    const data = await response.json();
                    
                    if (data.update) {
                        currentVersion = data.new_version;
                        updateImage();
                    }
                    // 无论是否有更新，都继续检查
                    checkForUpdates();
                } catch (error) {
                    console.error('更新检查失败:', error);
                    // 失败后重试
                    setTimeout(checkForUpdates, 2000);
                }
            }
            
            // 初始加载
            updateImage();
            // 开始长轮询更新检查
            checkForUpdates();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.4.4", port=8000)