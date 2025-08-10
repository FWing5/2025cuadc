import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import threading

app = FastAPI()

class FrameCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None  # 存储最新的BGR numpy数组

    def update(self, frame: np.ndarray):
        with self.lock:
            self.frame = frame.copy()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

frame_cache = FrameCache()

@app.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...)):
    # 接收客户端上传的JPEG图片，解码为numpy数组保存
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码上传的图片")

    frame_cache.update(img)
    return {"status": "frame updated"}

@app.get("/latest_frame")
def latest_frame():
    frame = frame_cache.get_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="暂无图片数据")

    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        raise HTTPException(status_code=500, detail="编码图片失败")

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
