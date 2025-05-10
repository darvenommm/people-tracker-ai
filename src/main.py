import uuid
import asyncio
import base64
import json
import cv2
from uuid import UUID
from pathlib import Path
from fastapi import (
  FastAPI, UploadFile, File, BackgroundTasks,
  HTTPException, Path as FastPath, Response, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

BASE_DIR   = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
VIDEO_DIR  = BASE_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Person Tracker")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

class StatsResponse(BaseModel):
  unique_people: int
  total_frames: int

class TracksResponse(BaseModel):
  tracks: list[dict]

class SessionData:
  def __init__(self):
    self.queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    self.unique_ids: set[int] = set()
    self.total_frames: int = 0
    self.running: bool = True
    self.last_tracks: list[dict] = []

sessions: dict[UUID, SessionData] = {}

model = YOLO('yolov8n.pt')

async def tracking_worker(session_id: UUID, video_path: str):
  sess = sessions.get(session_id)
  cap = await asyncio.to_thread(cv2.VideoCapture, video_path)
  deepsort = DeepSort(max_age=30)

  while sess and sess.running:
    ret, frame = await asyncio.to_thread(cap.read)
    if not ret:
      break
    sess.total_frames += 1

    results = await asyncio.to_thread(model, frame)
    dets = []
    for *xyxy, conf, cls in results[0].boxes.data.tolist():
      if int(cls) == 0 and conf > 0.3:
        x1, y1, x2, y2 = map(int, xyxy)
        dets.append(((x1, y1, x2 - x1, y2 - y1), conf, 'person'))

    tracks = await asyncio.to_thread(deepsort.update_tracks, dets, frame=frame)
    out_tracks = []
    for tr in tracks:
      if not tr.is_confirmed():
        continue

      tid = tr.track_id
      sess.unique_ids.add(tid)
      bx, by, bw, bh = tr.to_ltwh()
      x1, y1 = int(bx), int(by)
      x2, y2 = int(bx + bw), int(by + bh)
      out_tracks.append({'id': tid, 'bbox': [x1, y1, x2 - x1, y2 - y1]})

      def draw():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
          frame,
          f'ID {tid}',
          (x1, max(y1-5,0)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0,255,0),
          2,
        )
      await asyncio.to_thread(draw)

    sess.last_tracks = out_tracks

    _, buf = await asyncio.to_thread(cv2.imencode, '.jpg', frame)
    b64 = base64.b64encode(buf).decode('utf-8')

    try:
      sess.queue.put_nowait({'jpeg_b64': b64, 'tracks': out_tracks})
    except asyncio.QueueFull:
      pass

    await asyncio.sleep(0.03)

  if sess:
    sess.running = False

  cap.release()

@app.post("/track", summary="Загрузить видео")
async def start_tracking(
  file: UploadFile = File(...),
  background_tasks: BackgroundTasks = None,
):
    session_id = uuid.uuid4()
    path = VIDEO_DIR / f"{session_id}.mp4"

    with open(path, "wb") as f:
      f.write(await file.read())

    sessions[session_id] = SessionData()
    background_tasks.add_task(tracking_worker, session_id, str(path))

    return {"session_id": session_id}

@app.get(
  "/track/{session_id}/frames",
  response_class=EventSourceResponse,
)
async def stream_frames(session_id: UUID = FastPath(...)):
  sess = sessions.get(session_id)

  if not sess:
    raise HTTPException(404, "Session not found")

  async def event_generator():
    while sess.running or not sess.queue.empty():
      try:
        item = sess.queue.get_nowait()
        payload = json.dumps(item)
        yield {"event": "frame", "data": payload}
      except asyncio.QueueEmpty:
        await asyncio.sleep(0.1)

  return EventSourceResponse(event_generator())

@app.get(
  "/track/{session_id}/data",
  response_model=TracksResponse,
  summary="Получить последний набор треков",
)
async def get_latest_tracks(session_id: UUID = FastPath(...)):
  sess = sessions.get(session_id)

  if not sess:
    raise HTTPException(404, "Session not found")

  return TracksResponse(tracks=sess.last_tracks)

@app.get(
  "/stats/{session_id}",
  response_model=StatsResponse,
  summary="Статистика по сессии",
)
async def get_stats(session_id: UUID = FastPath(...)):
  sess = sessions.get(session_id)

  if not sess:
    raise HTTPException(404, "Session not found")

  return StatsResponse(
    unique_people=len(sess.unique_ids),
    total_frames=sess.total_frames,
  )

@app.delete(
  "/track/{session_id}",
  status_code=status.HTTP_204_NO_CONTENT,
  summary="Остановить трекинг и удалить сессию",
)
async def delete_session(session_id: UUID = FastPath(...)):
  sess = sessions.get(session_id)

  if not sess:
    raise HTTPException(404, "Session not found")

  sess.running = False
  sessions.pop(session_id, None)

  file_path = VIDEO_DIR / f"{session_id}.mp4"

  if file_path.exists():
    file_path.unlink()

  return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/", include_in_schema=False)
async def root():
  return FileResponse(STATIC_DIR / "client.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
  import uvicorn

  uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
