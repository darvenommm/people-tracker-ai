import uuid
import asyncio
import base64
import json
import time
import io

import cv2
import torch
import uvloop
import av

from uuid import UUID
from pathlib import Path
from collections import deque
from PIL import Image

from fastapi import (
  FastAPI, UploadFile, File, HTTPException,
  Response, status, Path as FPathParam
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ultralytics import YOLO
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from torch import amp

# fast event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# paths & device
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SCALE = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVERY_COUNT_FRAME = 1  # process every frame

# API setup
app = FastAPI(title="Person Tracker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# response schemas
class StatsResponse(BaseModel):
  unique_people: int
  total_frames: int

class TracksResponse(BaseModel):
  tracks: list[dict]

class EventRecord(BaseModel):
  track_id: int
  action: str
  timestamp: float

# load models
yolo = YOLO('yolov8n.pt')
action_model = r3d_18(weights=R3D_18_Weights.DEFAULT).to(DEVICE)
action_model.eval()

transform = transforms.Compose([
  transforms.Resize((112,112)),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.43216,0.394666,0.37645],
    std=[0.22803,0.22145,0.216989]
  ),
])

# action labels
labels = R3D_18_Weights.DEFAULT.meta["categories"]
DANCE_CLASSES = {i for i,name in enumerate(labels) if "danc" in name.lower() or "zumba" in name.lower() or "krump" in name.lower()}
DANCE_LABELS = {labels[i] for i in DANCE_CLASSES}

# in-memory sessions
sessions: dict[UUID, dict] = {}

async def processing_worker(sid: UUID):
  sess = sessions[sid]
  container = av.open(io.BytesIO(sess['video_bytes']), format='mp4', mode='r')
  deepsort = DeepSort(max_age=30, embedder_gpu=(DEVICE=='cuda'))

  frame_count = 0
  for packet in container.demux(video=0):
    if not sess['running']:
      break

    for frame in packet.decode():
      if not sess['running']:
        break

      # frame skipping
      frame_count += 1
      if frame_count % EVERY_COUNT_FRAME != 0:
        continue

      img = frame.to_ndarray(format='bgr24')
      ts = float(frame.time)
      sess['total_frames'] += 1

      # detection
      small = cv2.resize(img, (0,0), fx=SCALE, fy=SCALE)
      if DEVICE=='cuda':
        with amp.autocast(device_type='cuda', enabled=True):
          results = yolo(small)
      else:
        results = yolo(small)

      dets = []
      for *xyxy, conf, cls in results[0].boxes.data.tolist():
        if int(cls)==0 and conf>0.3:
          x1,y1,x2,y2 = map(int, xyxy)
          x1=int(x1/SCALE); y1=int(y1/SCALE)
          x2=int(x2/SCALE); y2=int(y2/SCALE)
          dets.append(((x1,y1,x2-x1,y2-y1), conf, 'person'))

      # tracking + action
      tracks = deepsort.update_tracks(dets, frame=img)
      out_tracks = []
      for tr in tracks:
        if not tr.is_confirmed():
          continue

        tid = tr.track_id
        sess['unique_ids'].add(tid)
        bx,by,bw,bh = tr.to_ltwh()
        x1,y1 = int(bx), int(by)
        out_tracks.append({'id':tid,'bbox':[x1,y1,int(bw),int(bh)]})

        # accumulate ROI frames
        x2,y2 = x1+int(bw), y1+int(bh)
        x1c,y1c = max(0,x1), max(0,y1)
        x2c,y2c = min(x2,img.shape[1]), min(y2,img.shape[0])

        if x2c>x1c and y2c>y1c:
          roi = img[y1c:y2c, x1c:x2c]
          if roi.size:
            buf16 = sess['track_buf'].setdefault(tid, deque(maxlen=16))
            buf16.append(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            if len(buf16)==16:
              clip = torch.stack([transform(Image.fromarray(f)) for f in buf16], dim=1).to(DEVICE)

              if DEVICE=='cuda':
                with amp.autocast(device_type='cuda', enabled=True):
                  preds = action_model(clip.unsqueeze(0))
              else:
                preds = action_model(clip.unsqueeze(0))
              idx = preds.argmax(1).item()
              label = labels[idx]
              prev = sess['statuses'].get(tid)
              sess['statuses'][tid] = label

              if label != prev:
                sess['events'].append({'track_id': tid,'action': label,'timestamp': time.time()})

      sess['last_tracks'] = out_tracks

      # prepare SSE payload
      payload = []
      for t in out_tracks:
        action = sess['statuses'].get(t['id'], 'unknown')
        icon   = '💃' if action in DANCE_LABELS else '🚶'
        payload.append({**t,'action': action,'icon': icon})

      _, jpg = cv2.imencode('.jpg', img)
      b64    = base64.b64encode(jpg).decode('utf-8')

      await sess['send_q'].put({'jpeg_b64': b64,'tracks': payload,'timestamp': ts})
      await asyncio.sleep(0)

  sess['running'] = False
  container.close()

# start processing
@app.post("/track")
async def start_tracking(file: UploadFile = File(...)):
  sid = uuid.uuid4()
  data = await file.read()
  sessions[sid] = {
    'video_bytes': data,
    'send_q':      asyncio.Queue(maxsize=10),
    'unique_ids':  set(),
    'total_frames':0,
    'running':     True,
    'last_tracks': [],
    'track_buf':   {},
    'statuses':    {},
    'events':      []
  }

  asyncio.create_task(processing_worker(sid))

  return {"session_id": sid}

# SSE stream
@app.get("/track/{session_id}/frames", response_class=EventSourceResponse)
async def stream_frames(session_id: UUID = FPathParam(...)):
  sess = sessions.get(session_id)

  if not sess:
    raise HTTPException(404, "Session not found")

  async def gen():
    while sess['running'] or not sess['send_q'].empty():
      msg = await sess['send_q'].get()
      yield {"event":"frame","data":json.dumps(msg)}

  return EventSourceResponse(gen())

# stats endpoint
@app.get("/stats/{session_id}", response_model=StatsResponse)
async def get_stats(session_id: UUID = FPathParam(...)):
  sess = sessions.get(session_id) or HTTPException(404, "Session not found")

  return StatsResponse(unique_people=len(sess['unique_ids']), total_frames=sess['total_frames'])

# last tracks
@app.get("/track/{session_id}/data", response_model=TracksResponse)
async def get_tracks(session_id: UUID = FPathParam(...)):
  sess = sessions.get(session_id) or HTTPException(404, "Session not found")

  return TracksResponse(tracks=sess['last_tracks'])

# session events
@app.get("/track/{session_id}/events", response_model=list[EventRecord])
async def get_events(session_id: UUID = FPathParam(...)):
  sess = sessions.get(session_id) or HTTPException(404, "Session not found")

  return [EventRecord(**e) for e in sess['events']]

# stop session
@app.delete("/track/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: UUID = FPathParam(...)):
  sess = sessions.pop(session_id, None)

  if not sess:
    raise HTTPException(404, "Session not found")

  sess['running'] = False

  return Response(status_code=status.HTTP_204_NO_CONTENT)

# serve client
@app.get("/", include_in_schema=False)
async def root():
  return FileResponse(STATIC_DIR / "client.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
