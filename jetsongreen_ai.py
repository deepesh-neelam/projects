"""
jetsongreen_ai.py

a single-file prototype for jetsongreen ai

features
- camera based detection using ultralytics yolov8 (or opencv + pre-trained model)
- simple predictor (exponential smoothing) for short term demand forecasting
- gpio relay control (uses jetson.gpio when available, else simulator)
- flask dashboard serving video and real-time stats via server-sent events

requirements
- python 3.8+
- opencv-python
- ultralytics (pip install ultralytics) or provide a yolov8 weights file
- flask
- numpy
- psutil (optional for system metrics)
- on nvidia jetson: jetson gpio python package (jetson_gpio)

note
- this is a prototype skeleton intended to run on an nvidia jetson device
- replace model weights, tune detection classes, and map relays to devices for your demo

usage
- edit RELAY_PINS mapping to match your relay wiring
- run: python3 jetsongreen_ai.py

"""

import threading
import time
import json
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, stream_with_context

try:
    # ultralytics provides yolov8 models and torch acceleration on jetson
    from ultralytics import YOLO # type: ignore
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

# try import jetson gpio otherwise provide simulator
try:
    import Jetson.GPIO as GPIO # type: ignore
    HAS_GPIO = True
except Exception:
    HAS_GPIO = False

# simple gpio simulator for development
class GPIOSim:
    def __init__(self):
        self.state = {}
    def setup(self,pin,mode):
        self.state[pin]=False
    def output(self,pin,val):
        self.state[pin]=bool(val)
    def input(self,pin):
        return self.state.get(pin,False)
    def cleanup(self):
        self.state={}

if not HAS_GPIO:
    GPIO_SIM = GPIOSim()
else:
    GPIO_SIM = None

# configuration
CAMERA_ID = 0  # default camera
MODEL_PATH = 'yolov8n.pt'  # replace with your model or pretrained
DETECTION_CLASSES = ['person','light','fan','monitor','ac']  # labels your model can detect

# mapping relays to device names and gpio pins (edit for your wiring)
RELAY_PINS = {
    'lights': 11,
    'fan': 13,
}

# control thresholds and params
OCCUPANCY_COOLDOWN = 60  # seconds to wait before turning off devices
PREDICT_WINDOW = 6  # number of historical samples used in predictor
SMOOTH_ALPHA = 0.3  # smoothing factor for exp smoothing

# global state shared between threads
state = {
    'frame': None,
    'detections': [],
    'occupied': False,
    'last_occupied_time': None,
    'estimated_power_w': 0.0,
    'predicted_power_w': 0.0,
    'logs': deque(maxlen=200),
}

class Detector:
    def __init__(self, camera_id=CAMERA_ID, model_path=MODEL_PATH):
        self.cap = cv2.VideoCapture(camera_id)
        self.model = None
        if HAS_YOLO:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print('yolo model load failed', e)
                self.model = None
        if self.model is None:
            print('yolo not available fallback to background subtraction for motion')
        # simple background subtractor
        self.backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # resize for faster inference
        h, w = frame.shape[:2]
        maxw = 640
        if w > maxw:
            frame = cv2.resize(frame, (maxw, int(h * maxw / w)))
        return frame

    def detect(self, frame):
        detections = []
        if self.model is not None:
            try:
                results = self.model.predict(frame, imgsz=640, conf=0.35, verbose=False)
                # results is a list one element per image
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
                        conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else 0.0
                        xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box, 'xyxy') else None
                        # map class id to name if possible
                        lab = str(cls) if cls is not None else 'obj'
                        detections.append({'label': lab, 'conf': conf, 'box': xyxy})
            except Exception as e:
                print('yolo inference failed', e)
        else:
            # fallback motion detection
            fgmask = self.backsub.apply(frame)
            cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion = False
            for c in cnts:
                if cv2.contourArea(c) > 500:
                    motion = True
                    break
            if motion:
                detections.append({'label':'motion','conf':1.0,'box':None})
        return detections

class Predictor:
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha = alpha
        self.last = None
        self.history = deque(maxlen=PREDICT_WINDOW)
    def update(self, sample):
        self.history.append(sample)
        if self.last is None:
            self.last = sample
        else:
            self.last = self.alpha * sample + (1-self.alpha)*self.last
        return self.last
    def predict(self):
        # simple predictor: return current smoothed value
        return self.last if self.last is not None else 0.0

class Controller:
    def __init__(self, relay_pins=RELAY_PINS):
        self.relay_pins = relay_pins
        self.gpio = GPIO if HAS_GPIO else GPIO_SIM
        self.setup()
        self.states = {k:False for k in relay_pins}
    def setup(self):
        if HAS_GPIO:
            GPIO.setmode(GPIO.BOARD)
            for pin in self.relay_pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
        else:
            for pin in self.relay_pins.values():
                GPIO_SIM.setup(pin, None)
    def set(self, device, on):
        pin = self.relay_pins.get(device)
        if pin is None:
            return
        val = GPIO.HIGH if HAS_GPIO else True
        if not on:
            val = GPIO.LOW if HAS_GPIO else False
        try:
            if HAS_GPIO:
                GPIO.output(pin, val)
            else:
                GPIO_SIM.output(pin, val)
            self.states[device] = bool(on)
            state['logs'].appendleft(f"{datetime.now().isoformat()} set {device} -> {on}")
        except Exception as e:
            print('gpio set failed', e)
    def cleanup(self):
        if HAS_GPIO:
            GPIO.cleanup()

# small utility to estimate power from detections
label_power_map = {
    'light': 15.0,
    'lights': 60.0,
    'fan': 50.0,
    'ac': 1000.0,
    'monitor': 25.0,
}

def estimate_power(detections):
    total = 0.0
    for d in detections:
        lab = d.get('label','').lower()
        for k,v in label_power_map.items():
            if k in lab:
                total += v
    # motion fallback assume small baseline
    if not detections:
        total = 5.0
    return total

# flask dashboard
app = Flask(__name__)

INDEX_HTML = '''
<html>
<head><title>jetson green ai dashboard</title></head>
<body>
<h3>jetson green ai live</h3>
<img id="stream" src="/video_feed" width="640" />
<div>
<pre id="stats"></pre>
</div>
<script>
var es = new EventSource('/stats');
es.onmessage = function(e){
  var data = JSON.parse(e.data);
  document.getElementById('stats').innerText = JSON.stringify(data,null,2);
}
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

def gen_frames():
    while True:
        frame = state.get('frame')
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    def event_stream():
        while True:
            payload = {
                'time': datetime.now().isoformat(),
                'occupied': state['occupied'],
                'estimated_power_w': state['estimated_power_w'],
                'predicted_power_w': state['predicted_power_w'],
                'last_occupied_time': state['last_occupied_time'].isoformat() if state['last_occupied_time'] else None,
                'logs': list(state['logs'])[:10]
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

# main application loop

def main_loop():
    detector = Detector()
    predictor = Predictor()
    controller = Controller()
    try:
        while True:
            frame = detector.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            detections = detector.detect(frame)
            # mark frame with simple overlays
            disp = frame.copy()
            y = 20
            for d in detections:
                lab = d.get('label','')
                conf = d.get('conf',0)
                cv2.putText(disp, f"{lab} {conf:.2f}",(10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
                y += 20
            # occupancy heuristics
            occupied = any('person' in (d.get('label','').lower()) or d.get('label')=='motion' for d in detections)
            now = datetime.now()
            if occupied:
                state['last_occupied_time'] = now
            else:
                # if cooldown passed since last occupied set occupied false
                if state['last_occupied_time'] is None:
                    pass
                else:
                    delta = (now - state['last_occupied_time']).total_seconds()
                    if delta > OCCUPANCY_COOLDOWN:
                        state['last_occupied_time'] = None
            state['occupied'] = state['last_occupied_time'] is not None

            # estimate power usage
            est = estimate_power(detections)
            state['estimated_power_w'] = est

            # predictor update and forecast
            predictor.update(est)
            pred = predictor.predict()
            state['predicted_power_w'] = pred

            # simple automation logic
            if not state['occupied']:
                # turn off lights and fan if currently on
                controller.set('lights', False)
                controller.set('fan', False)
            else:
                # ensure lights on during occupancy and low ambient light detected
                controller.set('lights', True)
                controller.set('fan', True)

            # overlay stats
            cv2.putText(disp, f"occupied: {state['occupied']}",(10,disp.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)
            cv2.putText(disp, f"est w: {state['estimated_power_w']:.1f}",(10,disp.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)
            cv2.putText(disp, f"pred w: {state['predicted_power_w']:.1f}",(10,disp.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)

            state['frame'] = disp
            time.sleep(0.1)
    except KeyboardInterrupt:
        controller.cleanup()

if __name__ == '__main__':
    # run main loop in background thread and flask in main thread
    t = threading.Thread(target=main_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
