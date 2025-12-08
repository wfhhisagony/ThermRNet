import multiprocessing
from multiprocessing import Process, Event, Value, Array, Queue
import numpy as np
import cv2
import socket
import json
import base64
import time
import sys
import copy
import logging
from collections import deque
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import scipy.signal  # Added for Median Filter

# Import your existing config and detectors
# Assuming rr_detector contains your YOLO pose model class
from rr_detector import MyYoloPoseModel
from rr_models import ThermRNet
import os

# --- Constants ---
THERMAL_FRAME_WIDTH = 192
THERMAL_FRAME_HEIGHT = 256
THERMAL_FPS = 25
# Model Requirements
MODEL_INPUT_H = 72
MODEL_INPUT_W = 72
MODEL_SEQ_LEN = 160  # Frames for the model context
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pretrained_model_files", "best_epoch_49.pth")  # SET THIS to your trained model path

# Signal Buffer Config
# We store ~30 seconds of model predictions to calculate RR reliably
SIGNAL_BUFFER_LEN = THERMAL_FPS * 30

# Buffer & Logic Config
SIGNAL_BUFFER_LEN = THERMAL_FPS * 30  # Store 30s of history
ADAPTATION_SECONDS = 20               # Wait 20s before first output
SMOOTH_WINDOW_SEC = 0.5               # 0.5s Median Filter window

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(levelname)s::%(funcName)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



# ------------------------------------------------------------------------------
# Part 2: Real-Time BPM Monitor Logic
# ------------------------------------------------------------------------------

def calculate_rr_from_buffer(signal_buffer, fps):
    """
    Applies Denoising + RealTimeBreathMonitor Logic to the whole buffer
    to find the instantaneous rate of the *last* valid breath.
    """
    if len(signal_buffer) < fps * ADAPTATION_SECONDS:
        return None  # Adaptation Period

    # 1. Convert to Numpy
    raw_sig = np.array(signal_buffer)

    # 2. Denoising (Median Filter)
    # Window size must be odd. 0.5s * 25fps ~ 13 frames.
    k_size = int(fps * SMOOTH_WINDOW_SEC)
    if k_size % 2 == 0: k_size += 1

    # Scipy medfilt is best for binary/step noise
    clean_sig = scipy.signal.medfilt(raw_sig, kernel_size=k_size)

    # 3. Binarize (Hysteresis-like thresholding)
    # Since we smoothed, a simple 0.5 cut is usually fine,
    # but we can simulate the state machine logic by iterating.

    # Let's find "Rising Edges" (0 -> 1)
    # Using 0.5 threshold on the smoothed signal
    binary_sig = (clean_sig > 0.5).astype(int)
    diff_sig = np.diff(binary_sig)
    rising_indices = np.where(diff_sig == 1)[0]

    if len(rising_indices) < 2:
        return None # Not enough breaths yet

    # 4. Filter Edges (Refractory Period Logic)
    # We must ensure the gap between edges is physiologically valid (>0.8s)
    min_dist = int(fps * 0.8)

    valid_edges = [rising_indices[0]]
    for i in range(1, len(rising_indices)):
        curr = rising_indices[i]
        prev = valid_edges[-1]
        if (curr - prev) > min_dist:
            valid_edges.append(curr)

    if len(valid_edges) < 2:
        return None

    # 5. Calculate Instantaneous Rate from LAST interval
    last_edge = valid_edges[-1]
    prev_edge = valid_edges[-2]

    frames_diff = last_edge - prev_edge
    duration_sec = frames_diff / fps
    instant_bpm = 60.0 / duration_sec

    return instant_bpm


# ------------------------------------------------------------------------------
# Part 3: Processes
# ------------------------------------------------------------------------------

class MyRRCameraProcess(Process):
    """
    Device Process: Handles Camera, YOLO, and writes RESIZED (72x72) crops to shared mem.
    """

    def __init__(self, shared_mem_array, val_w, val_h, event_lock, stop_flag,
                 thermal_w=THERMAL_FRAME_WIDTH, thermal_h=THERMAL_FRAME_HEIGHT):
        super().__init__()
        self.daemon = True
        self.shared_mem_array = shared_mem_array
        self.val_w = val_w
        self.val_h = val_h
        self.event_lock = event_lock
        self.stop_flag = stop_flag
        self.thermal_w = thermal_w
        self.thermal_h = thermal_h
        self.face_detect_freq = THERMAL_FPS

        # Sockets
        self.socket_reader_port_yolo = 30000
        self.socket_writer_port_yolo = 30001
        self.socket_reader_port_cmd = 30002
        self.socket_writer_port_cmd = 30003

        self.rect = ((0, 0), (1, 1))
        self.kpt = [[0, 0] for _ in range(5)]
        self.face_size = [1, 1]

    def run(self):
        print("[CameraProcess] Starting...")
        self.face_detect_model = MyYoloPoseModel()
        self.setup_sockets()
        print("[CameraProcess] Sockets Connected.")

        # Send Start Command
        self.send_cmd(f"2:3,{THERMAL_FPS * 60}")
        cnt = 0

        try:
            while not self.stop_flag.value:
                # Receive Image
                length_bytes = self.recvall(self.conn_reader_yolo, 16)
                if not length_bytes: break
                length = int(length_bytes)
                stringData = self.recvall(self.conn_reader_yolo, length)
                if not stringData: break

                bgr_img = cv2.imdecode(np.frombuffer(base64.b64decode(stringData), np.uint8), cv2.IMREAD_COLOR)

                # YOLO Detect
                if cnt % self.face_detect_freq == 0:
                    n_rect, n_kpt, n_face_w, n_face_h = self.face_detect_model.detect(bgr_img)
                    if len(n_rect) != 0:
                        self.rect = ((round(n_rect[0] * self.thermal_w), round(n_rect[1] * self.thermal_h)),
                                     (round(n_rect[2] * self.thermal_w), round(n_rect[3] * self.thermal_h)))
                        self.face_size = [n_face_w * self.thermal_w, n_face_h * self.thermal_h]
                        for i, p in enumerate(n_kpt):
                            self.kpt[i][0] = round(p[0] * self.thermal_w)
                            self.kpt[i][1] = round(p[1] * self.thermal_h)
                        self.send_feedback(n_rect, n_kpt, n_face_w, n_face_h)

                # Crop Nose ROI
                half_nose_w = self.face_size[0] * 0.14
                nose_top = self.face_size[1] * 0.01
                nose_bot = self.face_size[1] * 0.16
                nx, ny = self.kpt[2][0], self.kpt[2][1]

                x1 = int(np.clip(nx - half_nose_w, 0, self.thermal_w))
                y1 = int(np.clip(ny - nose_top, 0, self.thermal_h))
                x2 = int(np.clip(nx + half_nose_w, 0, self.thermal_w))
                y2 = int(np.clip(ny + nose_bot, 0, self.thermal_h))

                nose_frame = bgr_img[y1:y2, x1:x2]

                # Write to Shared Memory
                if nose_frame.size > 0:
                    # RESIZE TO 72x72 HERE for Model Efficiency
                    nose_resized = cv2.resize(nose_frame, (MODEL_INPUT_W, MODEL_INPUT_H))

                    flat_nose = nose_resized.flatten()
                    with self.val_w.get_lock():
                        self.val_h.value = MODEL_INPUT_H
                        self.val_w.value = MODEL_INPUT_W

                    limit = len(self.shared_mem_array)
                    if flat_nose.size <= limit:
                        self.shared_mem_array[:flat_nose.size] = flat_nose
                        self.event_lock.set()

                cnt += 1
        except Exception as e:
            print(f"[CameraProcess] Error: {e}")
        finally:
            self.cleanup()

    def setup_sockets(self):
        # ... (Same as original) ...
        self.server_address = "0.0.0.0"
        self.sk_writer_yolo = socket.socket()
        self.sk_reader_yolo = socket.socket()
        self.sk_reader_yolo.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sk_writer_yolo.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sk_reader_yolo.bind((self.server_address, self.socket_reader_port_yolo))
        self.sk_writer_yolo.bind((self.server_address, self.socket_writer_port_yolo))
        self.sk_reader_yolo.listen(1)
        self.sk_writer_yolo.listen(1)

        self.sk_writer_cmd = socket.socket()
        self.sk_reader_cmd = socket.socket()
        self.sk_reader_cmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sk_writer_cmd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sk_reader_cmd.bind((self.server_address, self.socket_reader_port_cmd))
        self.sk_writer_cmd.bind((self.server_address, self.socket_writer_port_cmd))
        self.sk_reader_cmd.listen(1)
        self.sk_writer_cmd.listen(1)

        self.conn_reader_yolo, _ = self.sk_reader_yolo.accept()
        self.conn_writer_yolo, _ = self.sk_writer_yolo.accept()
        self.conn_reader_cmd, _ = self.sk_reader_cmd.accept()
        self.conn_writer_cmd, _ = self.sk_writer_cmd.accept()

    def recvall(self, sock, count):
        buf = b''
        while count:
            try:
                newbuf = sock.recv(count)
                if not newbuf: return None
                buf += newbuf
                count -= len(newbuf)
            except: return None
        return buf

    def send_cmd(self, raw_cmd):
        try: self.conn_writer_cmd.sendall(f"{raw_cmd}\n".encode('utf-8'))
        except: pass

    def send_feedback(self, rect, kpt, w, h):
        try:
            d = {"rect": ",".join(map(str, rect.flatten())), "kpt": ",".join(map(str, kpt.flatten())), "w": w, "h": h}
            json_str = json.dumps(d).encode('utf-8')
            self.conn_writer_yolo.sendall(len(json_str).to_bytes(4, 'big') + json_str)
        except: pass

    def cleanup(self):
        try:
            if self.conn_reader_yolo: self.conn_reader_yolo.close()
            if self.conn_writer_yolo: self.conn_writer_yolo.close()
            if self.sk_reader_yolo: self.sk_reader_yolo.close()
            if self.sk_writer_yolo: self.sk_writer_yolo.close()
            if self.conn_reader_cmd: self.conn_reader_cmd.close()
            if self.conn_writer_cmd: self.conn_writer_cmd.close()
            if self.sk_reader_cmd: self.sk_reader_cmd.close()
            if self.sk_writer_cmd: self.sk_writer_cmd.close()
        except: pass


class MyRRModelProcess(Process):
    """
    Model Process:
    1. frame_buffer (160 frames) -> Model Input
    2. signal_buffer (30s) -> RR Calculation Context
    3. Runs Model every FPS frames (1 sec)
    4. Runs RR Logic on signal_buffer every 2*FPS frames (2 sec)
    """

    def __init__(self, shared_mem_array, val_w, val_h, event_lock, stop_flag,
                 result_queue):
        super().__init__()
        self.daemon = True
        self.shared_mem_array = shared_mem_array
        self.val_w = val_w
        self.val_h = val_h
        self.event_lock = event_lock
        self.stop_flag = stop_flag
        self.result_queue = result_queue

        # Config
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fps = THERMAL_FPS

        # Normalization
        self.mean_t = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std_t = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    def run(self):
        print(f"[ModelProcess] Starting on {self.device_str}...")

        # 1. Load Model
        try:
            self.model = ThermRNet(in_channels=3, base_dim=64, num_classes=2)
            if self.device_str == 'cuda':
                state_dict = torch.load(MODEL_PATH)['model_state_dict']
                self.model.load_state_dict(state_dict)
                self.model.cuda()
            else:
                state_dict = torch.load(MODEL_PATH, map_location='cpu')['model_state_dict']
                self.model.load_state_dict(state_dict)
            self.model.eval()
            print("[ModelProcess] ThermRNet Loaded.")
        except Exception as e:
            print(f"[ModelProcess] CRITICAL: Failed to load model. Error: {e}")

        # 2. Initialize Buffers
        self.frame_buffer = deque(maxlen=MODEL_SEQ_LEN)   # Latest 160 video frames
        self.signal_buffer = deque(maxlen=SIGNAL_BUFFER_LEN) # Latest 30s of probabilities

        cnt = 0

        while not self.stop_flag.value:
            if self.event_lock.wait(timeout=1.0):
                self.event_lock.clear()
                try:
                    w = self.val_w.value
                    h = self.val_h.value
                    if w == MODEL_INPUT_W and h == MODEL_INPUT_H:
                        size = w * h * 3
                        raw_data = self.shared_mem_array[:size]
                        nose_frame = np.array(raw_data, dtype=np.uint8).reshape((h, w, 3))

                        # Preprocessing
                        frame_t = torch.from_numpy(nose_frame).permute(2, 0, 1).float() / 255.0
                        frame_t = (frame_t - self.mean_t) / self.std_t

                        self.frame_buffer.append(frame_t)
                        cnt += 1

                        # --- A. RUN MODEL (Every FPS frames = 1 sec) ---
                        if cnt % self.fps == 0 and len(self.frame_buffer) == MODEL_SEQ_LEN:
                            self.run_inference_and_update_buffer()

                        # --- B. CALC & SEND RR (Every 2*FPS frames = 2 sec) ---
                        if cnt % (2 * self.fps) == 0:
                            bpm = calculate_rr_from_buffer(self.signal_buffer, self.fps)
                            if bpm is not None:
                                self.send_rr(bpm)

                except Exception as e:
                    print(f"[ModelProcess] Error: {e}")

    def run_inference_and_update_buffer(self):
        try:
            with torch.no_grad():
                # Prepare Batch: (1, 3, 160, 72, 72)
                input_seq = torch.stack(list(self.frame_buffer), dim=0).permute(1, 0, 2, 3)
                input_tensor = input_seq.unsqueeze(0).to(self.device_str)

                # Inference
                logits = self.model(input_tensor) # (1, 2, 160)
                probs = F.softmax(logits, dim=1)  # (1, 2, 160)

                # Extract 'Inhale' probabilities (Class 1)
                inhale_seq = probs[0, 1, :].cpu().numpy().tolist()

                # Extract NEW data (last FPS frames)
                # Since we run every FPS frames, these are strictly new data points
                new_predictions = inhale_seq[-self.fps:]

                # Add to signal buffer for RR calc
                self.signal_buffer.extend(new_predictions)

        except Exception as e:
            print(f"[ModelProcess] Inference Fail: {e}")

    def send_rr(self, bpm):
        try:
            if not self.result_queue.full():
                self.result_queue.put(bpm)
            else:
                try: self.result_queue.get_nowait()
                except: pass
                self.result_queue.put(bpm)
        except Exception as e:
            print(f"[ModelProcess] RR Send Fail: {e}")

class RRDetectorRealTime:
    def __init__(self):
        self.stop_flag = Value('i', 0)
        self.event_lock = Event()
        self.result_queue = Queue(maxsize=10)

        # Buffer for 72x72x3 image is small (~15KB)
        # We keep buffer size large enough just in case
        max_size = THERMAL_FRAME_WIDTH * THERMAL_FRAME_HEIGHT * 3
        self.shared_mem_array = Array('B', max_size)
        self.val_w = Value('i', 0)
        self.val_h = Value('i', 0)

        self.camera_process = None
        self.model_process = None

    def start(self):
        try: multiprocessing.set_start_method('spawn', force=True)
        except: pass

        self.stop_flag.value = 0
        self.camera_process = MyRRCameraProcess(self.shared_mem_array, self.val_w, self.val_h, self.event_lock,
                                                self.stop_flag)
        self.model_process = MyRRModelProcess(self.shared_mem_array, self.val_w, self.val_h, self.event_lock,
                                              self.stop_flag, self.result_queue)

        self.camera_process.start()
        self.model_process.start()
        print("RRDetectorRealTime Started.")

    def stop(self):
        self.stop_flag.value = 1
        if self.camera_process: self.camera_process.join()
        if self.model_process: self.model_process.join()
        print("RRDetectorRealTime Stopped.")

    def get_rr(self, timeout=None):
        return self.result_queue.get(block=True, timeout=timeout)


if __name__ == '__main__':
    detector = RRDetectorRealTime()
    try:
        detector.start()
        print("Main Process: Waiting for Instantaneous BPM...")
        while True:
            bpm = detector.get_rr()
            print(f"Current BPM: {bpm:.1f}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.stop()