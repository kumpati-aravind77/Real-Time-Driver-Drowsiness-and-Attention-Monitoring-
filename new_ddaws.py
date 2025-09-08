#!/usr/bin/env python3
"""
AIS-184 Complete DDAWS (single-file) — audio-enabled (Linux reliable):
 - All L1 and L2 warnings now play sound using aplay
 - Bottom banners do not cover face
"""

from __future__ import annotations
import argparse
import collections
import csv
import math
import os
import sys
import threading
import subprocess
import time
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

# MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# -------------------- Config --------------------
@dataclass
class Config:
    ear_threshold: float = 0.25
    blink_min_frames: int = 3
    mar_threshold: float = 0.40
    yawn_min_duration_s: float = 1.0
    perclos_window_s: int = 60
    perclos_warn_level1: float = 0.30
    perclos_warn_level2: float = 0.45
    offroad_yaw_deg: float = 25.0
    offroad_pitch_deg: float = 20.0
    offroad_warn_time_s: float = 1.5
    closure_warn_time_s: float = 1.0
    level1_min_interval_s: float = 3.0
    level2_min_interval_s: float = 6.0
    warning_display_s: float = 3.0
    draw_landmarks: bool = True

CONFIG = Config()

# -------------------- Landmarks --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14]
HEADPOSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],
    [-30.0, -30.0, -30.0],
    [30.0, -30.0, -30.0],
    [-40.0, 40.0, -30.0],
    [40.0, 40.0, -30.0],
    [0.0, 70.0, -5.0],
], dtype=np.float64)

# -------------------- Helpers --------------------
def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    A = _euclid(eye_pts[1], eye_pts[5])
    B = _euclid(eye_pts[2], eye_pts[4])
    C = _euclid(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0

def mouth_aspect_ratio(mouth_pts: np.ndarray) -> float:
    A = _euclid(mouth_pts[2], mouth_pts[3])
    C = _euclid(mouth_pts[0], mouth_pts[1])
    return A / C if C > 1e-6 else 0.0

def rotation_vector_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.degrees(math.atan2(R[2,1], R[2,2]))
        y = math.degrees(math.atan2(-R[2,0], sy))
        z = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        x = math.degrees(math.atan2(-R[1,2], R[1,1]))
        y = math.degrees(math.atan2(-R[2,0], sy))
        z = 0
    return x, y, z

def draw_text_with_bg(img, text, org, color=(255,255,255), bg=(0,0,0), scale=0.7, thickness=2, alpha=0.6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x-6, y-6), (x+tw+6, y+th+6), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x, y+th), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# -------------------- Audio --------------------
BEEP_PATH = "/home/s186/Music/DDAWS/beep.wav"

def play_beep():
    if os.path.isfile(BEEP_PATH):
        threading.Thread(target=lambda: subprocess.run(["aplay", "-q", BEEP_PATH], check=False), daemon=True).start()
    else:
        print("[Audio] Beep file not found:", BEEP_PATH)

# -------------------- Warnings --------------------
class WarningManager:
    def __init__(self, display_s: float = 3.0):
        self.msg: Optional[str] = None
        self.color: Tuple[int,int,int] = (0,255,255)
        self.expiry: float = 0.0
        self.display_s = display_s

    def trigger(self, msg: str, color: Tuple[int,int,int], sound: bool=True):
        self.msg = msg
        self.color = color
        self.expiry = time.time() + self.display_s
        if sound:
            play_beep()

    def draw(self, frame):
        if self.msg and time.time() < self.expiry:
            h, w = frame.shape[:2]
            banner_h = 70
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            (tw, th), _ = cv2.getTextSize(self.msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            x = max(10, (w - tw)//2)
            y_top = h - banner_h
            y = y_top + (banner_h - th)//2 + th
            cv2.putText(frame, self.msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color, 3, cv2.LINE_AA)

# -------------------- DDAWS Core --------------------
class DDAWS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = self._init_state()
        self.warn_mgr = WarningManager(display_s=cfg.warning_display_s)
        self.ear_buf = collections.deque(maxlen=5)
        self.mar_buf = collections.deque(maxlen=5)
        self.log_rows = []

    def _init_state(self):
        s = lambda: None
        s.last_blink_time = 0.0
        s.eye_closure_start: Optional[float] = None
        s.per_frame_closed: Deque[int] = collections.deque()
        s.per_frame_times: Deque[float] = collections.deque()
        s.last_level1_time = -1e9
        s.last_level2_time = -1e9
        s.offroad_start: Optional[float] = None
        s.yawn_start: Optional[float] = None
        return s

    def update_perclos(self, closed: bool):
        now = time.time()
        self.state.per_frame_closed.append(1 if closed else 0)
        self.state.per_frame_times.append(now)
        while self.state.per_frame_times and (now - self.state.per_frame_times[0] > self.cfg.perclos_window_s):
            self.state.per_frame_times.popleft()
            self.state.per_frame_closed.popleft()

    def compute_perclos(self) -> float:
        if not self.state.per_frame_closed: return 0.0
        return float(sum(self.state.per_frame_closed)) / float(len(self.state.per_frame_closed))

    def level1_ready(self) -> bool:
        return (time.time() - self.state.last_level1_time) >= self.cfg.level1_min_interval_s

    def level2_ready(self) -> bool:
        return (time.time() - self.state.last_level2_time) >= self.cfg.level2_min_interval_s

    def trigger_level1(self, reason: str):
        self.state.last_level1_time = time.time()
        self.warn_mgr.trigger(reason, (0,255,255), sound=True)
        self._log("L1", reason)

    def trigger_level2(self, reason: str):
        self.state.last_level2_time = time.time()
        self.warn_mgr.trigger(reason, (0,0,255), sound=True)
        self._log("L2", reason)

    def _log(self, level: str, reason: str):
        row = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "level": level,
            "reason": reason,
            "EAR": float(np.mean(self.ear_buf)) if self.ear_buf else None,
            "MAR": float(np.mean(self.mar_buf)) if self.mar_buf else None,
            "PERCLOS": self.compute_perclos()
        }
        self.log_rows.append(row)
        print("[EVENT]", row)

    def save_log(self, path="ddaws_events.csv"):
        if not self.log_rows: return
        header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.log_rows[0].keys()))
            if header: writer.writeheader()
            writer.writerows(self.log_rows)
        print(f"[DDAWS] Saved {len(self.log_rows)} events to {path}")
        self.log_rows = []

    def process(self, frame: np.ndarray, pts: Optional[np.ndarray]) -> Tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]
        info = {}
        now = time.time()

        if pts is None:
            draw_text_with_bg(frame, "Face not detected", (12, 18), color=(0,0,255), bg=(0,0,0), scale=0.6)
            self.update_perclos(False)
            self.warn_mgr.draw(frame)
            return frame, info

        def g(idxs):
            return np.array([pts[i] for i in idxs], dtype=np.float64)

        left = g(LEFT_EYE)
        right = g(RIGHT_EYE)
        mouth = g(MOUTH)

        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        self.ear_buf.append(ear)
        self.mar_buf.append(mar)
        ear_s = float(np.mean(self.ear_buf))
        mar_s = float(np.mean(self.mar_buf))

        closed = ear_s < self.cfg.ear_threshold
        self.update_perclos(closed)

        if closed:
            if self.state.eye_closure_start is None:
                self.state.eye_closure_start = now
        else:
            if self.state.eye_closure_start is not None:
                dur = now - self.state.eye_closure_start
                if dur * 30 >= self.cfg.blink_min_frames:
                    self.state.last_blink_time = now
                self.state.eye_closure_start = None

        if self.state.eye_closure_start is not None:
            closure_dur = now - self.state.eye_closure_start
            if closure_dur >= self.cfg.closure_warn_time_s and self.level2_ready():
                self.trigger_level2("Microsleep / Long eye closure")

        yaw = pitch = roll = 0.0
        try:
            image_points = g(HEADPOSE_LANDMARKS)
            focal_length = w
            center = (w/2.0, h/2.0)
            camera_matrix = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]], dtype=np.float64)
            dist_coeffs = np.zeros((4,1))
            success, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            if success:
                pitch, yaw, roll = rotation_vector_to_euler(rvec)
                draw_text_with_bg(frame, f"Yaw:{yaw:+.1f} Pitch:{pitch:+.1f}", (12,48), scale=0.6, color=(255,255,255), bg=(0,0,0), alpha=0.45)
        except: pass

        offroad = (abs(yaw) > self.cfg.offroad_yaw_deg) or (abs(pitch) > self.cfg.offroad_pitch_deg)
        if offroad:
            if self.state.offroad_start is None:
                self.state.offroad_start = now
            elif (now - self.state.offroad_start) >= self.cfg.offroad_warn_time_s and self.level1_ready():
                self.trigger_level1("Eyes off-road / Gaze deviation")
        else:
            self.state.offroad_start = None

        if mar >= self.cfg.mar_threshold:
            if self.state.yawn_start is None:
                self.state.yawn_start = now
            elif (now - self.state.yawn_start) >= self.cfg.yawn_min_duration_s and self.level1_ready():
                self.trigger_level1("Yawn detected")
        else:
            self.state.yawn_start = None

        perclos = self.compute_perclos()
        if perclos >= self.cfg.perclos_warn_level2 and self.level2_ready():
            self.trigger_level2("High drowsiness (PERCLOS)")
        elif perclos >= self.cfg.perclos_warn_level1 and self.level1_ready():
            self.trigger_level1("Drowsiness rising (PERCLOS)")

        draw_text_with_bg(frame, f"EAR:{ear_s:.3f} MAR:{mar_s:.3f} PERCLOS:{perclos:.2f}", (12,18), color=(200,255,200), scale=0.7, bg=(0,0,0), alpha=0.6)

        if self.cfg.draw_landmarks:
            for idx in LEFT_EYE + RIGHT_EYE + MOUTH + HEADPOSE_LANDMARKS:
                x, y = int(pts[idx][0]), int(pts[idx][1])
                cv2.circle(frame, (x,y), 1, (255,255,255), -1)

        self.warn_mgr.draw(frame)
        info.update({"EAR": ear_s, "MAR": mar_s, "PERCLOS": perclos, "Yaw": yaw, "Pitch": pitch})
        return frame, info

# -------------------- Main --------------------
def main():
    CONFIG.ear_threshold = CONFIG.ear_threshold
    CONFIG.mar_threshold = CONFIG.mar_threshold
    CONFIG.perclos_warn_level1 = CONFIG.perclos_warn_level1
    CONFIG.perclos_warn_level2 = CONFIG.perclos_warn_level2

    cap = cv2.VideoCapture(0)  # default camera
    dda = DDAWS(CONFIG)

    if not MP_AVAILABLE:
        print("MediaPipe not available, exiting.")
        return

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            pts = None
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                pts = [(int(p.x*frame.shape[1]), int(p.y*frame.shape[0])) for p in lm.landmark]
            frame, info = dda.process(frame, pts)
            cv2.imshow("DDAWS", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    dda.save_log()

if __name__ == "__main__":
    main()


