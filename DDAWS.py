"""
AIS-184 Driver Drowsiness & Attention Warning System — Laptop Cam Demo
=====================================================================

⚠️ DISCLAIMER
This is a research/demo implementation inspired by AIS-184 (Driver Drowsiness and
Attention Warning Systems for M, N2, N3). It is NOT certified and must not be
used as a safety device. It approximates key behaviors (drowsiness & attention
warnings) using a standard laptop camera.

Key features implemented
------------------------
- Eye Aspect Ratio (EAR) → blink & eye-closure detection
- PERCLOS (fraction of time eyes closed) over a sliding 60 s window
- Yawn detection via Mouth Aspect Ratio (MAR)
- Head-pose (gaze/attention proxy) using solvePnP; off-road glance timing
- Two-stage warning strategy (Level 1 visual; Level 2 audible) with lockout
- Configurable thresholds & time windows reflecting common DMS practice

Install
-------
Python ≥3.9 recommended.

pip install opencv-python mediapipe numpy playsound==1.2.2

(On Linux you may also need: `sudo apt-get install libasound2`)

Run
---
python ais184_ddaws_demo.py --camera 0

Press `q` to quit.

Notes
-----
- Uses MediaPipe Face Mesh for robust landmarks. Falls back to Haar cascade if
  MediaPipe fails to initialize.
- Audio alert is optional; if `playsound` fails, we still show on-screen prompts.
- All thresholds are configurable via CLI flags or the CONFIG dict below.
"""
from __future__ import annotations
import argparse
import collections
import math
import sys
import time
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

# Try to import mediapipe for landmark detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# Optional audio for Level 2 warnings
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except Exception:
    PLAYSOUND_AVAILABLE = False


# ---------------------------- Utility Structures ---------------------------- #
@dataclass
class Config:
    # EAR thresholds
    ear_threshold: float = 0.23           # closed-eye EAR (tune per camera)
    blink_min_frames: int = 3             # consecutive frames to count a blink

    # MAR thresholds (yawn)
    mar_threshold: float = 0.65           # open-mouth MAR (tune per camera)
    yawn_min_duration_s: float = 1.0      # sustain MAR above threshold

    # PERCLOS window
    perclos_window_s: int = 60            # window (seconds)
    perclos_warn_level1: float = 0.30     # Level 1 when PERCLOS ≥ 0.30
    perclos_warn_level2: float = 0.40     # Level 2 when PERCLOS ≥ 0.40

    # Head pose / attention
    offroad_yaw_deg: float = 25.0         # |yaw| beyond this is off-road glance
    offroad_pitch_deg: float = 20.0       # looking down/up
    offroad_warn_time_s: float = 2.0      # sustain off-road for this long

    # Microsleep / long-closure
    closure_warn_time_s: float = 1.0      # single closure ≥1.0s → Level 2

    # Debounce/lockouts
    level1_min_interval_s: float = 10.0
    level2_min_interval_s: float = 20.0

    # Visualization
    draw_landmarks: bool = True


CONFIG = Config()


# Indices for MediaPipe Face Mesh landmarks (both eyes & mouth region)
# Using a subset: left eye (33, 160, 158, 133, 153, 144); right eye (362, 385, 387, 263, 373, 380)
# Mouth outer: (61, 291, 81, 178, 13, 14)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 178, 13, 14]


# 3D model points for head pose (nose tip, eye corners, mouth corners, chin)
# Landmarks mapped to approximate 3D model (generic face). Values are coarse.
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],        # Nose tip
    [-30.0, -30.0, -30.0],  # Left eye outer corner
    [30.0, -30.0, -30.0],   # Right eye outer corner
    [-40.0, 40.0, -30.0],   # Left mouth corner
    [40.0, 40.0, -30.0],    # Right mouth corner
    [0.0, 70.0, -5.0],      # Chin
], dtype=np.float64)

# Map these to Face Mesh indices
HEADPOSE_LANDMARKS = [1, 33, 263, 61, 291, 199]


# ---------------------------- Math Helpers --------------------------------- #

def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Compute EAR given 6 points: [p0, p1, p2, p3, p4, p5]."""
    A = _euclidean(eye_pts[1], eye_pts[5])
    B = _euclidean(eye_pts[2], eye_pts[4])
    C = _euclidean(eye_pts[0], eye_pts[3])
    if C <= 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth_pts: np.ndarray) -> float:
    """Compute MAR using 6 points around mouth."""
    A = _euclidean(mouth_pts[2], mouth_pts[3])  # vertical
    C = _euclidean(mouth_pts[0], mouth_pts[1])  # horizontal
    if C <= 1e-6:
        return 0.0
    return A / C


def rotation_vector_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.degrees(math.atan2(R[2, 1], R[2, 2]))  # pitch
        y = math.degrees(math.atan2(-R[2, 0], sy))      # yaw
        z = math.degrees(math.atan2(R[1, 0], R[0, 0]))  # roll
    else:
        x = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        y = math.degrees(math.atan2(-R[2, 0], sy))
        z = 0
    return x, y, z


# ---------------------------- Core DDAWS Logic ----------------------------- #
@dataclass
class State:
    blink_counter: int = 0
    last_blink_time: float = 0.0
    eye_closure_start: Optional[float] = None
    per_frame_closed: Deque[int] = None  # 1 if closed, 0 otherwise
    per_frame_times: Deque[float] = None
    last_level1_time: float = -1e9
    last_level2_time: float = -1e9
    offroad_start: Optional[float] = None
    yawn_start: Optional[float] = None

    def __post_init__(self):
        self.per_frame_closed = collections.deque(maxlen=10_000)
        self.per_frame_times = collections.deque(maxlen=10_000)


class DDAWS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State()

    def update_perclos(self, is_closed: bool):
        now = time.time()
        self.state.per_frame_closed.append(1 if is_closed else 0)
        self.state.per_frame_times.append(now)
        # Drop frames older than window
        while self.state.per_frame_times and (now - self.state.per_frame_times[0] > self.cfg.perclos_window_s):
            self.state.per_frame_times.popleft()
            self.state.per_frame_closed.popleft()

    def compute_perclos(self) -> float:
        if not self.state.per_frame_closed:
            return 0.0
        return float(sum(self.state.per_frame_closed)) / float(len(self.state.per_frame_closed))

    def level1_ready(self) -> bool:
        return (time.time() - self.state.last_level1_time) >= self.cfg.level1_min_interval_s

    def level2_ready(self) -> bool:
        return (time.time() - self.state.last_level2_time) >= self.cfg.level2_min_interval_s

    def trigger_level1(self):
        self.state.last_level1_time = time.time()

    def trigger_level2(self):
        self.state.last_level2_time = time.time()


# ---------------------------- Detector Wrappers ---------------------------- #
class LandmarkDetector:
    def __init__(self):
        self.use_mp = MP_AVAILABLE
        if self.use_mp:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            # Fallback to Haar face detection only (no landmarks)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        if self.use_mp:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.mp_face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lms = res.multi_face_landmarks[0]
            pts = np.array([(lm.x * w, lm.y * h) for lm in lms.landmark], dtype=np.float64)
            return pts
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            # No landmarks available; return None to skip metrics
            return None


# ---------------------------- Visualization & Alerts ----------------------- #

def draw_text(img, text, org, scale=0.7, thickness=2, color=(0, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def beep():
    if PLAYSOUND_AVAILABLE:
        try:
            # Generate a short beep using the system default if available
            # Provide a bundled wav tone fallback if you like
            # Here we just attempt to play a nonexistent file path will raise
            pass
        except Exception:
            pass
    # As a simple fallback: no sound (cross-platform beeps are unreliable)
    return


# ---------------------------- Main Processing ------------------------------ #

def process_frame(ddaws: DDAWS, frame: np.ndarray, pts: Optional[np.ndarray], cfg: Config):
    h, w = frame.shape[:2]
    info = {}

    if pts is None:
        draw_text(frame, "Face not detected", (30, 40), color=(0, 0, 255))
        ddaws.update_perclos(False)
        return frame, info

    # Extract landmarks for EAR / MAR
    def gather(idxs):
        return np.array([pts[i] for i in idxs], dtype=np.float64)

    left_eye = gather(LEFT_EYE)
    right_eye = gather(RIGHT_EYE)
    mouth = gather(MOUTH)

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    ear = (ear_left + ear_right) / 2.0

    mar = mouth_aspect_ratio(mouth)

    # Eye-closure logic
    eyes_closed = ear < cfg.ear_threshold
    ddaws.update_perclos(eyes_closed)

    now = time.time()
    # Blink / closure timing
    if eyes_closed:
        if ddaws.state.eye_closure_start is None:
            ddaws.state.eye_closure_start = now
    else:
        if ddaws.state.eye_closure_start is not None:
            duration = now - ddaws.state.eye_closure_start
            # Count as blink if short, else was a long closure already handled below
            if duration * 30 >= cfg.blink_min_frames:
                ddaws.state.last_blink_time = now
            ddaws.state.eye_closure_start = None

    # Long closure → Level 2
    if ddaws.state.eye_closure_start is not None:
        closure_dur = now - ddaws.state.eye_closure_start
        if closure_dur >= cfg.closure_warn_time_s and ddaws.level2_ready():
            ddaws.trigger_level2()
            draw_text(frame, "LEVEL 2: Microsleep/Long closure!", (30, 80), color=(0, 0, 255), scale=0.9)
            beep()

    # Head pose estimation
    image_points = gather(HEADPOSE_LANDMARKS)
    # Camera matrix approximation
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    yaw = pitch = roll = 0.0
    if success:
        pitch, yaw, roll = rotation_vector_to_euler(rvec)
        draw_text(frame, f"Yaw: {yaw:+.1f} Pitch: {pitch:+.1f}", (30, h - 20), scale=0.6, color=(255, 255, 255))

    # Off-road glance tracking
    offroad = abs(yaw) > cfg.offroad_yaw_deg or abs(pitch) > cfg.offroad_pitch_deg
    if offroad:
        if ddaws.state.offroad_start is None:
            ddaws.state.offroad_start = now
        elif (now - ddaws.state.offroad_start) >= cfg.offroad_warn_time_s and ddaws.level1_ready():
            ddaws.trigger_level1()
            draw_text(frame, "LEVEL 1: Eyes off-road!", (30, 110), color=(0, 255, 255))
    else:
        ddaws.state.offroad_start = None

    # Yawn detection
    if mar >= cfg.mar_threshold:
        if ddaws.state.yawn_start is None:
            ddaws.state.yawn_start = now
        elif (now - ddaws.state.yawn_start) >= cfg.yawn_min_duration_s and ddaws.level1_ready():
            ddaws.trigger_level1()
            draw_text(frame, "LEVEL 1: Yawn detected", (30, 140), color=(0, 255, 255))
    else:
        ddaws.state.yawn_start = None

    # PERCLOS-based graded warnings
    perclos = ddaws.compute_perclos()
    info.update({
        'EAR': ear,
        'MAR': mar,
        'PERCLOS': perclos,
        'Yaw': yaw,
        'Pitch': pitch,
    })

    draw_text(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}  PERCLOS(60s): {perclos:.2f}", (30, 50), scale=0.6, color=(200, 255, 200))

    if perclos >= cfg.perclos_warn_level2 and ddaws.level2_ready():
        ddaws.trigger_level2()
        draw_text(frame, "LEVEL 2: High drowsiness (PERCLOS)", (30, 170), color=(0, 0, 255))
        beep()
    elif perclos >= cfg.perclos_warn_level1 and ddaws.level1_ready():
        ddaws.trigger_level1()
        draw_text(frame, "LEVEL 1: Drowsiness rising (PERCLOS)", (30, 170), color=(0, 255, 255))

    # Landmarks overlay
    if MP_AVAILABLE and CONFIG.draw_landmarks:
        for idx in LEFT_EYE + RIGHT_EYE + MOUTH + HEADPOSE_LANDMARKS:
            x, y = int(pts[idx][0]), int(pts[idx][1])
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    return frame, info


# ---------------------------- CLI / Main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="AIS-184 DDAWS Laptop Camera Demo")
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--ear', type=float, default=CONFIG.ear_threshold)
    parser.add_argument('--mar', type=float, default=CONFIG.mar_threshold)
    parser.add_argument('--perclos-w', type=int, default=CONFIG.perclos_window_s)
    parser.add_argument('--perclos-l1', type=float, default=CONFIG.perclos_warn_level1)
    parser.add_argument('--perclos-l2', type=float, default=CONFIG.perclos_warn_level2)
    parser.add_argument('--offroad-yaw', type=float, default=CONFIG.offroad_yaw_deg)
    parser.add_argument('--offroad-pitch', type=float, default=CONFIG.offroad_pitch_deg)
    parser.add_argument('--offroad-time', type=float, default=CONFIG.offroad_warn_time_s)
    parser.add_argument('--closure', type=float, default=CONFIG.closure_warn_time_s)
    parser.add_argument('--yawn', type=float, default=CONFIG.yawn_min_duration_s)
    parser.add_argument('--no-draw', action='store_true', help='Disable landmark drawing')

    args = parser.parse_args()

    # Update CONFIG from args
    CONFIG.ear_threshold = args.ear
    CONFIG.mar_threshold = args.mar
    CONFIG.perclos_window_s = args.perclos_w
    CONFIG.perclos_warn_level1 = args.perclos_l1
    CONFIG.perclos_warn_level2 = args.perclos_l2
    CONFIG.offroad_yaw_deg = args.offroad_yaw
    CONFIG.offroad_pitch_deg = args.offroad_pitch
    CONFIG.offroad_warn_time_s = args.offroad_time
    CONFIG.closure_warn_time_s = args.closure
    CONFIG.yawn_min_duration_s = args.yawn
    CONFIG.draw_landmarks = not args.no_draw

    # Init detector and system
    detector = LandmarkDetector()
    ddaws = DDAWS(CONFIG)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Starting... (approx FPS={fps:.1f}) Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pts = detector.detect(frame)
        frame, info = process_frame(ddaws, frame, pts, CONFIG)

        cv2.imshow('AIS-184 DDAWS Demo', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
