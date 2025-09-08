# Real-Time-Driver-Drowsiness-and-Attention-Monitoring-
# AIS-184 Driver Drowsiness & Attention Warning System

⚠ **DISCLAIMER**  
This is a **research/demo** implementation inspired by **AIS-184 (Driver Drowsiness and Attention Warning Systems for M, N2, N3 vehicles)**.  
It is **NOT certified** and must not be used as a safety device in real driving conditions.  

This project demonstrates how driver drowsiness and inattention can be approximated using a **standard laptop camera** with **computer vision**.

---

## Features
-  **Eye Aspect Ratio (EAR)** → Blink & eye-closure detection  
-  **PERCLOS** → Fraction of time eyes are closed (over 60s sliding window)  
-  **Yawn detection** → Using Mouth Aspect Ratio (MAR)  
-  **Head-pose estimation** → Approximate gaze & off-road glance timing via `solvePnP`  
-  **Two-stage warnings**  
  - Level 1 → Visual warning  
  - Level 2 → Audible warning (optional)  
-  **Configurable thresholds & windows** (via CLI flags or config)  
-  **Real-time visualization** with OpenCV  

---

##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Real-Time-Driver-Drowsiness-and-Attention-Monitoring.git
cd Real-Time-Driver-Drowsiness-and-Attention-Monitoring


Python \textbf{3.9+} is recommended. To install the required Python packages, run:


1. pip install opencv-python mediapipe numpy playsound==1.2.2


\noindent On Linux systems, you may also need to install the following dependency:


2. sudo apt-get install libasound2


Run the following command:


3. python DDAWS.py --camera 0



You can run with custom EAR, MAR, and PERCLOS thresholds as follows:

python DDAWS.py --camera 0 --ear 0.25 --mar 0.7 --perclos-w 90


