# 📘 OpenCV Week 2 Report - Multi-Stream RTSP Viewer with Motion and Tamper Detection

## 👨‍💻 Task Overview

The Week 2 assignment involved building a **real-time multi-stream video viewer** using OpenCV, capable of:
- Displaying **4 RTSP streams** in a 2x2 grid using multithreading.
- Performing **real-time motion detection**.
- Detecting **camera tampering or compromise** (blur, cover, or laser).
- Rendering overlays like “Motion Detected” or “Camera Compromised”.

---

## 📚 Learning Resources

| Topic | Source |
|-------|--------|
| RTSP Streaming & `cv2.VideoCapture` | [OpenCV Docs](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) |
| Threading in Python | [Python Threading Docs](https://docs.python.org/3/library/threading.html) |
| Motion Detection (MOG2) | [OpenCV Tutorial](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) |
| Camera Tamper Detection | StackOverflow threads, Laplacian for blur, histogram analysis for coverage |
| Video Grid Display | NumPy `hstack`, `vstack` |

---

## 🛠️ Implementation Summary

### ✅ Multi-Stream Viewer
- Created a `VideoStreamThread` class using `threading.Thread`.
- Used `cv2.VideoCapture()` to stream video from 4 `.mp4` URLs (mocking RTSP feeds).
- Applied FFMPEG backend for improved compatibility.

### ✅ Video Display
- Used `cv2.resize()` to normalize all frames to 640x480.
- Combined streams into a 2x2 grid using `np.hstack()` and `np.vstack()`.
- Overlayed stream indices and status for debugging.

### ✅ Motion Detection
- Used `cv2.createBackgroundSubtractorMOG2()` for each stream.
- Checked pixel change threshold to detect movement.
- Displayed “Motion Detected” on screen with contrasting background.

### ✅ Camera Integrity Check
- **Blur Detection**: Used Laplacian variance; threshold < 100 indicates blur.
- **Cover Detection**: Used histogram uniformity; >95% same color = covered.
- If detected, overlayed “Camera Compromised” in red box on the frame.

---

## 🧪 Experiments & Debugging

### 🔁 Frame Looping for `.mp4`
Some `.mp4` streams stopped after finishing. Used:
```python
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
