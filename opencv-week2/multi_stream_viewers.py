import cv2
import numpy as np
import threading

# List of video URLs (you can replace with RTSP streams later)
video_urls = [
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
]

# Thread class for video stream
class VideoStreamThread(threading.Thread):
    def __init__(self, url, index):
        super().__init__()
        self.url = url
        self.index = index
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # Force FFMPEG backend
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                print(f"[Stream {self.index}] Frame read failed. Attempting to loop or reconnect...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.ret else None

    def stop(self):
        self.running = False
        self.cap.release()

# Motion detection
fgbg_list = [cv2.createBackgroundSubtractorMOG2() for _ in range(4)]

def detect_motion(frame, fgbg):
    fgmask = fgbg.apply(frame)
    if np.sum(fgmask) > 5000:
        cv2.rectangle(frame, (10, 10), (250, 50), (255, 255, 255), -1)
        cv2.putText(frame, "Motion Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return frame

# Camera compromise check
def is_blurred(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def is_covered(frame, uniform_thresh=0.95):
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    return np.max(hist) / np.sum(hist) > uniform_thresh

# Initialize all video streams
streams = [VideoStreamThread(url, i) for i, url in enumerate(video_urls)]
for s in streams:
    s.start()

try:
    while True:
        frames = [s.get_frame() for s in streams]
        frames_resized = []

        for i, f in enumerate(frames):
            if f is None:
                # Show placeholder if stream failed
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Stream {i} Unavailable", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames_resized.append(placeholder)
            else:
                # Resize and process
                f = cv2.resize(f, (640, 480))
                f = detect_motion(f, fgbg_list[i])
                if is_blurred(f) or is_covered(f):
                    cv2.rectangle(f, (10, 60), (360, 100), (0, 0, 255), -1)
                    cv2.putText(f, "Camera Compromised", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frames_resized.append(f)

        # Combine into 2x2 grid
        top_row = np.hstack(frames_resized[:2])
        bottom_row = np.hstack(frames_resized[2:])
        grid = np.vstack([top_row, bottom_row])

        # Display
        cv2.imshow("Multi-Video Viewer", grid)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    for s in streams:
        s.stop()
    cv2.destroyAllWindows()
