<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>📹 OpenCV Week 2 - RTSP Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; background: #f9f9f9; color: #2c3e50; }
    h1, h2, h3, h4 { color: #2c3e50; }
    code, pre { background: #ecf0f1; padding: 4px 8px; border-radius: 4px; font-family: monospace; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { border: 1px solid #bdc3c7; padding: 10px; text-align: left; }
    th { background-color: #34495e; color: white; }
    img { max-width: 100%; border: 1px solid #ccc; }
    hr { border: none; border-top: 1px solid #ccc; margin: 30px 0; }
    ul, ol { margin-left: 20px; }
  </style>
</head>
<body>

<h1>📹 OpenCV Week 2 - Multi‑Stream RTSP Viewer with Motion and Camera Compromise Detection</h1>

<p>This project is the official submission for <strong>Week 2</strong> of the OpenCV real‑time streaming and processing challenge.<br>
The goal is to build a system that displays <strong>four real‑time video streams</strong> (simulated RTSP feeds), detects motion, and identifies whether the camera feed is blurred, covered, or compromised.</p>

<hr>

<h2>📌 Project Description</h2>
<ul>
  <li>Streams video from <strong>four different cameras</strong> in parallel</li>
  <li>Detects <strong>motion</strong> in each camera using background subtraction</li>
  <li>Detects <strong>tampering</strong> (blurred, covered, or laser interference) using image analysis</li>
  <li>Displays the 4 video streams in a <strong>2×2 grid</strong> window</li>
  <li>Shows alerts like <strong>“Motion Detected”</strong> or <strong>“Camera Compromised”</strong> as overlay text</li>
</ul>

<p>This system can be easily adapted for <strong>real RTSP cameras</strong>, surveillance systems, or home security.</p>

<hr>

<h2>🗂️ Project Structure</h2>
<pre>
opencv-week2/
├── code/
│   └── multi_stream_viewer.py       # Main implementation file
├── report/
│   └── week2_report.md              # Detailed explanation and findings
├── diagrams/
│   └── screenshot_output.png        # Screenshot of the 2×2 viewer
└── README.md                        # You are here
</pre>

<hr>

<h2>🔧 Requirements</h2>
<p>You need Python and OpenCV. Install dependencies using pip:</p>
<pre><code>pip install opencv-python numpy</code></pre>

<hr>

<h2>🛠️ How to Run</h2>
<h3>🔁 Step‑by‑Step Instructions</h3>
<ol>
  <li>Clone the Repository or download ZIP:
    <pre><code>git clone https://github.com/&lt;your-username&gt;/opencv-week2.git
cd opencv-week2/code</code></pre>
  </li>
  <li>Run the script:
    <pre><code>python multi_stream_viewer.py</code></pre>
  </li>
</ol>

<hr>

<h2>🔍 Features Explained</h2>

<h3>🎥 1. Multi‑Stream RTSP Viewer (4 Cameras)</h3>
<ul>
  <li>Uses <strong>Python threading</strong> to fetch frames from 4 streams concurrently</li>
  <li>Prevents UI blocking or lag</li>
  <li>Automatically loops <code>.mp4</code> test streams or attempts reconnection for RTSP</li>
</ul>

<h3>🔄 2. Real‑Time Motion Detection</h3>
<ul>
  <li>Utilizes <code>cv2.createBackgroundSubtractorMOG2</code> for motion detection</li>
  <li>If movement is detected, draws a white rectangle with <strong>“Motion Detected”</strong> in black text</li>
</ul>

<h3>🛑 3. Camera Tamper Detection</h3>

<h4>a. Blur Detection</h4>
<ul>
  <li>Uses the Laplacian variance metric</li>
  <li>If sharpness (variance) is below a threshold (e.g., 100), the frame is considered blurred</li>
</ul>

<h4>b. Cover / Laser / Overexposure Detection</h4>
<ul>
  <li>Uses the grayscale histogram</li>
  <li>If 95% or more of pixels are the same value (uniform distribution), then it is considered covered or overexposed</li>
  <li>If compromise is detected, draws a red rectangle with <strong>“Camera Compromised”</strong> overlay</li>
</ul>

<h3>🧱 4. 2×2 Video Grid Display</h3>
<ul>
  <li>Resizes all frames to the same size (e.g., 640×480)</li>
  <li>Combines them into a grid using <code>np.hstack()</code> and <code>np.vstack()</code></li>
  <li>Maintains alignment and consistent layout</li>
</ul>

<hr>

<h2>🖼️ Example Output Screenshot</h2>
<p>Here’s what your viewer will look like:</p>
<blockquote>🔲 Top‑left, Top‑right, Bottom‑left, Bottom‑right<br>✅ Motion detected or compromise warnings may appear overlayed.</blockquote>
<img src="diagrams/screenshot_output.png" alt="Screenshot of 4-camera RTSP viewer with alerts">

<hr>

<h2>🧾 Summary of Methods Used</h2>
<table>
  <tr><th>Task</th><th>Method / Technique</th></tr>
  <tr><td>Multi‑Stream Handling</td><td><code>threading.Thread</code>, thread‑safe capture</td></tr>
  <tr><td>Motion Detection</td><td><code>cv2.createBackgroundSubtractorMOG2</code> + thresholding</td></tr>
  <tr><td>Blur Detection</td><td><code>cv2.Laplacian()</code> variance</td></tr>
  <tr><td>Cover Detection</td><td>Histogram analysis via <code>cv2.calcHist()</code></td></tr>
  <tr><td>Video Display</td><td><code>cv2.resize()</code>, <code>np.hstack()</code>, <code>np.vstack()</code></td></tr>
  <tr><td>Overlay Text & Boxes</td><td><code>cv2.putText()</code>, <code>cv2.rectangle()</code></td></tr>
</table>

<hr>

<h2>🧠 Learnings & Insights</h2>
<ul>
  <li>RTSP / video streams can be <strong>unstable</strong>; reconnection logic is important</li>
  <li>Motion detection may have <strong>false positives</strong>, so tuning thresholds is important</li>
  <li>Tamper detection (blur, cover) is essential in surveillance setups</li>
  <li>Multithreading with OpenCV is effective for handling multiple streams in real time</li>
</ul>

<hr>

<h2>🧰 Troubleshooting Guide</h2>
<table>
  <tr><th>Problem</th><th>Likely Cause</th><th>Solution</th></tr>
  <tr><td>Only 2 out of 4 videos show</td><td>URLs invalid or stream failing</td><td>Check URL or internet, enable logs</td></tr>
  <tr><td>One window black or blank</td><td>That stream failed reading frames</td><td>Visual placeholder is shown for diagnostics</td></tr>
  <tr><td>Motion always triggered</td><td>Too low threshold</td><td>Increase motion threshold or filter noise</td></tr>
  <tr><td>Frequent “Camera Compromised” alerts</td><td>Too strict thresholds</td><td>Relax blur/histogram thresholds or conditions</td></tr>
</table>

</body>
</html>
