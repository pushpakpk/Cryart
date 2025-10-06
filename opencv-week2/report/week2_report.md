<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>📘 OpenCV Week 2 Report</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f9f9f9; color: #2c3e50; line-height: 1.6; padding: 20px; }
    h1, h2, h3, h4 { color: #2c3e50; }
    pre, code { background: #ecf0f1; padding: 6px 10px; border-radius: 4px; font-family: monospace; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #bdc3c7; padding: 10px; text-align: left; }
    th { background-color: #34495e; color: white; }
    hr { border: none; border-top: 1px solid #ccc; margin: 30px 0; }
    ul { margin-left: 20px; }
    a { color: #2980b9; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>

<h1>📘 OpenCV Week 2 Report - Multi-Stream RTSP Viewer with Motion and Tamper Detection</h1>

<hr>

<h2>👨‍💻 Task Overview</h2>
<p>The Week 2 assignment involved building a <strong>real-time multi-stream video viewer</strong> using OpenCV, capable of:</p>
<ul>
  <li>Displaying <strong>4 RTSP streams</strong> in a 2x2 grid using multithreading.</li>
  <li>Performing <strong>real-time motion detection</strong>.</li>
  <li>Detecting <strong>camera tampering or compromise</strong> (blur, cover, or laser).</li>
  <li>Rendering overlays like “Motion Detected” or “Camera Compromised”.</li>
</ul>

<hr>

<h2>📚 Learning Resources</h2>
<table>
  <tr><th>Topic</th><th>Source</th></tr>
  <tr>
    <td>RTSP Streaming & <code>cv2.VideoCapture</code></td>
    <td><a href="https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html" target="_blank">OpenCV Docs</a></td>
  </tr>
  <tr>
    <td>Threading in Python</td>
    <td><a href="https://docs.python.org/3/library/threading.html" target="_blank">Python Threading Docs</a></td>
  </tr>
  <tr>
    <td>Motion Detection (MOG2)</td>
    <td><a href="https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html" target="_blank">OpenCV Tutorial</a></td>
  </tr>
  <tr>
    <td>Camera Tamper Detection</td>
    <td>StackOverflow threads, Laplacian for blur, histogram analysis for coverage</td>
  </tr>
  <tr>
    <td>Video Grid Display</td>
    <td>NumPy <code>hstack</code>, <code>vstack</code></td>
  </tr>
</table>

<hr>

<h2>🛠️ Implementation Summary</h2>

<h3>✅ Multi-Stream Viewer</h3>
<ul>
  <li>Created a <code>VideoStreamThread</code> class using <code>threading.Thread</code>.</li>
  <li>Used <code>cv2.VideoCapture()</code> to stream video from 4 <code>.mp4</code> URLs (mocking RTSP feeds).</li>
  <li>Applied FFMPEG backend for improved compatibility.</li>
</ul>

<h3>✅ Video Display</h3>
<ul>
  <li>Used <code>cv2.resize()</code> to normalize all frames to 640x480.</li>
  <li>Combined streams into a 2x2 grid using <code>np.hstack()</code> and <code>np.vstack()</code>.</li>
  <li>Overlayed stream indices and status for debugging.</li>
</ul>

<h3>✅ Motion Detection</h3>
<ul>
  <li>Used <code>cv2.createBackgroundSubtractorMOG2()</code> for each stream.</li>
  <li>Checked pixel change threshold to detect movement.</li>
  <li>Displayed “Motion Detected” on screen with contrasting background.</li>
</ul>

<h3>✅ Camera Integrity Check</h3>
<ul>
  <li><strong>Blur Detection</strong>: Used Laplacian variance; threshold &lt; 100 indicates blur.</li>
  <li><strong>Cover Detection</strong>: Used histogram uniformity; &gt;95% same color = covered.</li>
  <li>If detected, overlayed “Camera Compromised” in red box on the frame.</li>
</ul>

<hr>

<h2>🧪 Experiments & Debugging</h2>

<h3>🔁 Frame Looping for <code>.mp4</code></h3>
<p>Some <code>.mp4</code> streams stopped after finishing. Used the following to loop:</p>
<pre><code>cap.set(cv2.CAP_PROP_POS_FRAMES, 0)</code></pre>

</body>
</html>
