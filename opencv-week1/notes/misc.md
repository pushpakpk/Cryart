# Miscellaneous Topics

## Theory of Image Processing
- **Pixel**: Smallest unit of an image.
- **Color Spaces**:
  - RGB (Red, Green, Blue)
  - Grayscale
  - HSV (Hue, Saturation, Value)
- **Convolution**:
  - Operation with kernels/filters.
  - Used in blurring, sharpening, edge detection.
- **Edge Detection**:
  - Sobel, Laplacian, Canny methods.
- **Fourier Transform**:
  - Converts image to frequency domain.

## Camera Calibration & 3D (cv2.calib3d)
- Intrinsic Parameters: focal length, optical center.
- Extrinsic Parameters: rotation, translation.
- Undistortion and stereo vision.

## Deep Neural Networks (cv2.dnn)
- Load pre-trained models in Caffe, TensorFlow, ONNX.
- Perform inference:
```python
net = cv2.dnn.readNetFromONNX("model.onnx")
output = net.forward()