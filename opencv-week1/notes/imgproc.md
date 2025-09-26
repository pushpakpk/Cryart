# Image Processing (cv2.imgproc)

## Geometric Transformations
- Resize: `cv2.resize()`
- Rotate: `cv2.getRotationMatrix2D()`
- Flip: `cv2.flip()`
- Perspective warp: `cv2.warpPerspective()`

## Filtering
- Blur: `cv2.blur(img, (5,5))`
- Gaussian: `cv2.GaussianBlur(img, (5,5), 0)`
- Median: `cv2.medianBlur(img, 5)`
- Bilateral: `cv2.bilateralFilter(img, 9, 75, 75)`

## Edge Detection
- Sobel, Laplacian, Canny
```python
edges = cv2.Canny(img, 100, 200)