# Core Module (cv2.core)

## Data Structures
- In C++: `Mat`
- In Python: `numpy.ndarray`
- Represents images, matrices, or multi-dimensional arrays.

## Array Operations
- Access pixel: `img[y, x]`
- Shape: `img.shape` (rows, cols, channels)
- Data type: `img.dtype`
- Copying: `clone()` vs `copy()`

## Drawing Functions
```python
cv2.line(img, (0,0), (100,100), (255,0,0), 2)
cv2.rectangle(img, (50,50), (200,200), (0,255,0), 3)
cv2.circle(img, (150,150), 50, (0,0,255), -1)
cv2.putText(img, "Hello OpenCV", (10,300),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)