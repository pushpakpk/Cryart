"""
Exercise 2: Geometric Transformations
"""

import cv2

# Load image
img = cv2.imread("sample.jpg")

if img is None:
    print("Error: Image not found")
    exit()

# Resize
resized = cv2.resize(img, (300, 300))

# Rotate
(h, w) = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# Flip
flipped = cv2.flip(img, 1)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Resized", resized)
cv2.imshow("Rotated", rotated)
cv2.imshow("Flipped", flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()
