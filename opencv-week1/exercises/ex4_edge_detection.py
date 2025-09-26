"""
Exercise 4: Edge Detection
"""

import cv2

img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found")
    exit()

# Canny edges
canny = cv2.Canny(img, 100, 200)

# Sobel edges
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

cv2.imshow("Original", img)
cv2.imshow("Canny", canny)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)

cv2.waitKey(0)
cv2.destroyAllWindows()

