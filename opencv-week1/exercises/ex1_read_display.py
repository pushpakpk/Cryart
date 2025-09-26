"""
Exercise 1: Read and Display an Image
"""

import cv2

# Read an image
img = cv2.imread("sample.jpg")  # replace with your image path

# Check if image is loaded
if img is None:
    print("Error: Image not found")
    exit()

# Display the image
cv2.imshow("Original Image", img)

# Wait until any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
