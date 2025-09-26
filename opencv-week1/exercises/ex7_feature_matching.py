"""
Exercise 7: Feature Detection and Matching using ORB
"""

import cv2

# Load images
# img1 = cv2.imread("nature.jpg", cv2.IMREAD_GRAYSCALE)  # query image
# img2 = cv2.imread("nature1.jpg", cv2.IMREAD_GRAYSCALE) # train image

img1 = cv2.imread("taj1.jpg", cv2.IMREAD_GRAYSCALE)  # query image
img2 = cv2.imread("taj.jpg", cv2.IMREAD_GRAYSCALE) # train image


if img1 is None or img2 is None:
    print("Error: One or both images not found")
    exit()

# ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 20 matches
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

cv2.imshow("Feature Matching", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
