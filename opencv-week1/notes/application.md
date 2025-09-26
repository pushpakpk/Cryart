# OpenCV Applications

## Real-World Use Cases
1. **Robotics**  
   - Navigation, SLAM (Simultaneous Localization and Mapping).
2. **Medical Imaging**  
   - Tumor detection, CT/MRI/X-ray analysis.
3. **Autonomous Vehicles**  
   - Lane detection, obstacle recognition.
4. **Surveillance**  
   - Face and motion detection.
5. **Augmented Reality**  
   - Marker tracking, overlaying 3D objects.
6. **Image Stitching**  
   - Panorama creation.
7. **Industrial Automation**  
   - Defect detection, quality control.
8. **Sports Analytics**  
   - Player tracking, ball trajectory analysis.

## Example: Face Detection
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)