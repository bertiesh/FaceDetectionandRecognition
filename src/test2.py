from ultralytics import YOLO

image_path = '/Users/xyx/Documents/spring2025/596E/Face/FaceDetectionandRecognition/resources/sample_images/me.png'
model = YOLO('yolov8n-face.pt')  # or your preferred YOLOv8 face detection model
results = model(image_path)
print(results)  # print results in a human-readable format