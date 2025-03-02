from deepface.modules import modeling, preprocessing
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from deepface.commons import image_utils
from deepface.models.FacialRecognition import FacialRecognition
from typing import Tuple, Union
import cv2
from PIL import Image
import numpy as np

def project_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    direction = 1 if angle >= 0 else -1
    angle = abs(angle) % 360
    if angle == 0:
        return facial_area

    # Angle in radians
    angle = angle * np.pi / 180

    height, weight = size

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return (x1, y1, x2, y2)

def align_img_wrt_eyes(
    img: np.ndarray,
    left_eye: Union[list, tuple],
    right_eye: Union[list, tuple],
) -> Tuple[np.ndarray, float]:
    # if eye could not be detected for the given image, return image itself
    if left_eye is None or right_eye is None:
        return img, 0

    # sometimes unexpectedly detected images come with nil dimensions
    if img.shape[0] == 0 or img.shape[1] == 0:
        return img, 0

    angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
    img = np.array(Image.fromarray(img).rotate(angle, resample=Image.BICUBIC))
    return img, angle

model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name="ArcFace"
    )
target_size = model.input_shape

# Initialize YOLOv8 model
face_detector: Detector = modeling.build_model(
        task="face_detector", model_name="yolov8"
    )
img, img_name = image_utils.load_image("/Users/xyx/Documents/spring2025/596E/Face/FaceDetectionandRecognition/resources/sample_images/me.png")
height, width, _ = img.shape
height_border = int(0.5 * height)
width_border = int(0.5 * width)

img = cv2.copyMakeBorder(
            img,
            height_border,
            height_border,
            width_border,
            width_border,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # Color of the border (black)
        )
facial_areas = face_detector.detect_faces(img)

results = []
for facial_area in facial_areas:
    x = facial_area.x
    y = facial_area.y
    w = facial_area.w
    h = facial_area.h
    left_eye = facial_area.left_eye
    right_eye = facial_area.right_eye
    confidence = facial_area.confidence

    # extract detected face unaligned
    detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

    aligned_img, angle = align_img_wrt_eyes(img=img, left_eye=left_eye, right_eye=right_eye)

    rotated_x1, rotated_y1, rotated_x2, rotated_y2 = project_facial_area(
        facial_area=(x, y, x + w, y + h), angle=angle, size=(img.shape[0], img.shape[1])
    )
    detected_face = aligned_img[
        int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
    ]

    # restore x, y, le and re before border added
    x = x - width_border
    y = y - height_border
    # w and h will not change
    if left_eye is not None:
        left_eye = (left_eye[0] - width_border, left_eye[1] - height_border)
    if right_eye is not None:
        right_eye = (right_eye[0] - width_border, right_eye[1] - height_border)

    result = DetectedFace(
        img=detected_face,
        facial_area=FacialAreaRegion(
            x=x, y=y, h=h, w=w, confidence=confidence, left_eye=left_eye, right_eye=right_eye
        ),
        confidence=confidence,
    )
    results.append(result)

resp_objs = []
for img_obj in result:
    img = img_obj["face"]

    # rgb to bgr
    img = img[:, :, ::-1]

    region = img_obj["facial_area"]
    confidence = img_obj["confidence"]

    # resize to expected shape of ml model
    img = preprocessing.resize_image(
        img=img,
        # thanks to DeepId (!)
        target_size=(target_size[1], target_size[0]),
    )

    # custom normalization
    img = preprocessing.normalize_input(img=img, normalization="base")

    embedding = model.forward(img)

    resp_objs.append(
        {
            "embedding": embedding,
            "facial_area": region,
            "face_confidence": confidence,
        }
    )

print(resp_objs)


