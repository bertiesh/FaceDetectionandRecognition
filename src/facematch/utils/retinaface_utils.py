import numpy as np
import cv2
import logging
import os
import onnxruntime as ort

from src.facematch.utils.embedding_utils import get_arcface_embedding

logger = logging.getLogger(__name__)

class PriorBox:
    """Prior box generator as used in the original RetinaFace implementation."""
    def __init__(self, cfg, image_size, format):
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.format = format

    def forward(self):
        """Generate prior boxes like the original RetinaFace implementation."""
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]
                        s_ky = min_size / self.image_size[0]
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors.append([cx, cy, s_kx, s_ky])
        
        output = np.array(anchors)
            
        if self.clip:
            output = np.clip(output, 0, 1)
                
        return output

def ceil(x):
    return int(np.ceil(x))

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landmarks from predictions using priors.

    Return:
        decoded landmark predictions
    """
    landms = np.concatenate((
        priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]), 1)
    return landms

def py_cpu_nms(dets, thresh):
    """Pure Python NMS implementation for bounding boxes."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def prepare_retinaface_input(img, input_size=None):
    img = img.astype(np.float32)
    
    img -= (104, 117, 123)  # BGR mean subtraction
    
    # Transpose to CHW format
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def process_retinaface_output(outputs, input_shape, original_shape, confidence_threshold=0.02):
    """Process RetinaFace outputs following the original implementation."""
    # Define configuration similar to RetinaFace's cfg_re50
    cfg = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
    }
    
    loc, conf, landms = outputs
    
    height, width = original_shape[:2]
    
    priorbox = PriorBox(cfg, image_size=(height, width), format="numpy")
    priors = priorbox.forward()
    
    # Decode bounding boxes and landmarks
    scale = np.array([width, height, width, height])
    boxes = decode(np.squeeze(loc, axis=0), priors, cfg['variance'])
    boxes = boxes * scale
    scores = np.squeeze(conf, axis=0)[:, 1]
    
    # Decode landmarks
    scale1 = np.array([width, height, width, height, width, 
                       height, width, height, width, height])
    landms = decode_landm(np.squeeze(landms, axis=0), priors, cfg['variance'])
    landms = landms * scale1
    
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    
    # Keep top-K before NMS
    top_k = 10
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    
    # Apply NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.7)  # NMS threshold
    
    # Keep top-K after NMS
    keep_top_k = 5
    dets = dets[keep, :][:keep_top_k, :]
    landms = landms[keep][:keep_top_k, :]
    
    # Format outputs for our pipeline
    result_boxes = []
    result_scores = []
    result_landmarks = []
    
    for i, (det, landm) in enumerate(zip(dets, landms)):
        box = det[:4]
        score = det[4]
        
        # Format landmarks into pairs
        landmarks = []
        for j in range(0, 10, 2):
            landmarks.append([landm[j], landm[j+1]])
        
        result_boxes.append(box)
        result_scores.append(score)
        result_landmarks.append(landmarks)
    
    return result_boxes, result_scores, result_landmarks

def detect_with_retinaface(image_path=None, img_rgb=None, model_path=None, confidence_threshold=0.02, visualize=False):
    """Run full RetinaFace detection pipeline.
    
    Returns:
        boxes, scores, landmarks: Lists of detected face boxes, confidence scores, and landmarks
    """
    
    if image_path is None and img_rgb is None:
        logger.error("Either image_path or img_rgb must be provided")
        return [], [], []
    
    if img_rgb is None:
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_raw is None:
            logger.error(f"Could not load image: {image_path}")
            return [], [], []
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        # Make a copy to avoid modifying the original
        img_raw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    height, width = img_rgb.shape[:2]
    logger.debug(f"Image size: {width}x{height}")
    
    img_input = prepare_retinaface_input(img_rgb)
    
    # Load model
    # session_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return [], [], []
    
    # Run inference
    try:
        outputs = session.run(None, {"input": img_input})
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return [], [], []
    
    # Process outputs
    try:
        # Define configuration similar to RetinaFace's cfg_re50
        cfg = {
            'name': 'Resnet50',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
        }
        
        loc, conf, landms = outputs
        
        # Generate prior boxes
        priorbox = PriorBox(cfg, image_size=(height, width), format="numpy")
        priors = priorbox.forward()
        
        scale = np.array([width, height, width, height])
        boxes = decode(np.squeeze(loc, axis=0), priors, cfg['variance'])
        boxes = boxes * scale
        scores = np.squeeze(conf, axis=0)[:, 1]
        
        scale1 = np.array([width, height, width, height, width, 
                           height, width, height, width, height])
        landms = decode_landm(np.squeeze(landms, axis=0), priors, cfg['variance'])
        landms = landms * scale1
        
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # Keep top-K before NMS
        top_k = 10
        if len(scores) > top_k:
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
        
        # Apply NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.7)  # NMS threshold
        
        # Keep top-K after NMS
        keep_top_k = 5
        if len(keep) > keep_top_k:
            keep = keep[:keep_top_k]
            
        dets = dets[keep, :]
        landms = landms[keep]
        
        # Format outputs for our pipeline
        result_boxes = []
        result_scores = []
        result_landmarks = []
        
        for i, (det, landm) in enumerate(zip(dets, landms)):
            box = det[:4]
            score = det[4]
            
            # Format landmarks into pairs
            landmarks = []
            for j in range(0, 10, 2):
                landmarks.append([landm[j], landm[j+1]])
            
            result_boxes.append(box)
            result_scores.append(score)
            result_landmarks.append(landmarks)
        
        logger.info(f"Detected {len(result_boxes)} faces")
    except Exception as e:
        logger.error(f"Error processing RetinaFace output: {e}")
        return [], [], []
    
    # Visualize if requested
    if visualize:
        try:
            result_img = img_raw.copy()
            for i, (box, score, landmark) in enumerate(zip(result_boxes, result_scores, result_landmarks)):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw score
                text = f"{score:.2f}"
                cv2.putText(result_img, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw landmarks
                if landmark:
                    for j, point in enumerate(landmark):
                        cv2.circle(result_img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
            
            # Save visualization
            os.makedirs("debug_detections", exist_ok=True)
            if image_path:
                output_path = os.path.join("debug_detections", os.path.basename(image_path))
            else:
                import time
                output_path = os.path.join("debug_detections", f"detection_{int(time.time())}.jpg")
            cv2.imwrite(output_path, result_img)
            logger.info(f"Visualization saved to {output_path}")
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    return result_boxes, result_scores, result_landmarks

def crop_face_for_embedding(face_img):
   
    if face_img is None or face_img.size == 0:
        return None
        
    h, w = face_img.shape[:2]
    
    # Calculate crop margins
    top_margin = int(h * 0.0)      # 0% from top
    bottom_margin = int(h * 0.2)  # 0% from bottom
    left_margin = int(w * 0.1)     # 10% from left
    right_margin = int(w * 0.1)    # 10% from right
    
    # Apply cropping
    y_start = top_margin
    y_end = h - bottom_margin
    x_start = left_margin
    x_end = w - right_margin
    
    # Ensure valid dimensions
    if y_end <= y_start or x_end <= x_start:
        return face_img
    
    cropped_face = face_img[y_start:y_end, x_start:x_end]
    
    return cropped_face

def align_face(face, img, region):
    """Align face based on eye positions"""
    if region["left_eye"] is None or region["right_eye"] is None:
        return face
    
    left_eye = region["left_eye"]
    right_eye = region["right_eye"]
    
    # Calculate angle for alignment
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Calculate desired eye position based on face dimensions
    face_width, face_height = region["w"], region["h"]
    desired_left_eye_x = 0.35  # Proportion from the left edge
    desired_right_eye_x = 1.0 - desired_left_eye_x
    
    desired_eye_y = 0.4  # Proportion from the top edge
    
    desired_dist = (desired_right_eye_x - desired_left_eye_x) * face_width
    actual_dist = np.sqrt((dx ** 2) + (dy ** 2))
    scale = desired_dist / actual_dist
    
    # Calculate rotation center (between the eyes)
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)
    
    # Update translation component of the matrix
    rotation_matrix[0, 2] += (face_width * 0.5) - eye_center[0]
    rotation_matrix[1, 2] += (face_height * desired_eye_y) - eye_center[1]
    
    # Apply affine transformation to the original image
    output_size = (face_width, face_height)
    aligned_face = cv2.warpAffine(
        img, rotation_matrix, output_size,
        flags=cv2.INTER_CUBIC
    )
    
    y_min = max(0, region["y"] - int(face_height * 0.1))
    y_max = min(img.shape[0], region["y"] + region["h"] + int(face_height * 0.1))
    x_min = max(0, region["x"] - int(face_width * 0.1))
    x_max = min(img.shape[1], region["x"] + region["w"] + int(face_width * 0.1))
    
    aligned_face = aligned_face[y_min:y_max, x_min:x_max]
    
    return aligned_face


def normalize_face(face, target_size, model_name, normalization=True):
    """Normalize face for the embedding model"""
    
    if face is None or face.size == 0:
        logger.warning("Empty face provided to normalize_face")
        return None
        
    if not normalization:
        return face
    
    # Resize to target dimensions required by the embedding model
    face_resized = cv2.resize(face, target_size)
    
    # Convert to the expected format based on embedding model    
    if model_name == "Facenet512":
        face_normalized = face_resized.astype(np.float32)
        face_normalized = (face_normalized - 127.5) / 128.0
        
    elif model_name in ["ArcFace", "SFace"]:
        face_normalized = face_resized.astype(np.float32)
        face_normalized = face_normalized / 255.0
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        face_normalized = (face_normalized - mean) / std
        
    else:
        face_normalized = face_resized.astype(np.float32) / 255.0
    
    return face_normalized

def prepare_for_embedding(face, model_name, normalization):
    """Final preparation to make the face compatible with embedding model's expectations"""
    # If we've normalized, we need to convert back
    if normalization and face is not None:
        # Special handling for different models
        if model_name == "Facenet512":
            # For FaceNet, revert normalization
            face_uint8 = ((face * 128.0) + 127.5).astype(np.uint8)
            return face_uint8
        
        elif model_name in ["ArcFace", "SFace"]:
            # For ArcFace/SFace, revert normalization
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            face_uint8 = ((face * std) + mean) * 255
            face_uint8 = np.clip(face_uint8, 0, 255).astype(np.uint8)
            return face_uint8
        
        else:
            # Default reversion
            face_uint8 = (face * 255).astype(np.uint8)
            return face_uint8
    else:
        if face is not None:
            return face.astype(np.uint8)
        return None


def create_square_bounds_from_landmarks(landmarks, img_shape, scale_factor=1.5):
    """Create a square bounding box centered on facial landmarks."""
    if not landmarks or len(landmarks) < 2:
        return None
    
    # Find the center point (average of all landmarks)
    center_x = sum(point[0] for point in landmarks) / len(landmarks)
    center_y = sum(point[1] for point in landmarks) / len(landmarks)
    
    # Calculate the distances from center to each landmark
    distances = [
        max(abs(point[0] - center_x), abs(point[1] - center_y))
        for point in landmarks
    ]
    
    max_distance = max(distances)
    
    # Apply scale factor to control box size
    side_half = max_distance * scale_factor
    
    # Create square box centered on landmarks
    x1 = int(center_x - side_half)
    y1 = int(center_y - side_half)
    x2 = int(center_x + side_half)
    y2 = int(center_y + side_half)
    
    # Ensure box is within image boundaries
    height, width = img_shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    return [x1, y1, x2, y2]

def process_retinaface_detections(img, align, target_size, normalization, visualize, image_path, model_name, model_onnx_path, path_str, boxes, scores, landmarks):
    
    face_embeddings = []
                
    for i, (box, score, landmark) in enumerate(zip(boxes, scores, landmarks)):
        try:
            # Use landmarks to create better bounding box if available
            if landmark and len(landmark) >= 5:
                improved_box = create_square_bounds_from_landmarks(landmark, img.shape, scale_factor=4.0)
                if improved_box:
                    x1, y1, x2, y2 = improved_box
                else:
                    x1, y1, x2, y2 = map(int, box)
            else:
                x1, y1, x2, y2 = map(int, box)
                                
            # Ensure coordinates are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
                                
            if x2 <= x1 or y2 <= y1:
                continue
                                
            # Extract face
            face = img[y1:y2, x1:x2].copy()
                                
            # Create region info
            region = {
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "confidence": float(score)
            }
                                
            # Set landmarks for alignment
            if landmark and len(landmark) >= 2:
                region["left_eye"] = tuple(map(int, landmark[0]))
                region["right_eye"] = tuple(map(int, landmark[1]))
            else:
                region["left_eye"] = None
                region["right_eye"] = None
                                
            # Align face using landmarks
            if align and region["left_eye"] is not None and region["right_eye"] is not None:
                aligned_face = align_face(face, img, region)
                if aligned_face is not None and aligned_face.size > 0:
                    face = aligned_face
                                
            # Crop and normalize
            face = crop_face_for_embedding(face)
            face_normalized = normalize_face(face, target_size, model_name, normalization)
            if face_normalized is None:
                continue
                                    
            detection = prepare_for_embedding(face_normalized, model_name, normalization)
            if detection is None:
                continue
                                    
            # Visualize processed face
            if visualize and isinstance(image_path, str):
                debug_dir = "debug_faces"
                os.makedirs(debug_dir, exist_ok=True)
                face_path = os.path.join(debug_dir, f"{os.path.basename(image_path)}_face_{i}.jpg")
                if isinstance(detection, np.ndarray):
                    cv2.imwrite(face_path, cv2.cvtColor(detection, cv2.COLOR_RGB2BGR))

            embedding = get_arcface_embedding(detection, model_onnx_path)
                                
            if embedding is not None:

                face_embeddings.append({
                    "image_path": path_str,
                    "embedding": embedding,
                    "bbox": [region["x"], region["y"], region["w"], region["h"]],
                    "confidence": region["confidence"]
                })
                            
        except Exception as e:
            logger.error(f"Error processing face {i}: {str(e)}")
            continue
    
    return face_embeddings