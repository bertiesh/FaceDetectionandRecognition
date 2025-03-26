import os
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import logging
import matplotlib
matplotlib.use('Agg')

from src.facematch.utils.embedding_utils import get_arcface_embedding

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_detections(image_path, boxes, scores, landmarks=None, save_path=None):
    """Visualize face detections with bounding boxes and optional landmarks"""    
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path.copy()
    
    # Draw detections
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        score_text = f"{score:.2f}"
        cv2.putText(img, score_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if landmarks and i < len(landmarks) and landmarks[i] is not None:
            for point in landmarks[i]:
                x, y = map(int, point)
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()


def get_target_size(model_name):
    """Get target size based on the embedding model"""
    if model_name == "Facenet512":
        return (160, 160)
    elif model_name == "OpenFace":
        return (96, 96)
    elif model_name in ["DeepFace", "DeepID"]:
        return (152, 152)
    elif model_name in ["ArcFace", "SFace"]:
        return (112, 112)
    elif model_name == "Dlib":
        return (150, 150)
    else:
        return (224, 224)


def crop_face_for_embedding(face_img):
   
    if face_img is None or face_img.size == 0:
        return None
        
    h, w = face_img.shape[:2]
    
    # Calculate crop margins
    top_margin = int(h * 0.1)      # 0% from top
    bottom_margin = int(h * 0.2)  # 20% from bottom
    left_margin = int(w * 0.15)     # 10% from left
    right_margin = int(w * 0.15)    # 10% from right
    
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


def process_yolov8_output(outputs, letterbox_info=None, height_factor=1.25):
    """Process YOLOv8 face detection output in grid format (1, 5, 8400). Creates square bounding boxes and returns them in a format compatible with the rest of the pipeline."""
    boxes, scores, landmarks = [], [], []
    
    # YOLOv8-face in grid format
    output = outputs[0][0]  # Shape (5, 8400)
    
    confidence = output[4]  # Shape (8400,)
    
    # Get indices of potential faces (confidence above threshold)
    threshold = 0.7  # Adjust as needed
    mask = confidence > threshold
    indices = np.nonzero(mask)[0]
    
    logger.info(f"Found {len(indices)} potential faces above threshold {threshold}")
    
    if len(indices) == 0:
        return [], [], []
    
    # Extract filtered boxes
    x = output[0][indices]
    y = output[1][indices]
    w = output[2][indices]
    h = output[3][indices]
    conf = confidence[indices]
    
    # Convert to corner format (x1, y1, x2, y2)
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    
    # Adjust for letterbox
    if letterbox_info:
        scale = letterbox_info["scale"]
        pad_w = letterbox_info["pad_w"]
        pad_h = letterbox_info["pad_h"]
        
        # Remove padding and rescale
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip to image boundaries
        orig_w, orig_h = letterbox_info["orig_size"]
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
    
    # Apply NMS on initial boxes for efficiency
    initial_boxes = [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in range(len(indices))]
    scores = [float(conf[i]) for i in range(len(indices))]
    landmarks = [None] * len(indices)
    
    if len(initial_boxes) > 1:
        try:
            nms_indices = cv2.dnn.NMSBoxes(
                initial_boxes, scores, score_threshold=threshold, nms_threshold=0.45
            )
            
            # Filter initial boxes, scores, and landmarks based on NMS
            filtered_boxes = [initial_boxes[i] for i in nms_indices]
            filtered_scores = [scores[i] for i in nms_indices]
            filtered_landmarks = [landmarks[i] for i in nms_indices]
            
            initial_boxes, scores, landmarks = filtered_boxes, filtered_scores, filtered_landmarks
            logger.info(f"NMS reduced detections from {len(indices)} to {len(initial_boxes)}")
        except Exception as e:
            logger.error(f"Error applying NMS: {e}")
    
    # Now create square boxes
    square_boxes = []
    for box in initial_boxes:
        x1, y1, x2, y2 = box
        
        # Current dimensions
        current_width = x2 - x1
        current_height = y2 - y1
        
        # Apply height factor to expand height
        expanded_height = current_height * height_factor
        
        # Center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Use the expanded height as the side length for the square
        side_length = expanded_height
        
        # Calculate new coordinates to make a square box
        new_x1 = center_x - (side_length / 2)
        new_y1 = center_y - (side_length / 2)
        new_x2 = center_x + (side_length / 2)
        new_y2 = center_y + (side_length / 2)
        
        # Ensure box is within image boundaries
        if letterbox_info:
            orig_w, orig_h = letterbox_info["orig_size"]
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(orig_w, new_x2)
            new_y2 = min(orig_h, new_y2)
        
        square_boxes.append([new_x1, new_y1, new_x2, new_y2])
    
    logger.info(f"Final face count: {len(square_boxes)}")
    return square_boxes, scores, landmarks


def process_yolov9_output(outputs, letterbox_info=None):
    """Process YOLOv9 grid-based output format
    
    YOLOv9 has outputs in the format:
    - First output (1, 5, 8400): Box predictions
    - Second output (1, 5, 8400): Class predictions
    - Other outputs: Feature maps
    """
    boxes, scores, landmarks = [], [], []
    
    try:
        # Standard YOLOv9 format: first two outputs are boxes and classes
        boxes_output = outputs[0][0]  # Shape (5, 8400)
        classes_output = outputs[1][0]  # Shape (5, 8400)
        
        num_classes = classes_output.shape[0]
        num_detections = boxes_output.shape[1]
                
        valid_detections = 0
        
        # Transpose for easier processing (8400, 5)
        boxes_output = boxes_output.transpose()
        classes_output = classes_output.transpose()
        
        for i in range(num_detections):
            x_center, y_center, width, height, confidence = boxes_output[i]
            
            class_scores = classes_output[i]
            class_id = int(np.argmax(class_scores))
            class_confidence = float(class_scores[class_id])
            
            combined_confidence = float(confidence * class_confidence)
            final_confidence = min(1.0, combined_confidence / 300.0)  # Scale down very high values
            
            if final_confidence < 0.3:
                continue
            
            valid_detections += 1
            
            # Convert center format to corner format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Adjust for letterbox
            if letterbox_info:
                scale = letterbox_info["scale"]
                pad_w = letterbox_info["pad_w"]
                pad_h = letterbox_info["pad_h"]
                
                # Remove padding and rescale
                x1 = (x1 - pad_w) / scale
                y1 = (y1 - pad_h) / scale
                x2 = (x2 - pad_w) / scale
                y2 = (y2 - pad_h) / scale
                
                # Clip to image boundaries
                orig_w, orig_h = letterbox_info["orig_size"]
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))
            
            width = x2 - x1
            height = y2 - y1
            
            # Check if this is likely a body detection (height > 2*width)
            is_body = height > 2 * width
            
            if is_body:
                # For body detections, extract just the face
                face_height = height * 0.2  # Take top 20% of body height
                face_width = min(width, face_height)
                face_center_x = (x1 + x2) / 2
                face_top = y1 + height * 0.02  # Small offset from very top
                
                # Calculate new face coordinates
                new_x1 = face_center_x - face_width/2
                new_y1 = face_top
                new_x2 = face_center_x + face_width/2
                new_y2 = face_top + face_height
            else:
                new_x1 = x1
                new_y1 = y1
                new_x2 = x2
                new_y2 = y2
            
            # Ensure coordinates are valid
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            if letterbox_info:
                orig_w, orig_h = letterbox_info["orig_size"]
                new_x2 = min(orig_w, new_x2)
                new_y2 = min(orig_h, new_y2)
            
            # Save detection
            boxes.append([new_x1, new_y1, new_x2, new_y2])
            scores.append(final_confidence)
            landmarks.append(None)
        
        # Apply non-maximum suppression to filter overlapping boxes
        if len(boxes) > 1:
            try:
                import cv2
                indices = cv2.dnn.NMSBoxes(
                    boxes, scores, score_threshold=0.3, nms_threshold=0.45
                )
                
                filtered_boxes = [boxes[i] for i in indices]
                filtered_scores = [scores[i] for i in indices]
                filtered_landmarks = [landmarks[i] for i in indices]
                
                boxes, scores, landmarks = filtered_boxes, filtered_scores, filtered_landmarks
            except Exception as e:
                logger.error(f"Error applying NMS: {e}")
                        
    except Exception as e:
        logger.error(f"Error processing YOLOv9 output: {str(e)}", exc_info=True)
        return [], [], []
    
    return boxes, scores, landmarks


def process_yolo11_output(outputs, letterbox_info=None):
    """Process YOLOv11 output with enhanced face extraction from body detections"""
    boxes, scores, landmarks = [], [], []
    
    # Get detections array
    detections = outputs[0]
    
    logger.info(f"Processing {len(detections[0])} YOLOv11 detections")
    
    valid_count = 0
    for detection in detections[0]:
        # Extract values
        x1, y1, x2, y2, confidence, class_id = detection
        
        # Usually class 0 is person and class 1 might be face, adjust based on your model
        if int(class_id) not in [0] or confidence < 0.3:
            continue
            
        valid_count += 1
        
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        # Adjust for letterbox
        if letterbox_info:
            scale = letterbox_info["scale"]
            pad_w = letterbox_info["pad_w"]
            pad_h = letterbox_info["pad_h"]
            
            # Remove padding and rescale
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            
            # Clip to image boundaries
            orig_w, orig_h = letterbox_info["orig_size"]
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))
        
        # Calculate box dimensions
        width, height = x2 - x1, y2 - y1
        
        # Check if this is likely a body detection (height > 2*width)
        is_body = height > 2 * width
        
        if is_body:
            # For body detections, extract just the face
            face_height = height * 0.2  # Take top 20% of body height
            
            face_width = min(width, face_height)

            face_center_x = (x1 + x2) / 2
            
            face_top = y1 + height * 0.02  # Small offset from very top
            
            new_x1 = face_center_x - face_width/2
            new_y1 = face_top
            new_x2 = face_center_x + face_width/2
            new_y2 = face_top + face_height
        else:
            top_margin = height * 0.1     # Trim 10% from top
            bottom_margin = height * 0.3  # Trim 30% from bottom
            left_margin = width * 0.2     # Trim 20% from left
            right_margin = width * 0.2    # Trim 20% from right
            
            new_x1 = x1 + left_margin
            new_y1 = y1 + top_margin
            new_x2 = x2 - right_margin
            new_y2 = y2 - bottom_margin
        
        # Ensure coordinates are valid
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        if letterbox_info:
            orig_w, orig_h = letterbox_info["orig_size"]
            new_x2 = min(orig_w, new_x2)
            new_y2 = min(orig_h, new_y2)
        
        # Save detection
        boxes.append([new_x1, new_y1, new_x2, new_y2])
        scores.append(float(confidence))
        landmarks.append(None)
    
    logger.info(f"Found {valid_count} valid detections, processed to {len(boxes)} face regions")
    return boxes, scores, landmarks


def extract_face(img, box, landmark, detector_backend):
    """Extract face region based on bounding box"""
    img_height, img_width = img.shape[:2]
    
    # Parse box coordinates - handle both formats
    if len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
    else:
        logger.warning(f"Invalid box format: {box}")
        return None, None
    
    # Ensure coordinates are within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    # Check if box is valid
    if x2 <= x1 or y2 <= y1 or x2 > img_width or y2 > img_height:
        logger.warning(f"Invalid face coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return None, None
    
    # Extract face region
    try:
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            logger.warning("Extracted face has zero size")
            return None, None
    except Exception as e:
        logger.error(f"Error extracting face: {e}")
        return None, None
    
    # Create region info with landmarks
    region = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "left_eye": None, "right_eye": None
    }
    
    # Add landmarks if available
    if landmark is not None:
        # We expect landmarks in format: [(x1,y1), (x2,y2), ...]
        if len(landmark) >= 2:
            # First two points are usually left eye, right eye
            region["left_eye"] = (int(landmark[0][0]), int(landmark[0][1]))
            region["right_eye"] = (int(landmark[1][0]), int(landmark[1][1]))
    
    return face, region


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
        # FaceNet/FaceNet512 preprocessing
        face_normalized = face_resized.astype(np.float32)
        face_normalized = (face_normalized - 127.5) / 128.0
        
    elif model_name in ["ArcFace", "SFace"]:
        # ArcFace/SFace preprocessing
        face_normalized = face_resized.astype(np.float32)
        face_normalized = face_normalized / 255.0
        # Standard normalization
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        face_normalized = (face_normalized - mean) / std
        
    else:
        face_normalized = face_resized.astype(np.float32) / 255.0
    
    return face_normalized


def prepare_for_embedding(face, model_name, normalization):
    """
    Final preparation to make the face compatible with embedding model's expectations
    """
    # DeepFace's models generally expect uint8 input (0-255)
    # If we've normalized, we need to convert back
    if normalization and face is not None:
        
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


def process_yolo_detections(img, boxes, scores, landmarks, align=True, target_size=None, normalization=True, visualize=False, image_path=None, model_name="ArcFace", model_onnx_path=None, path_str=None,face_confidence_threshold=0.02,detector_backend="yolov8"):
    """ Process YOLO face detections and generate embeddings."""
    face_embeddings = []
    
    if len(boxes) == 0:
        return face_embeddings
        
    valid_faces = sum(1 for score in scores if score >= face_confidence_threshold)
    if valid_faces == 0:
        return face_embeddings
        
    # Process each detected face
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < face_confidence_threshold:
            continue
            
        landmark = landmarks[i] if landmarks and i < len(landmarks) else None
        
        face, region = extract_face(img, box, landmark, detector_backend)
        
        if face is None or face.size == 0:
            continue
        
        region["confidence"] = float(score)
        
        # Align face if landmarks available
        if align and region["left_eye"] is not None and region["right_eye"] is not None:
            face = align_face(face, img, region)
        
        # Process for embedding
        face = crop_face_for_embedding(face)
        face_normalized = normalize_face(face, target_size, model_name, normalization)
        if face_normalized is None:
            continue
            
        detection = prepare_for_embedding(face_normalized, model_name, normalization)
        if detection is None:
            continue

        # Visualize processed faces
        if visualize and isinstance(image_path, str):
            debug_dir = "debug_faces"
            os.makedirs(debug_dir, exist_ok=True)
            face_path = os.path.join(debug_dir, f"{os.path.basename(image_path)}_face_{i}.jpg")
            if isinstance(detection, np.ndarray):
                cv2.imwrite(face_path, cv2.cvtColor(detection, cv2.COLOR_RGB2BGR))
        
        # Generate embedding
        try:
            embedding = get_arcface_embedding(detection, model_onnx_path)
                                
            if embedding is not None:

                face_embeddings.append({
                    "image_path": path_str,
                    "embedding": embedding,
                    "bbox": [region["x"], region["y"], region["w"], region["h"]],
                    "confidence": region["confidence"]
                })

        except Exception as e:
            logger.error(f"Error getting embedding for face {i}: {str(e)}")
            continue
    
    return face_embeddings