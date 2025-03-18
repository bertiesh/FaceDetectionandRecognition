import os
import cv2
import numpy as np
import onnxruntime as ort
from deepface import DeepFace
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.facematch.utils.detector_utils import (get_target_size,
                                                crop_face_for_embedding,
                                                process_yolov8_output,
                                                process_yolov9_output,
                                                process_yolo11_output,
                                                extract_face,
                                                create_face_bounds_from_landmarks,
                                                align_face,
                                                normalize_face,
                                                prepare_for_deepface,
                                                visualize_detections)

from src.facematch.utils.retinaface_utils import detect_with_retinaface

def detect_faces_and_get_embeddings(
    image_path, 
    model_name="ArcFace", 
    detector_backend="retinaface",
    detector_onnx_path=None,
    face_confidence_threshold=0.02,
    align=True,
    normalization=True,
    input_size=(640, 640),
    visualize=False
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    # Set detector path based on backend
    if detector_backend == "yolov8":
        detector_onnx_path = os.path.join(models_dir, "yolov8-face-detection.onnx")
    elif detector_backend == "yolo11":
        detector_onnx_path = os.path.join(models_dir, "yolo11m.onnx")
    elif detector_backend == "yolov9":
        detector_onnx_path = os.path.join(models_dir, "yolov9.onnx")
    elif detector_backend == "retinaface":
        detector_onnx_path = os.path.join(models_dir, "retinaface-resnet50.onnx")
        
    if visualize:
        os.makedirs("debug_detections", exist_ok=True)
    
    try:
        if detector_onnx_path is None or not os.path.isfile(detector_onnx_path):
            logger.error(f"ONNX model not found: {detector_onnx_path}")
            return False, []
        
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False, []
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            path_str = image_path
        else:
            img = image_path
            if len(img.shape) == 3 and img.shape[2] == 3:
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
            path_str = "array_input"
                    
        target_size = get_target_size(model_name)
        original_size = (img.shape[1], img.shape[0])
        
        # Special handling for RetinaFace
        if detector_backend == "retinaface":
            try:
                
                boxes, scores, landmarks = detect_with_retinaface(
                    image_path=image_path if isinstance(image_path, str) else None,
                    img_rgb=img if not isinstance(image_path, str) else None,
                    model_path=detector_onnx_path,
                    confidence_threshold=.02,
                    visualize=visualize
                )
                
                face_embeddings = []
                
                for i, (box, score, landmark) in enumerate(zip(boxes, scores, landmarks)):
                    try:
                        # Use landmarks to create better bounding box if available
                        if landmark and len(landmark) >= 5:
                            improved_box = create_face_bounds_from_landmarks(landmark, img.shape, margin_ratio=1.30)
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
                            
                        detection = prepare_for_deepface(face_normalized, model_name, normalization)
                        if detection is None:
                            continue
                            
                        # Visualize processed face
                        if visualize and isinstance(image_path, str):
                            debug_dir = "debug_faces"
                            os.makedirs(debug_dir, exist_ok=True)
                            face_path = os.path.join(debug_dir, f"{os.path.basename(image_path)}_face_{i}.jpg")
                            if isinstance(detection, np.ndarray):
                                cv2.imwrite(face_path, cv2.cvtColor(detection, cv2.COLOR_RGB2BGR))
                        
                        # Generate embedding
                        embedding_results = DeepFace.represent(
                            img_path=detection,
                            model_name=model_name,
                            detector_backend='skip',
                            enforce_detection=False,
                            align=False,
                            normalization="base"
                        )
                        
                        if embedding_results:
                            embedding = embedding_results[0]["embedding"]
                            face_embeddings.append({
                                "image_path": path_str,
                                "embedding": embedding,
                                "bbox": [region["x"], region["y"], region["w"], region["h"]],
                                "confidence": region["confidence"]
                            })
                    
                    except Exception as e:
                        logger.error(f"Error processing face {i}: {str(e)}")
                        continue
                
                if len(face_embeddings) > 0:
                    return True, face_embeddings
                return False, []
                
            except Exception as e:
                logger.error(f"Error in RetinaFace: {str(e)}")
                # Fall back to YOLO detector
                detector_backend = "yolov9"
                detector_onnx_path = os.path.join(models_dir, "yolov9.onnx")
        
        # YOLO models processing
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')
        elif 'CoreMLExecutionProvider' in available_providers:
            providers.insert(0, 'CoreMLExecutionProvider')
        elif 'MetalPerformanceShadersExecutionProvider' in available_providers:
            providers.insert(0, 'MetalPerformanceShadersExecutionProvider')
                
        detector_session = ort.InferenceSession(
            detector_onnx_path, 
            sess_options=session_options,
            providers=providers
        )
        
        model_inputs = detector_session.get_inputs()
        input_name = model_inputs[0].name
        
        # YOLO preprocessing
        letterbox_info = None
        if detector_backend in ["yolov8", "yolo11", "yolov9"]:
            scale = min(input_size[0] / original_size[0], input_size[1] / original_size[1])
            new_w = int(original_size[0] * scale)
            new_h = int(original_size[1] * scale)
            pad_w = (input_size[0] - new_w) // 2
            pad_h = (input_size[1] - new_h) // 2
            
            letterbox_info = {
                "scale": scale,
                "pad_w": pad_w,
                "pad_h": pad_h,
                "orig_size": original_size
            }
        
            img_resized = cv2.resize(img, (new_w, new_h))
            letterbox_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
            letterbox_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = img_resized
            img_norm = letterbox_img.astype(np.float32) / 255.0
            img_input = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0)
        
            outputs = detector_session.run(None, {input_name: img_input})
            
            if detector_backend == 'yolov8':
                boxes, scores, landmarks = process_yolov8_output(outputs, letterbox_info)
            elif detector_backend == "yolov9":
                boxes, scores, landmarks = process_yolov9_output(outputs, letterbox_info)
            elif detector_backend == "yolo11":
                boxes, scores, landmarks = process_yolo11_output(outputs, letterbox_info)

        # Visualize detections
        if visualize and isinstance(image_path, str) and len(boxes) > 0:
            debug_dir = "debug_detections"
            os.makedirs(debug_dir, exist_ok=True)
            vis_path = os.path.join(debug_dir, os.path.basename(image_path) + "_detect.jpg")
            visualize_detections(img, boxes, scores, landmarks, save_path=vis_path)
        
        # Handle no detections
        if len(boxes) == 0:
            return False, []
            
        # Count valid detections
        valid_faces = sum(1 for score in scores if score >= face_confidence_threshold)
        if valid_faces == 0:
            return False, []
            
        # Process each detected face
        face_embeddings = []
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < face_confidence_threshold:
                continue
                
            landmark = landmarks[i] if landmarks and i < len(landmarks) else None
            
            # Extract face region
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
                
            detection = prepare_for_deepface(face_normalized, model_name, normalization)
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
                embedding_results = DeepFace.represent(
                    img_path=detection,
                    model_name=model_name,
                    detector_backend='skip',
                    enforce_detection=False,
                    align=False,
                    normalization="base"
                )
                
                if embedding_results:
                    embedding = embedding_results[0]["embedding"]
                    face_embeddings.append({
                        "image_path": path_str,
                        "embedding": embedding,
                        "bbox": [region["x"], region["y"], region["w"], region["h"]],
                        "confidence": region["confidence"]
                    })
            except Exception as e:
                logger.error(f"Error getting embedding for face {i}: {str(e)}")
                continue
        
        if len(face_embeddings) == 0:
            return False, []
            
        return True, face_embeddings
        
    except Exception as e:
        logger.error(f"Error in detect_faces_and_get_embeddings: {str(e)}")
        return False, [e]