import os
import logging
import numpy as np
import cv2
from src.facematch.utils.retinaface_utils import detect_with_retinaface
from src.facematch.utils.detector_utils import crop_face_for_embedding, create_face_bounds_from_landmarks, get_target_size, normalize_face, prepare_for_deepface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    model_path = "/Users/davidthibodeau/Desktop/CS596E/group_proj/FaceDetectionandRecognition/src/facematch/models/retinaface-resnet50.onnx"
    image_path = "resources/sample_images/me.png"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return
    
    try:
        logger.info("Running RetinaFace detection...")
        boxes, scores, landmarks = detect_with_retinaface(
            image_path=image_path,
            model_path=model_path,
            visualize=True
        )
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization directory
        viz_dir = "visualization"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Process each face with visualization at each step
        if len(boxes) > 0:
            logger.info(f"Detection successful, found {len(boxes)} faces")
            
            for i, (box, score, landmark) in enumerate(zip(boxes, scores, landmarks)):
                # Create a copy of original image for drawing
                viz_img = img.copy()
                
                # 1. Draw original detection
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
                cv2.putText(viz_img, f"Original: {score:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 2. Draw landmarks
                if landmark:
                    for j, point in enumerate(landmark):
                        x, y = map(int, point)
                        cv2.circle(viz_img, (x, y), 2, (255, 0, 0), -1)  # Blue
                        cv2.putText(viz_img, str(j), (x+3, y+3), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                
                # 3. Create and draw landmark-based bounds
                if landmark and len(landmark) >= 5:
                    improved_box = create_face_bounds_from_landmarks(landmark, img.shape, margin_ratio=0.15)
                    if improved_box:
                        lx1, ly1, lx2, ly2 = improved_box
                        cv2.rectangle(viz_img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)  # Green
                        cv2.putText(viz_img, "Landmark-based", (lx1, ly1-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save this visualization
                step1_path = os.path.join(viz_dir, f"{os.path.basename(image_path)}_step1_face{i}.jpg")
                cv2.imwrite(step1_path, viz_img)
                
                # 4. Extract face - use landmark-based box if available
                if landmark and len(landmark) >= 5:
                    improved_box = create_face_bounds_from_landmarks(landmark, img.shape, margin_ratio=.70)
                    if improved_box:
                        x1, y1, x2, y2 = improved_box
                
                face = img_rgb[y1:y2, x1:x2].copy()
                
                face_path = os.path.join(viz_dir, f"{os.path.basename(image_path)}_step2_extracted_face{i}.jpg")
                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                
                # 5. Crop face for embedding
                cropped_face = crop_face_for_embedding(face)
                crop_path = os.path.join(viz_dir, f"{os.path.basename(image_path)}_step3_cropped_face{i}.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                
                # 6. Normalize face for embedding model
                model_name = "ArcFace"
                target_size = get_target_size(model_name)
                normalized_face = normalize_face(cropped_face, target_size, model_name, True)
                
                # 7. Prepare for DeepFace
                final_face = prepare_for_deepface(normalized_face, model_name, True)
                final_path = os.path.join(viz_dir, f"{os.path.basename(image_path)}_step4_final_face{i}.jpg")
                cv2.imwrite(final_path, cv2.cvtColor(final_face, cv2.COLOR_RGB2BGR))
                
                logger.info(f"Visualization for face {i+1} saved to {viz_dir}")
        else:
            logger.warning("No faces detected")
            
    except Exception as e:
        logger.error(f"Error during detection: {e}")

if __name__ == "__main__":
    main()