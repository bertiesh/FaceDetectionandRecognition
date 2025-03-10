import numpy as np
import cv2
import logging
import os
import onnxruntime as ort

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
        
        if self.format == "torch":
            import torch
            output = torch.Tensor(anchors)
        else:
            output = np.array(anchors)
            
        if self.clip:
            if self.format == "torch":
                output.clamp_(max=1, min=0)
            else:
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
    keep = py_cpu_nms(dets, 0.5)  # NMS threshold
    
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
    session_options = ort.SessionOptions()
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
        keep = py_cpu_nms(dets, 0.4)  # NMS threshold
        
        # Keep top-K after NMS
        keep_top_k = 5  # Reduced from 750 for efficiency
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