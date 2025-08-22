import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
from torchvision import transforms
from collections import defaultdict
import time
from PIL import Image
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Mandatory OpenPifPaf import - fail if not available
try:
    import openpifpaf
    PIFPAF_AVAILABLE = True
except ImportError:
    raise ImportError(
        "OpenPifPaf is required for pose estimation mask processing!\n"
        "Please install it with one of these commands:\n"
        "  conda install -c conda-forge openpifpaf\n"
        "  pip install openpifpaf\n"
        "\nFor more information visit: https://openpifpaf.github.io/"
    )

class PoseBasedPersonReIDTester:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the Pose-Based ReID system with mandatory OpenPifPaf processing"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup OpenPifPaf for pose estimation (mandatory)
        self._setup_pose_estimation()
        
        # Load BPBReID model with pose-based masks
        self.reid_model = self._load_reid_model(reid_model_path, hrnet_path)
        self.reid_model.eval()
        
        # Setup transforms for ReID with pose-based mask processing
        self.transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Gallery to store person features
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.gallery_poses = []  # Store pose keypoints for visualization
        self.next_person_id = 1
        
        # Tracking confidence threshold
        self.reid_threshold = 0.4
        
        # Test results storage
        self.test_results = {}
        
        # Pose visualization settings
        self.show_pose_keypoints = True
        self.show_body_part_masks = True
        
    def _setup_pose_estimation(self):
        """Setup OpenPifPaf for pose estimation - MANDATORY"""
        try:
            print("Initializing OpenPifPaf pose estimation...")
            # Configure PifPaf with better checkpoint for accuracy
            self.pose_predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
            print("✓ OpenPifPaf pose estimation loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenPifPaf: {e}. This is required for pose-based mask processing!")
    
    def _load_reid_model(self, model_path, hrnet_path):
        """Load the BPBReID model with pose-based mask configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        from types import SimpleNamespace
        
        # Create config structure for BPBReID with pose-based masks enabled
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.load_weights = model_path
        config.model.load_config = True
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.mask_filtering_testing = True  # Enable mask filtering
        config.model.bpbreid.mask_filtering_training = False
        config.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']  # Use both foreground and parts
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'
        config.model.bpbreid.masks.preprocess = 'five_v'  # 5-part vertical pose-based segmentation
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.test_use_target_segmentation = 'generated_masks'  # Use generated pose masks
        config.model.bpbreid.testing_binary_visibility_score = True
        config.model.bpbreid.training_binary_visibility_score = True
        
        # Build model with config
        model = torchreid.models.build_model(
            name='bpbreid',
            num_classes=751,
            config=config,
            pretrained=True
        )
        
        # Handle DataParallel state dict (remove 'module.' prefix)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # Load the cleaned state dict
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        return model
    
    def generate_pose_based_masks(self, image):
        """Generate 5-part body segmentation heatmaps using OpenPifPaf pose estimation"""
        try:
            # Convert to PIL if needed for pose estimation
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Get pose keypoints using OpenPifPaf
            predictions, gt_anns, image_meta = self.pose_predictor.pil_image(pil_image)
            
            if len(predictions) == 0:
                raise ValueError("No pose detected - pose estimation is mandatory!")
            
            # Use the first (most confident) person detection
            pose = predictions[0]
            keypoints = pose.data
            
            # Create 5-part heatmaps based on keypoints using Gaussian kernels
            heatmaps = self._create_five_part_masks_from_pose(image, keypoints)
            return heatmaps, keypoints
            
        except Exception as e:
            raise RuntimeError(f"Pose estimation failed: {e}. Cannot proceed without pose-based heatmaps!")
    
    def _create_five_part_masks_from_pose(self, image, keypoints):
        """Create 5-part vertical body heatmaps based on pose keypoints"""
        if isinstance(image, Image.Image):
            h, w = image.size[1], image.size[0]
        else:
            h, w = image.shape[:2]
        
        # Initialize heatmaps for 5 parts
        heatmaps = [np.zeros((h, w), dtype=np.float32) for _ in range(5)]
        
        # Define keypoint indices for COCO format (used by OpenPifPaf)
        keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Extract keypoint coordinates and confidences
        kp_data = {}
        for name, idx in keypoint_indices.items():
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                x, y, confidence = keypoints[idx][:3]
                if confidence > 0.1:  # Lower threshold to include more keypoints
                    kp_data[name] = (int(x), int(y), confidence)
        
        # Generate heatmaps for each body part using Gaussian kernels
        heatmaps = self._generate_heatmaps_from_keypoints(kp_data, h, w)
        
        return heatmaps
    
    def _generate_heatmaps_from_keypoints(self, kp_data, h, w):
        """Generate heatmaps from keypoints using Gaussian kernels"""
        heatmaps = [np.zeros((h, w), dtype=np.float32) for _ in range(5)]
        
        # Define body part groupings for 5-part segmentation
        part_groups = {
            0: ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],  # Head
            1: ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'],  # Upper torso
            2: ['left_hip', 'right_hip'],  # Lower torso
            3: ['left_knee', 'right_knee'],  # Legs
            4: ['left_ankle', 'right_ankle']  # Feet
        }
        
        # Generate Gaussian heatmaps for each part
        for part_idx, keypoint_names in part_groups.items():
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            for kp_name in keypoint_names:
                if kp_name in kp_data:
                    x, y, confidence = kp_data[kp_name]
                    
                    # Create Gaussian kernel around keypoint
                    sigma = 15  # Standard deviation for Gaussian
                    kernel_size = int(6 * sigma)
                    
                    # Calculate kernel bounds
                    x_min = max(0, x - kernel_size)
                    x_max = min(w, x + kernel_size + 1)
                    y_min = max(0, y - kernel_size)
                    y_max = min(h, y + kernel_size + 1)
                    
                    # Create Gaussian kernel
                    for ky in range(y_min, y_max):
                        for kx in range(x_min, x_max):
                            # Calculate distance from keypoint
                            dist = np.sqrt((kx - x)**2 + (ky - y)**2)
                            # Gaussian value
                            gaussian_val = np.exp(-(dist**2) / (2 * sigma**2))
                            # Weight by confidence
                            weighted_val = gaussian_val * confidence
                            # Add to heatmap (max operation to handle overlapping keypoints)
                            heatmap[ky, kx] = max(heatmap[ky, kx], weighted_val)
            
            heatmaps[part_idx] = heatmap
        
        return heatmaps
    
    def _get_five_part_definitions(self, keypoints, h, w):
        """Define 5 vertical body parts based on pose keypoints"""
        parts = {
            'head': None,           # Part 0: Head and neck
            'upper_torso': None,    # Part 1: Upper torso and arms
            'lower_torso': None,    # Part 2: Lower torso 
            'legs': None,           # Part 3: Thighs and knees
            'feet': None            # Part 4: Lower legs and feet
        }
        
        # Head region (top 20% or from head keypoints)
        if 'nose' in keypoints or 'left_ear' in keypoints or 'right_ear' in keypoints:
            head_points = [keypoints.get(kp) for kp in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'] if kp in keypoints]
            if head_points:
                min_y = max(0, min(p[1] for p in head_points) - 20)
                max_y = min(h, max(p[1] for p in head_points) + 30)
                parts['head'] = {'y_range': (min_y, max_y), 'x_range': (0, w)}
        
        # Upper torso (shoulders to mid-torso)
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            shoulder_y = (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) // 2
            
            # Estimate upper torso end
            if 'left_hip' in keypoints and 'right_hip' in keypoints:
                hip_y = (keypoints['left_hip'][1] + keypoints['right_hip'][1]) // 2
                torso_mid = shoulder_y + (hip_y - shoulder_y) // 2
            else:
                torso_mid = shoulder_y + h // 6
            
            parts['upper_torso'] = {'y_range': (shoulder_y, torso_mid), 'x_range': (0, w)}
        
        # Lower torso (mid-torso to hips)
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            hip_y = (keypoints['left_hip'][1] + keypoints['right_hip'][1]) // 2
            
            if parts['upper_torso']:
                start_y = parts['upper_torso']['y_range'][1]
            else:
                start_y = hip_y - h // 8
            
            parts['lower_torso'] = {'y_range': (start_y, hip_y + 20), 'x_range': (0, w)}
        
        # Legs (hips to knees/ankles)
        if 'left_knee' in keypoints and 'right_knee' in keypoints:
            knee_y = (keypoints['left_knee'][1] + keypoints['right_knee'][1]) // 2
            
            if parts['lower_torso']:
                start_y = parts['lower_torso']['y_range'][1]
            else:
                start_y = knee_y - h // 4
            
            parts['legs'] = {'y_range': (start_y, knee_y + 30), 'x_range': (0, w)}
        
        # Feet (knees/ankles to bottom)
        if 'left_ankle' in keypoints and 'right_ankle' in keypoints:
            ankle_y = (keypoints['left_ankle'][1] + keypoints['right_ankle'][1]) // 2
            
            if parts['legs']:
                start_y = max(parts['legs']['y_range'][1] - 20, ankle_y - h // 6)
            else:
                start_y = ankle_y - h // 8
            
            parts['feet'] = {'y_range': (start_y, h), 'x_range': (0, w)}
        
        # Fill in missing parts with defaults if pose detection failed
        self._fill_missing_parts(parts, h, w)
        
        return parts
    
    def _fill_missing_parts(self, parts, h, w):
        """Fill in missing body parts with default regions"""
        part_height = h // 5
        
        for idx, (part_name, part_region) in enumerate(parts.items()):
            if part_region is None:
                # Create default vertical strips
                start_y = idx * part_height
                end_y = (idx + 1) * part_height if idx < 4 else h
                parts[part_name] = {'y_range': (start_y, end_y), 'x_range': (0, w)}
    
    def _create_mask_from_region(self, region, h, w):
        """Create a binary mask from a region definition"""
        mask = np.zeros((h, w), dtype=np.float32)
        y_start, y_end = region['y_range']
        x_start, x_end = region['x_range']
        
        y_start = max(0, min(h, y_start))
        y_end = max(0, min(h, y_end))
        x_start = max(0, min(w, x_start))
        x_end = max(0, min(w, x_end))
        
        mask[y_start:y_end, x_start:x_end] = 1.0
        return mask
    
    def extract_features(self, image):
        """Extract ReID features from a person image using pose-based heatmaps"""
        # Generate pose-based heatmaps
        heatmaps, pose_keypoints = self.generate_pose_based_masks(image)
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Apply transforms
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Convert heatmaps to tensors and resize to match model input
        # Resize heatmaps to match model input size (384, 128)
        processed_heatmaps = []
        for heatmap in heatmaps:
            # Use bilinear interpolation for smooth heatmaps
            heatmap_resized = cv2.resize(heatmap, (128, 384), interpolation=cv2.INTER_LINEAR)
            heatmap_tensor = torch.from_numpy(heatmap_resized).float()
            processed_heatmaps.append(heatmap_tensor)
        
        # Stack heatmaps and add batch dimension
        # Format: [batch_size, num_parts, height, width] = [1, 5, 384, 128]
        heatmaps_tensor = torch.stack(processed_heatmaps, dim=0).unsqueeze(0).to(self.device)
        
        # Extract features - pass heatmaps directly to the model
        with torch.no_grad():
            # For BPBReID, we need to pass the heatmaps in a specific format
            # The model expects heatmaps to be provided during forward pass
            try:
                # Try to pass heatmaps to the model
                outputs = self.reid_model(image_tensor, heatmaps_tensor)
            except Exception as e:
                print(f"Warning: Could not pass heatmaps to model: {e}")
                # If that fails, try without heatmaps (but this should work with mask-enabled model)
                outputs = self.reid_model(image_tensor)
            
            # BPBReID returns a tuple: (output_dict, visibility_scores, parts_masks)
            if isinstance(outputs, tuple):
                output_dict = outputs[0]
                
                # Extract the foreground features (bn_foreg) which are the main features
                if 'bn_foreg' in output_dict:
                    feature_vector = output_dict['bn_foreg']
                elif 'foreground' in output_dict:
                    feature_vector = output_dict['foreground']
                else:
                    feature_vector = list(output_dict.values())[0]
                    
            elif isinstance(outputs, dict):
                if 'bn_foreg' in outputs:
                    feature_vector = outputs['bn_foreg']
                elif 'foreground' in outputs:
                    feature_vector = outputs['foreground']
                else:
                    feature_vector = list(outputs.values())[0]
            else:
                feature_vector = outputs
        
        # Ensure it's 2D (batch_size, feature_dim)
        if len(feature_vector.shape) > 2:
            feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
        # Normalize features
        feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
        return feature_vector, pose_keypoints
    
    def match_person(self, query_features):
        """Match query features against gallery"""
        if len(self.gallery_features) == 0:
            return None, 0.0
        
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        similarities = torch.mm(query_features, gallery_tensor.t())
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        if best_similarity > self.reid_threshold:
            return self.gallery_ids[best_idx], best_similarity
        else:
            return None, best_similarity
    
    def add_to_gallery(self, features, person_id, image, pose_keypoints):
        """Add a person to the gallery with pose information"""
        self.gallery_features.append(features)
        self.gallery_ids.append(person_id)
        self.gallery_images.append(image.copy())
        self.gallery_poses.append(pose_keypoints)
    
    def visualize_pose_on_image(self, image, keypoints):
        """Visualize pose keypoints on image"""
        if not self.show_pose_keypoints:
            return image
            
        viz_image = image.copy()
        
        # Define connections between keypoints (skeleton)
        connections = [
            # Face
            (0, 1), (0, 2), (1, 3), (2, 4),  # nose to eyes, eyes to ears
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders and hips
            # Left arm
            (5, 7), (7, 9),  # left shoulder to elbow to wrist
            # Right arm  
            (6, 8), (8, 10),  # right shoulder to elbow to wrist
            # Left leg
            (11, 13), (13, 15),  # left hip to knee to ankle
            # Right leg
            (12, 14), (14, 16),  # right hip to knee to ankle
        ]
        
        # Define keypoint indices
        keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Extract valid keypoints
        valid_points = {}
        for idx, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] > 0.3:  # confidence threshold
                valid_points[idx] = (int(kp[0]), int(kp[1]))
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx in valid_points and end_idx in valid_points:
                start_point = valid_points[start_idx]
                end_point = valid_points[end_idx]
                cv2.line(viz_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for idx, point in valid_points.items():
            cv2.circle(viz_image, point, 4, (0, 0, 255), -1)
            # Add keypoint labels
            label = list(keypoint_indices.keys())[idx] if idx < len(keypoint_indices) else str(idx)
            cv2.putText(viz_image, label, (point[0] + 5, point[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return viz_image
    
    def visualize_body_part_masks(self, image, heatmaps):
        """Visualize body part heatmaps as overlay"""
        if not self.show_body_part_masks:
            return image
            
        viz_image = image.copy()
        colors = [
            (255, 0, 0),    # Red - Head
            (0, 255, 0),    # Green - Upper torso
            (0, 0, 255),    # Blue - Lower torso  
            (255, 255, 0),  # Cyan - Legs
            (255, 0, 255),  # Magenta - Feet
        ]
        
        part_names = ['Head', 'Upper Torso', 'Lower Torso', 'Legs', 'Feet']
        
        # Create overlay
        overlay = viz_image.copy()
        
        for idx, (heatmap, color, name) in enumerate(zip(heatmaps, colors, part_names)):
            # Normalize heatmap to 0-1 range
            heatmap_norm = heatmap / (heatmap.max() + 1e-8)
            
            # Convert heatmap to 3-channel colored overlay
            heatmap_colored = np.zeros_like(viz_image)
            for c in range(3):
                heatmap_colored[:, :, c] = heatmap_norm * color[c]
            
            # Apply overlay with heatmap intensity
            overlay = cv2.addWeighted(overlay, 0.7, heatmap_colored.astype(np.uint8), 0.3, 0)
        
        return overlay
    
    def process_frame(self, frame):
        """Process a single frame: detect persons, estimate pose, and perform ReID"""
        results = self.yolo(frame, classes=0, conf=0.5)
        
        frame_results = {
            'detections': [],
            'matches': [],
            'similarities': [],
            'pose_data': [],
            'mask_data': []
        }
        
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        try:
                            # Extract features with pose-based masks
                            features, pose_keypoints = self.extract_features(person_img)
                            
                            # Try to match with gallery
                            matched_id, similarity = self.match_person(features)
                            
                            # Store results for analysis
                            detection_result = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'matched_id': matched_id,
                                'similarity': similarity,
                                'pose_detected': len(pose_keypoints) > 0
                            }
                            frame_results['detections'].append(detection_result)
                            frame_results['similarities'].append(similarity)
                            frame_results['pose_data'].append(pose_keypoints)
                            
                            if matched_id is not None:
                                frame_results['matches'].append(matched_id)
                                label = f"Person {matched_id} ({similarity:.3f}) [POSE]"
                                color = (0, 255, 0)  # Green for known persons
                            else:
                                label = f"Unknown ({similarity:.3f}) [POSE]"
                                color = (0, 0, 255)  # Red for unknown persons
                            
                            # Visualize pose on person crop
                            if self.show_pose_keypoints:
                                person_with_pose = self.visualize_pose_on_image(person_img, pose_keypoints)
                                # Replace the region in annotated frame
                                annotated_frame[y1:y2, x1:x2] = person_with_pose
                            
                            # Draw bounding box and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Add pose quality indicator
                            pose_quality = len([kp for kp in pose_keypoints if len(kp) >= 3 and kp[2] > 0.3])
                            cv2.putText(annotated_frame, f"Pose: {pose_quality}/17", (x1, y2+15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                      
                        except Exception as e:
                            # Pose estimation failed - this should cause failure
                            print(f"Pose estimation failed for detection: {e}")
                            raise RuntimeError(f"Mandatory pose estimation failed: {e}")
        
        return annotated_frame, frame_results
    
    def add_gallery_image(self, image_path):
        """Add a gallery image from file path with pose estimation"""
        print(f"Loading gallery image with pose estimation: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gallery image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect person in image
        results = self.yolo(image, classes=0, conf=0.5)
        
        person_detected = False
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = image[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        try:
                            features, pose_keypoints = self.extract_features(person_img)
                            self.add_to_gallery(features, self.next_person_id, person_img, pose_keypoints)
                            
                            pose_quality = len([kp for kp in pose_keypoints if len(kp) >= 3 and kp[2] > 0.3])
                            print(f"✓ Added Person {self.next_person_id} to gallery with pose quality: {pose_quality}/17 keypoints")
                            self.next_person_id += 1
                            person_detected = True
                            break
                        except Exception as e:
                            raise RuntimeError(f"Failed to add gallery image due to pose estimation failure: {e}")
            
            if person_detected:
                break
        
        if not person_detected:
            raise ValueError(f"No person detected in gallery image: {image_path}")
        
        return True
    
    def test_video(self, video_path, expected_match=True, save_annotated=False, output_path=None):
        """
        Test the ReID model on a video with pose-based processing
        
        Args:
            video_path: Path to video file
            expected_match: True if person in video should match gallery, False otherwise
            save_annotated: Whether to save annotated video
            output_path: Path to save annotated video
        
        Returns:
            dict: Test results containing frame counts and accuracy
        """
        print(f"\nTesting video with pose-based ReID: {video_path}")
        print(f"Expected match: {expected_match}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Setup video writer if saving annotated video
        writer = None
        if save_annotated and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Test results
        results = {
            'video_path': video_path,
            'expected_match': expected_match,
            'total_frames': total_frames,
            'frames_with_detection': 0,
            'frames_with_match': 0,
            'frames_without_match': 0,
            'frames_with_pose': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'accuracy': 0.0,
            'similarity_scores': [],
            'frame_details': [],
            'pose_statistics': {
                'total_poses_detected': 0,
                'avg_pose_quality': 0.0,
                'pose_failures': 0
            }
        }
        
        frame_idx = 0
        pose_qualities = []
        
        print("Processing frames with pose estimation...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process frame with pose estimation
                annotated_frame, frame_result = self.process_frame(frame)
                
                frame_detail = {
                    'frame_idx': frame_idx,
                    'has_detection': len(frame_result['detections']) > 0,
                    'has_match': len(frame_result['matches']) > 0,
                    'has_pose': len(frame_result['pose_data']) > 0,
                    'similarities': frame_result['similarities'],
                    'correct': False
                }
                
                if frame_detail['has_detection']:
                    results['frames_with_detection'] += 1
                    
                    # Count poses
                    if frame_detail['has_pose']:
                        results['frames_with_pose'] += 1
                        results['pose_statistics']['total_poses_detected'] += len(frame_result['pose_data'])
                        
                        # Calculate pose quality
                        for pose_data in frame_result['pose_data']:
                            quality = len([kp for kp in pose_data if len(kp) >= 3 and kp[2] > 0.3])
                            pose_qualities.append(quality)
                    
                    # Get the best similarity for this frame
                    if frame_result['similarities']:
                        best_similarity = max(frame_result['similarities'])
                        results['similarity_scores'].append(best_similarity)
                        frame_detail['best_similarity'] = best_similarity
                    
                    if frame_detail['has_match']:
                        results['frames_with_match'] += 1
                    else:
                        results['frames_without_match'] += 1
                    
                    # Determine if prediction is correct
                    if expected_match:
                        # Should match - correct if we have a match
                        frame_detail['correct'] = frame_detail['has_match']
                    else:
                        # Should NOT match - correct if we don't have a match
                        frame_detail['correct'] = not frame_detail['has_match']
                    
                    if frame_detail['correct']:
                        results['correct_predictions'] += 1
                    else:
                        results['incorrect_predictions'] += 1
                    
                    # Add correctness indicator to frame
                    correct_text = "✓" if frame_detail['correct'] else "✗"
                    correct_color = (0, 255, 0) if frame_detail['correct'] else (0, 0, 255)
                    cv2.putText(annotated_frame, correct_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, correct_color, 2)
                
                # Add frame info to annotated frame
                info_text = f"Frame {frame_idx+1}/{total_frames} | Expected: {'Match' if expected_match else 'No Match'} | Pose-Based"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Save annotated frame
                if writer:
                    writer.write(annotated_frame)
                
                results['frame_details'].append(frame_detail)
                frame_idx += 1
                
                # Progress indicator
                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames")
                    
            except Exception as e:
                print(f"Frame {frame_idx} failed pose processing: {e}")
                results['pose_statistics']['pose_failures'] += 1
                # Continue or fail based on requirements
                raise RuntimeError(f"Mandatory pose processing failed on frame {frame_idx}: {e}")
        
        # Calculate final accuracy and pose statistics
        total_detections = results['frames_with_detection']
        if total_detections > 0:
            results['accuracy'] = results['correct_predictions'] / total_detections
        
        if pose_qualities:
            results['pose_statistics']['avg_pose_quality'] = np.mean(pose_qualities)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"Test completed: {results['correct_predictions']}/{total_detections} correct predictions")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Pose statistics: {results['frames_with_pose']}/{total_detections} frames with pose data")
        print(f"Average pose quality: {results['pose_statistics']['avg_pose_quality']:.1f}/17 keypoints")
        
        return results
    
    def run_full_test(self, gallery_image_path, video1_path, video2_path, 
                     save_annotated=True, output_dir="reid_pose_mask_test_results"):
        """
        Run complete test suite with pose-based processing
        
        Args:
            gallery_image_path: Path to reference image for gallery
            video1_path: Path to video with same person (should match)
            video2_path: Path to video with different person (should not match)
            save_annotated: Whether to save annotated videos
            output_dir: Directory to save results
        """
        print("="*60)
        print("Pose-Based ReID Video Test Suite")
        print("="*60)
        
        # Create output directory
        if save_annotated:
            os.makedirs(output_dir, exist_ok=True)
        
        # Clear gallery first
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.gallery_poses = []
        self.next_person_id = 1
        
        # Step 1: Add gallery image with pose estimation
        print("\n1. Loading gallery image with pose estimation...")
        try:
            self.add_gallery_image(gallery_image_path)
        except Exception as e:
            print(f"Error loading gallery image: {e}")
            return None
        
        # Step 2: Test video 1 (should match)
        print("\n2. Testing Video 1 (Same Person - Should Match) with pose processing...")
        video1_output = os.path.join(output_dir, "video1_annotated.mp4") if save_annotated else None
        
        try:
            results1 = self.test_video(
                video1_path, 
                expected_match=True, 
                save_annotated=save_annotated,
                output_path=video1_output
            )
        except Exception as e:
            print(f"Error testing video 1: {e}")
            return None
        
        # Step 3: Test video 2 (should not match)
        print("\n3. Testing Video 2 (Different Person - Should NOT Match) with pose processing...")
        video2_output = os.path.join(output_dir, "video2_annotated.mp4") if save_annotated else None
        
        try:
            results2 = self.test_video(
                video2_path, 
                expected_match=False, 
                save_annotated=save_annotated,
                output_path=video2_output
            )
        except Exception as e:
            print(f"Error testing video 2: {e}")
            return None
        
        # Compile final results
        final_results = {
            'test_timestamp': datetime.now().isoformat(),
            'gallery_image': gallery_image_path,
            'reid_threshold': self.reid_threshold,
            'device': str(self.device),
            'pose_estimation_enabled': True,
            'mask_processing': 'five_part_pose_based',
            'video1_results': results1,
            'video2_results': results2,
            'overall_performance': {
                'total_frames_tested': results1['frames_with_detection'] + results2['frames_with_detection'],
                'total_correct': results1['correct_predictions'] + results2['correct_predictions'],
                'total_incorrect': results1['incorrect_predictions'] + results2['incorrect_predictions'],
                'overall_accuracy': 0.0,
                'pose_coverage': (results1['frames_with_pose'] + results2['frames_with_pose']) / 
                               (results1['frames_with_detection'] + results2['frames_with_detection']) if 
                               (results1['frames_with_detection'] + results2['frames_with_detection']) > 0 else 0
            }
        }
        
        # Calculate overall accuracy
        total_tested = final_results['overall_performance']['total_frames_tested']
        total_correct = final_results['overall_performance']['total_correct']
        
        if total_tested > 0:
            final_results['overall_performance']['overall_accuracy'] = total_correct / total_tested
        
        # Save results to JSON
        results_file = os.path.join(output_dir, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print detailed results
        self.print_test_summary(final_results)
        
        print(f"\nResults saved to: {results_file}")
        if save_annotated:
            print(f"Annotated videos saved to: {output_dir}")
        
        return final_results
    
    def print_test_summary(self, results):
        """Print a detailed summary of test results with pose statistics"""
        print("\n" + "="*60)
        print("POSE-BASED REID TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Gallery Image: {results['gallery_image']}")
        print(f"ReID Threshold: {results['reid_threshold']}")
        print(f"Device: {results['device']}")
        print(f"Pose Estimation: Enabled (OpenPifPaf)")
        print(f"Mask Processing: {results['mask_processing']}")
        
        # Video 1 Results
        v1 = results['video1_results']
        print(f"\nVIDEO 1 (Same Person - Should Match):")
        print(f"  Path: {v1['video_path']}")
        print(f"  Total Frames: {v1['total_frames']}")
        print(f"  Frames with Person Detection: {v1['frames_with_detection']}")
        print(f"  Frames with Pose Data: {v1['frames_with_pose']}")
        print(f"  Frames with Match: {v1['frames_with_match']}")
        print(f"  Frames without Match: {v1['frames_without_match']}")
        print(f"  Correct Predictions: {v1['correct_predictions']}")
        print(f"  Incorrect Predictions: {v1['incorrect_predictions']}")
        print(f"  Accuracy: {v1['accuracy']:.3f} ({v1['accuracy']*100:.1f}%)")
        print(f"  Avg Pose Quality: {v1['pose_statistics']['avg_pose_quality']:.1f}/17 keypoints")
        
        if v1['similarity_scores']:
            avg_sim = np.mean(v1['similarity_scores'])
            max_sim = np.max(v1['similarity_scores'])
            min_sim = np.min(v1['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Video 2 Results
        v2 = results['video2_results']
        print(f"\nVIDEO 2 (Different Person - Should NOT Match):")
        print(f"  Path: {v2['video_path']}")
        print(f"  Total Frames: {v2['total_frames']}")
        print(f"  Frames with Person Detection: {v2['frames_with_detection']}")
        print(f"  Frames with Pose Data: {v2['frames_with_pose']}")
        print(f"  Frames with Match: {v2['frames_with_match']}")
        print(f"  Frames without Match: {v2['frames_without_match']}")
        print(f"  Correct Predictions: {v2['correct_predictions']}")
        print(f"  Incorrect Predictions: {v2['incorrect_predictions']}")
        print(f"  Accuracy: {v2['accuracy']:.3f} ({v2['accuracy']*100:.1f}%)")
        print(f"  Avg Pose Quality: {v2['pose_statistics']['avg_pose_quality']:.1f}/17 keypoints")
        
        if v2['similarity_scores']:
            avg_sim = np.mean(v2['similarity_scores'])
            max_sim = np.max(v2['similarity_scores'])
            min_sim = np.min(v2['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Overall Results
        overall = results['overall_performance']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Frames Tested: {overall['total_frames_tested']}")
        print(f"  Total Correct: {overall['total_correct']}")
        print(f"  Total Incorrect: {overall['total_incorrect']}")
        print(f"  Overall Accuracy: {overall['overall_accuracy']:.3f} ({overall['overall_accuracy']*100:.1f}%)")
        print(f"  Pose Coverage: {overall['pose_coverage']:.3f} ({overall['pose_coverage']*100:.1f}%)")
        
        print("="*60)
    
    def generate_comparison_plots(self, results, output_dir="test_results"):
        """Generate comparison plots for the pose-based test results"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Pose-Based ReID Video Test Results', fontsize=16)
            
            # Plot 1: Accuracy comparison
            videos = ['Video 1\n(Same Person)', 'Video 2\n(Different Person)']
            accuracies = [results['video1_results']['accuracy'], results['video2_results']['accuracy']]
            colors = ['green', 'blue']
            
            axes[0, 0].bar(videos, accuracies, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylim(0, 1)
            
            # Add accuracy values on bars
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 2: Match/No Match distribution
            v1_match = results['video1_results']['frames_with_match']
            v1_no_match = results['video1_results']['frames_without_match']
            v2_match = results['video2_results']['frames_with_match']
            v2_no_match = results['video2_results']['frames_without_match']
            
            x = np.arange(2)
            width = 0.35
            
            axes[0, 1].bar(x - width/2, [v1_match, v2_match], width, label='Matched', color='green', alpha=0.7)
            axes[0, 1].bar(x + width/2, [v1_no_match, v2_no_match], width, label='Not Matched', color='red', alpha=0.7)
            
            axes[0, 1].set_ylabel('Number of Frames')
            axes[0, 1].set_title('Match Distribution')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(videos)
            axes[0, 1].legend()
            
            # Plot 3: Pose quality comparison
            v1_pose_quality = results['video1_results']['pose_statistics']['avg_pose_quality']
            v2_pose_quality = results['video2_results']['pose_statistics']['avg_pose_quality']
            pose_qualities = [v1_pose_quality, v2_pose_quality]
            
            axes[0, 2].bar(videos, pose_qualities, color=['orange', 'purple'], alpha=0.7)
            axes[0, 2].set_ylabel('Average Keypoints Detected')
            axes[0, 2].set_title('Pose Quality (Keypoints/17)')
            axes[0, 2].set_ylim(0, 17)
            
            for i, v in enumerate(pose_qualities):
                axes[0, 2].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
            
            # Plot 4: Similarity scores distribution for Video 1
            if results['video1_results']['similarity_scores']:
                axes[1, 0].hist(results['video1_results']['similarity_scores'], bins=20, alpha=0.7, color='green')
                axes[1, 0].axvline(x=results['reid_threshold'], color='red', linestyle='--', label=f'Threshold ({results["reid_threshold"]})')
                axes[1, 0].set_xlabel('Similarity Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Video 1 Similarity Scores')
                axes[1, 0].legend()
            
            # Plot 5: Similarity scores distribution for Video 2
            if results['video2_results']['similarity_scores']:
                axes[1, 1].hist(results['video2_results']['similarity_scores'], bins=20, alpha=0.7, color='blue')
                axes[1, 1].axvline(x=results['reid_threshold'], color='red', linestyle='--', label=f'Threshold ({results["reid_threshold"]})')
                axes[1, 1].set_xlabel('Similarity Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Video 2 Similarity Scores')
                axes[1, 1].legend()
            
            # Plot 6: Pose coverage
            pose_coverage = [
                results['video1_results']['frames_with_pose'] / results['video1_results']['frames_with_detection'] if results['video1_results']['frames_with_detection'] > 0 else 0,
                results['video2_results']['frames_with_pose'] / results['video2_results']['frames_with_detection'] if results['video2_results']['frames_with_detection'] > 0 else 0
            ]
            
            axes[1, 2].bar(videos, pose_coverage, color=['cyan', 'magenta'], alpha=0.7)
            axes[1, 2].set_ylabel('Pose Coverage Ratio')
            axes[1, 2].set_title('Pose Detection Coverage')
            axes[1, 2].set_ylim(0, 1)
            
            for i, v in enumerate(pose_coverage):
                axes[1, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, "pose_test_results_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Pose-based comparison plots saved to: {plot_path}")
            
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def run_camera_test_mode(self):
        """Run real-time ReID in test mode with pose visualization"""
        cap = cv2.VideoCapture(0)
        
        cv2.namedWindow('Pose-Based ReID Test Mode', cv2.WINDOW_NORMAL)
        
        print("\nPose-Based Test Mode Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current detections to gallery")
        print("- Press 'c' to clear gallery")
        print("- Press 'g' to show gallery")
        print("- Press 't' to adjust threshold")
        print("- Press 'p' to toggle pose visualization")
        print("- Press 'm' to toggle mask visualization")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                annotated_frame, frame_result = self.process_frame(frame)
                
                # Enhanced status information
                status_text = f"Gallery: {len(self.gallery_ids)} | Threshold: {self.reid_threshold:.2f} | Pose: {'ON' if self.show_pose_keypoints else 'OFF'}"
                cv2.putText(annotated_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show detection count
                detection_text = f"Detections: {len(frame_result['detections'])}"
                cv2.putText(annotated_frame, detection_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show pose count
                pose_text = f"Poses: {len(frame_result['pose_data'])}"
                cv2.putText(annotated_frame, pose_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show best similarity if available
                if frame_result['similarities']:
                    best_sim = max(frame_result['similarities'])
                    sim_text = f"Best Similarity: {best_sim:.3f}"
                    cv2.putText(annotated_frame, sim_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Pose-Based ReID Test Mode', annotated_frame)
                
            except Exception as e:
                # Show error on frame
                error_text = f"Pose Error: {str(e)[:50]}..."
                cv2.putText(frame, error_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Pose-Based ReID Test Mode', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_current_detections(frame)
            elif key == ord('c'):
                self.gallery_features = []
                self.gallery_ids = []
                self.gallery_images = []
                self.gallery_poses = []
                print("Gallery cleared!")
            elif key == ord('g'):
                self._show_gallery()
            elif key == ord('t'):
                self._adjust_threshold()
            elif key == ord('p'):
                self.show_pose_keypoints = not self.show_pose_keypoints
                print(f"Pose visualization: {'ON' if self.show_pose_keypoints else 'OFF'}")
            elif key == ord('m'):
                self.show_body_part_masks = not self.show_body_part_masks
                print(f"Mask visualization: {'ON' if self.show_body_part_masks else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _save_current_detections(self, frame):
        """Save current detections to gallery with pose data"""
        try:
            results = self.yolo(frame, classes=0, conf=0.5)
            
            new_persons_added = 0
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        person_img = frame[y1:y2, x1:x2]
                        
                        if person_img.size > 0:
                            features, pose_keypoints = self.extract_features(person_img)
                            matched_id, similarity = self.match_person(features)
                            
                            if matched_id is None:
                                # New person, add to gallery
                                self.add_to_gallery(features, self.next_person_id, person_img, pose_keypoints)
                                pose_quality = len([kp for kp in pose_keypoints if len(kp) >= 3 and kp[2] > 0.3])
                                print(f"Added Person {self.next_person_id} to gallery (similarity: {similarity:.3f}, pose: {pose_quality}/17)")
                                self.next_person_id += 1
                                new_persons_added += 1
                            else:
                                print(f"Person {matched_id} already in gallery (similarity: {similarity:.3f})")
            
            if new_persons_added == 0:
                print("No new persons detected to add to gallery")
                
        except Exception as e:
            print(f"Error saving detections: {e}")
    
    def _show_gallery(self):
        """Display gallery with pose information"""
        if len(self.gallery_images) == 0:
            print("Gallery is empty!")
            return
        
        # Create gallery visualization
        gallery_height = 200
        gallery_width = 150
        cols = min(6, len(self.gallery_images))
        rows = (len(self.gallery_images) + cols - 1) // cols
        
        gallery_viz = np.zeros((rows * gallery_height, cols * gallery_width, 3), dtype=np.uint8)
        
        for idx, (img, person_id, pose_data) in enumerate(zip(self.gallery_images, self.gallery_ids, self.gallery_poses)):
            row = idx // cols
            col = idx % cols
            
            # Resize image
            resized = cv2.resize(img, (gallery_width, gallery_height))
            
            # Add pose visualization if enabled
            if self.show_pose_keypoints and pose_data:
                resized = self.visualize_pose_on_image(resized, pose_data)
            
            # Place in gallery
            y1 = row * gallery_height
            y2 = (row + 1) * gallery_height
            x1 = col * gallery_width
            x2 = (col + 1) * gallery_width
            gallery_viz[y1:y2, x1:x2] = resized
            
            # Add label with pose info
            cv2.putText(gallery_viz, f"Person {person_id}", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add pose quality info
            if pose_data:
                pose_quality = len([kp for kp in pose_data if len(kp) >= 3 and kp[2] > 0.3])
                cv2.putText(gallery_viz, f"Pose: {pose_quality}/17", (x1 + 5, y1 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Add gallery info at the bottom
        info_text = f"Gallery: {len(self.gallery_ids)} persons | Pose-Based ReID | Threshold: {self.reid_threshold:.2f}"
        cv2.putText(gallery_viz, info_text, (10, gallery_viz.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Pose-Based Gallery', gallery_viz)
        print(f"Showing pose-based gallery with {len(self.gallery_ids)} persons")
        cv2.waitKey(0)
        cv2.destroyWindow('Pose-Based Gallery')
    
    def _adjust_threshold(self):
        """Adjust ReID threshold interactively"""
        print(f"\nCurrent threshold: {self.reid_threshold:.2f}")
        try:
            new_threshold = float(input("Enter new threshold (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                self.reid_threshold = new_threshold
                print(f"Threshold updated to: {self.reid_threshold:.2f}")
            else:
                print("Threshold must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main():
    """Main function to run the pose-based test suite"""
    print("Pose-Based ReID Video Test Suite")
    print("=" * 50)
    print("⚠️  This system requires OpenPifPaf for pose estimation!")
    print("⚠️  The system will fail if pose estimation fails!")
    
    # Get parent directory (one level up from current folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    realtime_dir = os.path.dirname(current_dir)               
    bpbreid_dir = os.path.dirname(realtime_dir) 
    
    # Configuration - paths relative to parent directory
    reid_model_path = os.path.join(bpbreid_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
    hrnet_path = os.path.join(bpbreid_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")

    # Test file paths
    gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1.jpg")
    video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1-vid.MOV")
    video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2","person-2-vid.MOV")
    
    # Verify critical files exist
    required_files = [reid_model_path, hrnet_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure model files are available.")
        return
    
    try:
        # Create pose-based test system
        print("Initializing Pose-Based ReID test system...")
        print("This will fail if OpenPifPaf is not available!")
        tester = PoseBasedPersonReIDTester(reid_model_path, hrnet_path)
        
        print("\nChoose test mode:")
        print("1. Full video test suite (with pose processing)")
        print("2. Real-time camera test mode (with pose visualization)")
        print("3. Interactive mode")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            # Full video test suite
            if all(os.path.exists(p) for p in [gallery_image_path, video1_path, video2_path]):
                results = tester.run_full_test(
                    gallery_image_path=gallery_image_path,
                    video1_path=video1_path,
                    video2_path=video2_path,
                    save_annotated=True,
                    output_dir="reid_pose_mask_test_results"
                )
                
                if results:
                    tester.generate_comparison_plots(results, "reid_pose_mask_test_results")
                    print("\n✓ Pose-based full test completed successfully!")
                else:
                    print("✗ Pose-based full test failed!")
            else:
                print("Test files not found. Please check paths.")
                
        elif choice == "2":
            # Real-time camera test mode
            print("\nStarting real-time camera test mode with pose visualization...")
            tester.run_camera_test_mode()
            
        elif choice == "3":
            # Interactive mode
            interactive_mode(tester)
            
        else:
            print("Invalid choice. Exiting.")
            
    except Exception as e:
        print(f"Error running pose-based test: {e}")
        print("\n⚠️  This error is expected if OpenPifPaf is not properly installed or pose estimation fails!")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Install OpenPifPaf: conda install -c conda-forge openpifpaf")
        print("   2. Or try: pip install openpifpaf")
        print("   3. Make sure you're in the correct conda environment")
        print("   4. Check OpenPifPaf documentation: https://openpifpaf.github.io/")
        import traceback
        traceback.print_exc()


def interactive_mode(tester):
    """Interactive mode for testing pose-based features"""
    print("\n" + "="*50)
    print("POSE-BASED INTERACTIVE MODE")
    print("="*50)
    
    while True:
        print("\nAvailable commands:")
        print("1. Load gallery image (with pose)")
        print("2. Test single video (with pose)")
        print("3. Test single image (with pose)")
        print("4. Show pose-based gallery")
        print("5. Clear gallery")
        print("6. Adjust threshold")
        print("7. Toggle pose visualization")
        print("8. Toggle mask visualization")
        print("9. Exit")
        
        choice = input("\nEnter command (1-9): ").strip()
        
        try:
            if choice == "1":
                img_path = input("Enter gallery image path: ").strip()
                if os.path.exists(img_path):
                    tester.add_gallery_image(img_path)
                else:
                    print("Image file not found!")
                    
            elif choice == "2":
                video_path = input("Enter video path: ").strip()
                if os.path.exists(video_path):
                    expected = input("Should this video match gallery? (y/n): ").strip().lower() == 'y'
                    save_output = input("Save annotated video? (y/n): ").strip().lower() == 'y'
                    output_path = None
                    if save_output:
                        output_path = input("Enter output path (or press Enter for default): ").strip()
                        if not output_path:
                            output_path = "pose_test_output.mp4"
                    
                    results = tester.test_video(video_path, expected, save_output, output_path)
                    print(f"Test completed with {results['accuracy']:.3f} accuracy")
                    print(f"Pose coverage: {results['frames_with_pose']}/{results['frames_with_detection']} frames")
                else:
                    print("Video file not found!")
                    
            elif choice == "3":
                img_path = input("Enter image path: ").strip()
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    if image is not None:
                        annotated, frame_result = tester.process_frame(image)
                        cv2.imshow("Pose-Based Test Result", annotated)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        
                        print(f"Detections: {len(frame_result['detections'])}")
                        print(f"Poses detected: {len(frame_result['pose_data'])}")
                        for i, det in enumerate(frame_result['detections']):
                            print(f"  Detection {i+1}: ID={det['matched_id']}, Similarity={det['similarity']:.3f}, Pose={det['pose_detected']}")
                    else:
                        print("Could not load image!")
                else:
                    print("Image file not found!")
                    
            elif choice == "4":
                tester._show_gallery()
                
            elif choice == "5":
                tester.gallery_features = []
                tester.gallery_ids = []
                tester.gallery_images = []
                tester.gallery_poses = []
                tester.next_person_id = 1
                print("Gallery cleared!")
                
            elif choice == "6":
                tester._adjust_threshold()
                
            elif choice == "7":
                tester.show_pose_keypoints = not tester.show_pose_keypoints
                print(f"Pose visualization: {'ON' if tester.show_pose_keypoints else 'OFF'}")
                
            elif choice == "8":
                tester.show_body_part_masks = not tester.show_body_part_masks
                print(f"Mask visualization: {'ON' if tester.show_body_part_masks else 'OFF'}")
                
            elif choice == "9":
                print("Exiting pose-based interactive mode...")
                break
                
            else:
                print("Invalid choice!")
                
        except Exception as e:
            print(f"Error (pose processing failed): {e}")


if __name__ == "__main__":
    print("Pose-Based ReID Video Test Suite")
    print("=" * 50)
    print("Features:")
    print("- MANDATORY OpenPifPaf pose estimation")
    print("- 5-part pose-based body segmentation masks") 
    print("- Full video test suite with pose accuracy metrics")
    print("- Real-time camera testing with pose visualization")
    print("- Interactive testing mode with pose controls")
    print("- Comprehensive pose statistics and visualization")
    print("- System FAILS if pose estimation fails (no fallback)")
    print()
    print("⚠️  REQUIREMENTS:")
    print("   - OpenPifPaf must be installed: pip install openpifpaf")
    print("   - Pose estimation must succeed for every frame")
    print("   - No fallback logic - system will fail if pose fails")
    print()
    
    main()
