#!/usr/bin/env python3
"""
Improved BPBreID Re-Identification with YOLO Detection and Pose-Based Masking

This version incorporates the successful techniques from corrected_masked_reid_test_default.py:
1. Proper gallery image processing with YOLO detection and cropping
2. Corrected feature extraction and normalization
3. Combined similarity metrics
4. Better feature key selection
5. Consistent processing for both gallery and query images
6. Visual mask overlay on detected persons
7. YOLO Pose-based masking with 5 body sections
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import sys
from pathlib import Path
import time
from PIL import Image
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import torchreid
from torchvision import transforms
import json
from datetime import datetime
from types import SimpleNamespace

# YOLO Pose is built into ultralytics, so no additional imports needed

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

class ImprovedBPBreIDYOLOMaskedReID:
    """
    Improved BPBreID re-identification system with corrected processing
    """
    
    def __init__(self, reid_model_path, hrnet_path, yolo_model_path='yolov8n-pose.pt', 
             keypoint_confidence_threshold=0.5, person_detection_threshold=0.6):  # Add this parameter
        """
        Initialize the improved BPBreID re-identification system
        
        Args:
            reid_model_path: Path to BPBreID model weights
            hrnet_path: Path to HRNet pretrained weights
            yolo_model_path: Path to YOLO model weights
        """
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO for person detection and pose estimation
        print("Loading YOLO model for detection and pose estimation...")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize BPBreID model using corrected configuration
        print("Loading BPBreID model with corrected configuration...")
        self.config = self._create_corrected_config(reid_model_path, hrnet_path)
        self.model = self._load_corrected_model()
        
        # Setup transforms for BPBreID
        self.setup_transforms()
        
        # Gallery storage
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # ReID threshold (from corrected version)
        self.reid_threshold = 0.46
        
        # Test results storage
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'video_path': '',
            'gallery_path': '',
            'total_frames': 0,
            'frames_with_detection': 0,
            'frames_with_match': 0,
            'frames_without_match': 0,
            'similarity_scores': [],
            'reid_threshold': self.reid_threshold
        }
        
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self.person_detection_threshold = person_detection_threshold
        
        print("Improved BPBreID YOLO Masked ReID system initialized successfully")
    
    def _create_corrected_config(self, model_path, hrnet_path):
        """Create corrected BPBreID configuration (from corrected version)"""
        config = SimpleNamespace()
        
        # Model configuration
        config.model = SimpleNamespace()
        config.model.name = 'bpbreid'
        config.model.load_weights = model_path
        config.model.pretrained = True
        
        # BPBreid configuration - use settings from corrected version
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'  # Important: use identity normalization
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.test_use_target_segmentation = 'soft'
        config.model.bpbreid.testing_binary_visibility_score = True
        config.model.bpbreid.training_binary_visibility_score = True
        config.model.bpbreid.mask_filtering_testing = True
        config.model.bpbreid.mask_filtering_training = True
        
        # Mask configuration - keep 5 parts
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.masks.preprocess = 'five_v'
        config.model.bpbreid.masks.softmax_weight = 1.0
        config.model.bpbreid.masks.background_computation_strategy = 'threshold'
        config.model.bpbreid.masks.mask_filtering_threshold = 0.3
        
        # Data configuration
        config.data = SimpleNamespace()
        config.data.height = 384
        config.data.width = 128
        config.data.norm_mean = [0.485, 0.456, 0.406]
        config.data.norm_std = [0.229, 0.224, 0.225]
        
        # Loss configuration
        config.loss = SimpleNamespace()
        config.loss.name = 'part_based'
        
        return config
    
    def _load_corrected_model(self):
        """Load BPBreID model with corrected loading (from corrected version)"""
        print("Loading corrected BPBreid model...")
        
        try:
            # Import BPBreID directly since build_model doesn't pass config parameter
            print("Importing BPBreID model directly...")
            
            # Add the correct path to torchreid models
            torchreid_path = str(Path(__file__).parent.parent.parent / "torchreid")
            if torchreid_path not in sys.path:
                sys.path.append(torchreid_path)
                print(f"Added torchreid path: {torchreid_path}")
            
            from models.bpbreid import bpbreid
            print("✅ BPBreID model imported successfully")
            
            # Create configuration
            print("Creating BPBreID configuration...")
            from types import SimpleNamespace
            config = SimpleNamespace()
            config.model = SimpleNamespace()
            config.model.bpbreid = SimpleNamespace()
            
            # Set configuration values
            config.model.bpbreid.backbone = 'hrnet32'
            config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(self.config.model.bpbreid.hrnet_pretrained_path) + '/'
            print(f"HRNet pretrained path: {config.model.bpbreid.hrnet_pretrained_path}")
            print(f"HRNet path exists: {os.path.exists(config.model.bpbreid.hrnet_pretrained_path)}")
            config.model.bpbreid.pooling = 'gwap'
            config.model.bpbreid.normalization = 'identity'
            config.model.bpbreid.dim_reduce = 'after_pooling'
            config.model.bpbreid.dim_reduce_output = 512
            config.model.bpbreid.last_stride = 1
            config.model.bpbreid.shared_parts_id_classifier = False
            config.model.bpbreid.learnable_attention_enabled = True
            config.model.bpbreid.test_use_target_segmentation = 'soft'
            config.model.bpbreid.testing_binary_visibility_score = True
            config.model.bpbreid.training_binary_visibility_score = True
            config.model.bpbreid.mask_filtering_testing = True
            config.model.bpbreid.mask_filtering_training = True
            
            # Mask configuration
            config.model.bpbreid.masks = SimpleNamespace()
            config.model.bpbreid.masks.parts_num = 5
            config.model.bpbreid.masks.preprocess = 'five_v'
            config.model.bpbreid.masks.softmax_weight = 1.0
            config.model.bpbreid.masks.background_computation_strategy = 'threshold'
            config.model.bpbreid.masks.mask_filtering_threshold = 0.3
            
            # Create BPBreID model directly
            print("Creating BPBreID model...")
            model = bpbreid(
                num_classes=751,
                loss='part_based',
                pretrained=True,
                config=config
            )
            
            print("✅ BPBreID model created successfully")
            print(f"Model type: {type(model).__name__}")
            print(f"Model class: {model.__class__.__name__}")
            
            # Check if it's actually BPBreID
            if 'BPBreID' not in str(type(model)):
                print("❌ WARNING: Model is not BPBreID! This might be a fallback to ResNet")
                raise TypeError("Model is not BPBreID")
            
            # Load weights
            print(f"Loading weights from: {self.config.model.load_weights}")
            checkpoint = torch.load(self.config.model.load_weights, map_location=self.device)
            
            # Handle state dict
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
            
            # Load state dict with strict=False
            print("Loading state dict...")
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            
            model = model.to(self.device)
            model.eval()
            
            print("✅ Corrected BPBreid model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Error loading corrected BPBreid model: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def setup_transforms(self):
        """Setup transforms for BPBreID input preprocessing"""
        self.transform = transforms.Compose([
            transforms.Resize((self.config.data.height, self.config.data.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.data.norm_mean, std=self.config.data.norm_std)
        ])
        print(f"Transforms setup for {self.config.data.height}x{self.config.data.width}")
    

    
    def generate_yolo_pose_masks(self, person_img):
        """Generate pose-based masks using YOLO Pose skeleton structure with 5 body sections:
        1. Head
        2. Upper body (upper half of torso + upper arms)
        3. Lower body (lower half of torso + lower arms)
        4. Upper legs (thighs and upper calf - stops at 75% to ankle)
        5. Foot (lower calf from 75% + ankle + foot area)
        """
        
        try:
            # Run YOLO pose estimation
            results = self.yolo(person_img, task='pose')
            
            # Check if results exist and have keypoints
            if len(results) == 0:
                return None
            
            if not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
                return None
            
            if len(results[0].keypoints.data) == 0:
                return None
            
            # Use the first pose detection
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: [17, 3] - (x, y, confidence)
            
            # Validate keypoints shape
            if keypoints.shape[0] != 17:
                return None
            
            # Create 5-part masks from skeleton
            h, w = person_img.shape[:2]
            feat_h, feat_w = self.config.data.height // 8, self.config.data.width // 8
            
            # Initialize temporary masks for each part (before priority assignment)
            temp_masks = torch.zeros(6, feat_h, feat_w)  # 5 parts + will add background later
            
            # Scale factors for keypoint coordinates
            scale_x = feat_w / w
            scale_y = feat_h / h
            
            # Helper function to draw thick line on mask
            def draw_skeleton_line(mask, kp1_idx, kp2_idx, thickness=1):
                """Draw a thick line between two keypoints on the mask"""
                if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints) and 
                    keypoints[kp1_idx, 2] > self.keypoint_confidence_threshold and keypoints[kp2_idx, 2] > self.keypoint_confidence_threshold):
                    
                    x1 = int(keypoints[kp1_idx, 0] * scale_x)
                    y1 = int(keypoints[kp1_idx, 1] * scale_y)
                    x2 = int(keypoints[kp2_idx, 0] * scale_x)
                    y2 = int(keypoints[kp2_idx, 1] * scale_y)
                    
                    # Clip coordinates
                    x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                    y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                    
                    # Create temporary image for line drawing
                    temp_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
                    cv2.line(temp_mask, (x1, y1), (x2, y2), 1.0, thickness)
                    
                    return temp_mask
                return np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Helper function to fill polygon area
            def fill_polygon(keypoint_indices):
                """Fill the polygon formed by given keypoints"""
                valid_points = []
                for idx in keypoint_indices:
                    if idx < len(keypoints) and keypoints[idx, 2] > self.keypoint_confidence_threshold:
                        x = int(keypoints[idx, 0] * scale_x)
                        y = int(keypoints[idx, 1] * scale_y)
                        x = np.clip(x, 0, feat_w - 1)
                        y = np.clip(y, 0, feat_h - 1)
                        valid_points.append([x, y])
                
                if len(valid_points) >= 3:
                    temp_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
                    points = np.array(valid_points, dtype=np.int32)
                    cv2.fillPoly(temp_mask, [points], 1.0)
                    return temp_mask
                return np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Helper function to fill area between two keypoints
            def fill_area_between(kp1_idx, kp2_idx, width=1):  # Default width reduced from 2 to 1
                """Fill the area around a line between two keypoints"""
                if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints) and 
                    keypoints[kp1_idx, 2] > self.keypoint_confidence_threshold and keypoints[kp2_idx, 2] > self.keypoint_confidence_threshold):
                    
                    x1 = int(keypoints[kp1_idx, 0] * scale_x)
                    y1 = int(keypoints[kp1_idx, 1] * scale_y)
                    x2 = int(keypoints[kp2_idx, 0] * scale_x)
                    y2 = int(keypoints[kp2_idx, 1] * scale_y)
                    
                    # Clip coordinates
                    x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                    y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                    
                    temp_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
                    cv2.line(temp_mask, (x1, y1), (x2, y2), 1.0, width)
                    
                    # Use smaller dilation kernel to prevent connection between legs
                    kernel_size = max(1, width)
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # Make odd for symmetry
                    # Use smaller kernel - maximum of 2x2 instead of width-based
                    kernel_size = min(kernel_size, 2)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)
                    
                    return temp_mask
                return np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Part 1: Head (index 1)
            head_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Create head area around nose keypoint
            if keypoints[0, 2] > self.keypoint_confidence_threshold:  # Nose
                nose_x = int(keypoints[0, 0] * scale_x)
                nose_y = int(keypoints[0, 1] * scale_y)
                nose_x = np.clip(nose_x, 0, feat_w - 1)
                nose_y = np.clip(nose_y, 0, feat_h - 1)
                
                # Create circular head area
                head_radius = 4
                y_coords, x_coords = np.ogrid[:feat_h, :feat_w]
                head_mask = ((x_coords - nose_x)**2 + (y_coords - nose_y)**2 <= head_radius**2).astype(np.float32)
                
                # Add connections to eyes and ears if available
                head_mask += draw_skeleton_line(head_mask, 0, 1, thickness=1)  # nose to left eye
                head_mask += draw_skeleton_line(head_mask, 0, 2, thickness=1)  # nose to right eye
                head_mask += draw_skeleton_line(head_mask, 1, 3, thickness=1)  # left eye to left ear
                head_mask += draw_skeleton_line(head_mask, 2, 4, thickness=1)  # right eye to right ear
            
            temp_masks[1] = torch.from_numpy(np.clip(head_mask, 0, 1))
            
            # Part 2: Upper body (upper half of torso + upper arms) (index 2)
            upper_body_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Get torso keypoints
            if (keypoints[5, 2] > self.keypoint_confidence_threshold and keypoints[6, 2] > self.keypoint_confidence_threshold and 
                keypoints[11, 2] > self.keypoint_confidence_threshold and keypoints[12, 2] > self.keypoint_confidence_threshold):
                
                # Calculate torso center and height
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                
                # Find midpoint between shoulders and hips
                shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_mid_y = (left_hip[1] + right_hip[1]) / 2
                
                # Upper body is from shoulders to midpoint of torso
                upper_torso_y = (shoulder_mid_y + hip_mid_y) / 2
                
                # Create upper torso area
                upper_torso_points = [
                    [left_shoulder[0], left_shoulder[1]],
                    [right_shoulder[0], right_shoulder[1]],
                    [right_shoulder[0], upper_torso_y],
                    [left_shoulder[0], upper_torso_y]
                ]
                
                # Scale and clip points
                scaled_points = []
                for point in upper_torso_points:
                    x = int(point[0] * scale_x)
                    y = int(point[1] * scale_y)
                    x = np.clip(x, 0, feat_w - 1)
                    y = np.clip(y, 0, feat_h - 1)
                    scaled_points.append([x, y])
                
                if len(scaled_points) >= 3:
                    points = np.array(scaled_points, dtype=np.int32)
                    cv2.fillPoly(upper_body_mask, [points], 1.0)
            
            # Add upper arms (shoulder to elbow)
            upper_body_mask += fill_area_between(5, 7, width=2)  # left upper arm
            upper_body_mask += fill_area_between(6, 8, width=2)  # right upper arm
            
            temp_masks[2] = torch.from_numpy(np.clip(upper_body_mask, 0, 1))
            
            # Part 3: Lower body (lower half of torso + lower arms) (index 3)
            lower_body_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Get torso keypoints again
            if (keypoints[5, 2] > self.keypoint_confidence_threshold and keypoints[6, 2] > self.keypoint_confidence_threshold and 
                keypoints[11, 2] > self.keypoint_confidence_threshold and keypoints[12, 2] > self.keypoint_confidence_threshold):
                
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_hip = keypoints[11]
                right_hip = keypoints[12]
                
                # Find midpoint between shoulders and hips
                shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_mid_y = (left_hip[1] + right_hip[1]) / 2
                
                # Lower body is from midpoint of torso to hips
                upper_torso_y = (shoulder_mid_y + hip_mid_y) / 2
                
                # Create lower torso area
                lower_torso_points = [
                    [left_shoulder[0], upper_torso_y],
                    [right_shoulder[0], upper_torso_y],
                    [right_hip[0], right_hip[1]],
                    [left_hip[0], left_hip[1]]
                ]
                
                # Scale and clip points
                scaled_points = []
                for point in lower_torso_points:
                    x = int(point[0] * scale_x)
                    y = int(point[1] * scale_y)
                    x = np.clip(x, 0, feat_w - 1)
                    y = np.clip(y, 0, feat_h - 1)
                    scaled_points.append([x, y])
                
                if len(scaled_points) >= 3:
                    points = np.array(scaled_points, dtype=np.int32)
                    cv2.fillPoly(lower_body_mask, [points], 1.0)
            
            # Add lower arms (elbow to wrist)
            lower_body_mask += fill_area_between(7, 9, width=2)  # left lower arm
            lower_body_mask += fill_area_between(8, 10, width=2)  # right lower arm
            
            temp_masks[3] = torch.from_numpy(np.clip(lower_body_mask, 0, 1))
            
            # Part 4: Upper legs (upper and lower leg) (index 4)
            upper_legs_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Add thighs (hip to knee) - upper leg
            # Increase width for thighs but keep moderate
            upper_legs_mask += fill_area_between(11, 13, width=2)  # left thigh - increased from 1 to 2
            upper_legs_mask += fill_area_between(12, 14, width=2)  # right thigh - increased from 1 to 2
            
            # Add calves (knee to ankle) - but stop before reaching ankle
            # Create partial calf mask (stop at 75% of the way from knee to ankle)
            for knee_idx, ankle_idx in [(13, 15), (14, 16)]:  # left and right legs
                if (knee_idx < len(keypoints) and ankle_idx < len(keypoints) and 
                    keypoints[knee_idx, 2] > self.keypoint_confidence_threshold and keypoints[ankle_idx, 2] > self.keypoint_confidence_threshold):
                    
                    knee_x = keypoints[knee_idx, 0]
                    knee_y = keypoints[knee_idx, 1]
                    ankle_x = keypoints[ankle_idx, 0]
                    ankle_y = keypoints[ankle_idx, 1]
                    
                    # Calculate point 75% of the way from knee to ankle
                    partial_x = knee_x + 0.75 * (ankle_x - knee_x)
                    partial_y = knee_y + 0.75 * (ankle_y - knee_y)
                    
                    # Create temporary keypoints for partial calf
                    temp_keypoints = keypoints.copy()
                    temp_keypoints[ankle_idx, 0] = partial_x
                    temp_keypoints[ankle_idx, 1] = partial_y
                    
                    # Draw partial calf
                    x1 = int(knee_x * scale_x)
                    y1 = int(knee_y * scale_y)
                    x2 = int(partial_x * scale_x)
                    y2 = int(partial_y * scale_y)
                    
                    x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                    y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                    
                    temp_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
                    # Increase calf line thickness - increased from 1 to 2
                    cv2.line(temp_mask, (x1, y1), (x2, y2), 1.0, 2)
                    # Use moderate dilation kernel - increased from (1,1) to (2,2)
                    kernel = np.ones((2, 2), np.uint8)
                    temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)
                    
                    upper_legs_mask += temp_mask

            # Add knee areas - keep moderate size
            for knee_idx in [13, 14]:  # left and right knees
                if knee_idx < len(keypoints) and keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                    x = int(keypoints[knee_idx, 0] * scale_x)
                    y = int(keypoints[knee_idx, 1] * scale_y)
                    x = np.clip(x, 0, feat_w - 1)
                    y = np.clip(y, 0, feat_h - 1)
                    # Keep knee circle moderate - increased from 1 to 2
                    cv2.circle(upper_legs_mask, (x, y), 2, 1.0, -1)

            temp_masks[4] = torch.from_numpy(np.clip(upper_legs_mask, 0, 1))
            
            # Part 5: Lower legs (foot) (index 5)
            lower_legs_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
            
            # Add foot areas around ankles
            for ankle_idx, knee_idx in [(15, 13), (16, 14)]:  # left and right ankles with corresponding knees
                if ankle_idx < len(keypoints) and keypoints[ankle_idx, 2] > self.keypoint_confidence_threshold:
                    ankle_x = int(keypoints[ankle_idx, 0] * scale_x)
                    ankle_y = int(keypoints[ankle_idx, 1] * scale_y)
                    ankle_x = np.clip(ankle_x, 0, feat_w - 1)
                    ankle_y = np.clip(ankle_y, 0, feat_h - 1)
                    
                    # Keep ankle circles thin - stay at 1 (don't increase)
                    cv2.circle(lower_legs_mask, (ankle_x, ankle_y), 1, 1.0, -1)
                    
                    # Add lower calf area (from 75% of knee-ankle to ankle)
                    if knee_idx < len(keypoints) and keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                        knee_x = keypoints[knee_idx, 0]
                        knee_y = keypoints[knee_idx, 1]
                        ankle_x_orig = keypoints[ankle_idx, 0]
                        ankle_y_orig = keypoints[ankle_idx, 1]
                        
                        # Start from 75% point (where upper leg ends)
                        start_x = knee_x + 0.75 * (ankle_x_orig - knee_x)
                        start_y = knee_y + 0.75 * (ankle_y_orig - knee_y)
                        
                        # Draw lower calf portion
                        x1 = int(start_x * scale_x)
                        y1 = int(start_y * scale_y)
                        x2 = ankle_x
                        y2 = ankle_y
                        
                        x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                        y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                        
                        # Keep lower calf lines thin - stay at 1 (don't increase)
                        cv2.line(lower_legs_mask, (x1, y1), (x2, y2), 1.0, 1)
                    
                    # Extend foot area below ankle (simulate actual foot)
                    # Create an elliptical foot shape below ankle
                    if ankle_y < feat_h - 4:  # Make sure there's room below
                        # Draw ellipse for foot - keep it small
                        center = (ankle_x, min(ankle_y + 2, feat_h - 1))
                        axes = (1, 1)  # width, height of ellipse - keep at (1,1)
                        angle = 0
                        startAngle = 0
                        endAngle = 360
                        cv2.ellipse(lower_legs_mask, center, axes, angle, startAngle, endAngle, 1.0, -1)
                    
                    # Add area extending downward from ankle - keep it narrow
                    for dy in range(1, 3):  # Extend 2 pixels down
                        y_pos = ankle_y + dy
                        if y_pos < feat_h:
                            # Create narrow area as we go down (foot shape)
                            width = 1  # Keep narrow - stay at 1
                            x_start = max(0, ankle_x - width)
                            x_end = min(feat_w - 1, ankle_x + width)
                            cv2.line(lower_legs_mask, (x_start, y_pos), (x_end, y_pos), 1.0, 1)
            
            temp_masks[5] = torch.from_numpy(np.clip(lower_legs_mask, 0, 1))
            
            # Apply morphological operations to smooth masks
            for i in range(1, 6):
                if temp_masks[i].max() > 0:
                    mask_np = temp_masks[i].numpy()
                    # Use symmetric 3x3 kernel for even expansion
                    kernel = np.ones((3, 3), np.uint8)
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)
                    # Use symmetric Gaussian blur for even smoothing
                    mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.5)
                    temp_masks[i] = torch.from_numpy(mask_np)
            
            # PRIORITY-BASED OVERLAP HANDLING
            # Priority order (higher number = higher priority):
            # Head: 5 (highest), Upper body: 4, Lower body: 3, Foot: 2, Upper legs: 1 (lowest)
            part_priorities = {
                1: 5,  # Head - highest priority
                2: 4,  # Upper body - second highest priority
                3: 3,  # Lower body - middle priority
                4: 1,  # Upper legs (thighs and upper calf) - lowest priority
                5: 2,  # Foot (lower calf + ankle + foot area) - second lowest priority
            }
            
            # Create final masks with priority-based assignment
            final_masks = torch.zeros(1, 6, feat_h, feat_w)
            
            # Create assignment map
            assignment_map = torch.zeros(feat_h, feat_w, dtype=torch.long)
            priority_map = torch.zeros(feat_h, feat_w)
            
            # Threshold for considering a pixel as part of a mask
            activation_threshold = 0.2
            
            # Assign each pixel to the highest priority part
            for y in range(feat_h):
                for x in range(feat_w):
                    max_priority = 0
                    assigned_part = 0
                    
                    for part_idx in range(1, 6):
                        if temp_masks[part_idx, y, x] > activation_threshold:
                            part_priority = part_priorities[part_idx]
                            if part_priority > max_priority:
                                max_priority = part_priority
                                assigned_part = part_idx
                    
                    assignment_map[y, x] = assigned_part
                    priority_map[y, x] = max_priority
            
            # Create hard masks based on assignment
            for part_idx in range(1, 6):
                final_masks[0, part_idx] = (assignment_map == part_idx).float()
            
            # Apply smoothing to reduce harsh boundaries
            smooth_kernel = np.array([[1, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 1]], dtype=np.float32) / 10.0
            
            for i in range(1, 6):
                if final_masks[0, i].max() > 0:
                    mask_np = final_masks[0, i].numpy()
                    # Apply symmetric smoothing kernel
                    mask_np = cv2.filter2D(mask_np, -1, smooth_kernel)
                    # Use symmetric Gaussian blur for final smoothing
                    mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.3)
                    final_masks[0, i] = torch.from_numpy(mask_np)
            
            # Create background mask (pixels not assigned to any part)
            final_masks[0, 0] = (assignment_map == 0).float()
            
            # Ensure each pixel sums to 1 (normalization)
            mask_sum = final_masks.sum(dim=1, keepdim=True)
            mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
            final_masks = final_masks / mask_sum
            
            return final_masks.to(self.device)
            
        except Exception as e:
            print(f"YOLO Pose skeleton mask generation failed: {e}")
            return None

    def extract_features_corrected(self, image, return_masks=False):
        """Extract features with corrected processing (adapted from corrected version)"""
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate YOLO Pose masks
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            masks = self.generate_yolo_pose_masks(img_bgr)
            
            # Extract features
            with torch.no_grad():
                # Forward pass with masks
                outputs = self.model(image_tensor, external_parts_masks=masks)
                
                # Process outputs correctly (from corrected version)
                features = self._process_corrected_output(outputs)
                
                if features is not None and features.numel() > 0:
                    # Apply corrected normalization
                    features = self._apply_corrected_normalization(features)
                    
                    if return_masks:
                        return features, masks
                    else:
                        return features
                else:
                    print("Warning: Empty features extracted")
                    if return_masks:
                        return torch.randn(1, 512).to(self.device), masks
                    else:
                        return torch.randn(1, 512).to(self.device)
                    
        except Exception as e:
            print(f"Feature extraction error: {e}")
            if return_masks:
                return torch.randn(1, 512).to(self.device), None
            else:
                return torch.randn(1, 512).to(self.device)
    
    def _process_corrected_output(self, outputs):
        """Process model output correctly (from corrected version)"""
        try:
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                embeddings_dict = outputs[0]
                
                if isinstance(embeddings_dict, dict):
                    # Priority order for feature selection (from corrected version)
                    priority_keys = ['bn_foreg', 'foreground', 'bn_global', 'global', 'parts']
                    
                    for key in priority_keys:
                        if key in embeddings_dict:
                            features = embeddings_dict[key]
                            if isinstance(features, torch.Tensor) and features.numel() > 0:
                                if len(features.shape) > 2:
                                    features = features.view(features.size(0), -1)
                                return features
                    
                    # If no priority keys, use first available tensor
                    for key, value in embeddings_dict.items():
                        if isinstance(value, torch.Tensor) and value.numel() > 0:
                            if len(value.shape) > 2:
                                value = value.view(value.size(0), -1)
                            return value
                
                elif isinstance(embeddings_dict, torch.Tensor):
                    if len(embeddings_dict.shape) > 2:
                        return embeddings_dict.view(embeddings_dict.size(0), -1)
                    return embeddings_dict
            
            return None
            
        except Exception as e:
            print(f"Error processing output: {e}")
            return None
    
    def _apply_corrected_normalization(self, features):
        """Apply corrected feature normalization (from corrected version)"""
        # L2 normalization
        features = F.normalize(features, p=2, dim=1)
        
        # Additional stabilization - subtract mean to center features
        features = features - features.mean(dim=1, keepdim=True)
        
        # Re-normalize after centering
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def compute_corrected_similarity(self, query_features, gallery_features):
        """Compute corrected similarity with multiple metrics (from corrected version)"""
        # Ensure proper dimensions
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        if gallery_features.dim() == 1:
            gallery_features = gallery_features.unsqueeze(0)
        
        # Cosine similarity
        cosine_sim = torch.mm(query_features, gallery_features.t())
        
        # Euclidean distance (converted to similarity)
        query_expanded = query_features.unsqueeze(1)
        gallery_expanded = gallery_features.unsqueeze(0)
        euclidean_dist = torch.norm(query_expanded - gallery_expanded, p=2, dim=2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine similarities (weighted average from corrected version)
        combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        return combined_sim
    
    def load_gallery_person_corrected(self, gallery_path: str):
        """Load gallery person with corrected processing (from corrected version)"""
        print(f"Loading gallery person with corrected processing: {gallery_path}")
        
        if not os.path.exists(gallery_path):
            raise FileNotFoundError(f"Gallery image not found: {gallery_path}")
        
        # Load gallery image
        image = cv2.imread(gallery_path)
        if image is None:
            raise ValueError(f"Could not load gallery image: {gallery_path}")
        
        # Detect person in image using YOLO (important: same as corrected version)
        results = self.yolo(image, classes=0, conf=self.person_detection_threshold)
        
        person_detected = False
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = image[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        # Extract features from cropped person
                        features = self.extract_features_corrected(person_img)
                        
                        # Add to gallery
                        self.gallery_features.append(features)
                        self.gallery_ids.append(self.next_person_id)
                        self.gallery_images.append(person_img.copy())
                        
                        print(f"Added Person {self.next_person_id} to gallery from {gallery_path}")
                        print(f"Feature shape: {features.shape}")
                        self.next_person_id += 1
                        person_detected = True
                        break
            
            if person_detected:
                break
        
        if not person_detected:
            raise ValueError(f"No person detected in gallery image: {gallery_path}")
        
        return True
    
    def match_person_corrected(self, query_features, part_weights=None):
        """Match person with corrected similarity computation and adaptive thresholding"""
        if len(self.gallery_features) == 0:
            return None, 0.0, self.reid_threshold
        
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        
        # Use corrected similarity computation
        similarities = self.compute_corrected_similarity(query_features, gallery_tensor)
        
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        # Apply adaptive thresholding based on visible body parts
        if part_weights is not None:
            # Count parts with significant coverage (>5%)
            visible_parts = sum(1 for w in part_weights if w > 5.0)
            # Adjust threshold: fewer parts = lower threshold
            adaptive_threshold = self.reid_threshold * (visible_parts / 5.0) ** 0.5
            # Ensure threshold doesn't go below 0.2 or above original threshold
            adaptive_threshold = max(0.15, min(adaptive_threshold, self.reid_threshold))
        else:
            adaptive_threshold = self.reid_threshold
        
        if best_similarity > adaptive_threshold:
            return self.gallery_ids[best_idx], best_similarity, adaptive_threshold
        else:
            return None, best_similarity, adaptive_threshold
    
    def visualize_keypoints(self, frame: np.ndarray, person_img: np.ndarray, bbox: Tuple[int, int, int, int, float]) -> np.ndarray:
        """Visualize keypoints on the person image and overlay on frame"""
        try:
            
            # Run YOLO pose estimation
            results = self.yolo(person_img, task='pose')
            
            # Check if results exist and have keypoints
            if len(results) == 0:
                return frame
            
            if not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
                return frame
            
            if len(results[0].keypoints.data) == 0:
                return frame
            
            # Use the first pose detection
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: [17, 3] - (x, y, confidence)
            
            # Validate keypoints shape
            if keypoints.shape[0] != 17:
                return frame
            
            # COCO keypoint names for reference
            keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            # Define colors for different keypoint groups (BGR format)
            head_colors = [(0, 0, 255), (50, 50, 255), (100, 100, 255), (150, 150, 255), (200, 200, 255)]  # Blue shades
            torso_colors = [(0, 255, 0), (50, 255, 50), (100, 255, 100)]  # Green shades
            arm_colors = [(255, 0, 0), (255, 50, 50), (255, 100, 100), (255, 150, 150)]  # Red shades
            leg_colors = [(0, 255, 255), (50, 255, 255), (100, 255, 255), (150, 255, 255)]  # Yellow shades
            
            x1, y1, x2, y2, conf = bbox
            
            # Draw keypoints with labels
            for i, (name, keypoint) in enumerate(zip(keypoint_names, keypoints)):
                if keypoint[2] > self.keypoint_confidence_threshold:  # Only draw if confidence > 0.3
                    # Convert from person image coordinates to frame coordinates
                    kp_x = int(keypoint[0]) + x1
                    kp_y = int(keypoint[1]) + y1
                    
                    # Choose color based on keypoint type
                    if i < 5:  # Head keypoints
                        color = head_colors[i]
                    elif i < 7:  # Shoulders
                        color = torso_colors[i-5]
                    elif i < 11:  # Arms
                        color = arm_colors[i-7]
                    elif i < 13:  # Hips
                        color = torso_colors[i-11]
                    else:  # Legs
                        color = leg_colors[i-13]
                    
                    # Draw keypoint circle
                    cv2.circle(frame, (kp_x, kp_y), 3, color, -1)
                    cv2.circle(frame, (kp_x, kp_y), 5, (255, 255, 255), 1)  # White border
                    
                    # Draw keypoint number (smaller text for video)
                    label = f"{i}"
                    cv2.putText(frame, label, (kp_x + 6, kp_y - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw skeleton connections
            skeleton_connections = [
                # Head connections
                (0, 1), (0, 2), (1, 3), (2, 4),  # nose to eyes to ears
                # Torso connections
                (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders to hips
                # Arms
                (5, 7), (7, 9),  # left arm
                (6, 8), (8, 10),  # right arm
                # Legs
                (11, 13), (13, 15),  # left leg
                (12, 14), (14, 16),  # right leg
            ]
            
            # Draw skeleton lines
            for connection in skeleton_connections:
                kp1_idx, kp2_idx = connection
                if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints) and 
                    keypoints[kp1_idx, 2] > self.keypoint_confidence_threshold and keypoints[kp2_idx, 2] > self.keypoint_confidence_threshold):
                    
                    # Convert from person image coordinates to frame coordinates
                    x1_kp = int(keypoints[kp1_idx, 0]) + x1
                    y1_kp = int(keypoints[kp1_idx, 1]) + y1
                    x2_kp = int(keypoints[kp2_idx, 0]) + x1
                    y2_kp = int(keypoints[kp2_idx, 1]) + y1
                    
                    # Choose line color based on connection type
                    if kp1_idx < 5 or kp2_idx < 5:  # Head connections
                        line_color = (255, 100, 100)  # Light red
                    elif kp1_idx < 11 or kp2_idx < 11:  # Upper body connections
                        line_color = (100, 255, 100)  # Light green
                    else:  # Leg connections
                        line_color = (100, 100, 255)  # Light blue
                    
                    cv2.line(frame, (x1_kp, y1_kp), (x2_kp, y2_kp), line_color, 1)

            
            return frame
            
        except Exception as e:
            print(f"YOLO Pose keypoint visualization error: {e}")
            return frame

    def visualize_reid_result(self, frame: np.ndarray, bbox: Tuple[int, int, int, int, float], 
                             similarity: float, is_match: bool, masks: Optional[torch.Tensor] = None,
                             person_img: Optional[np.ndarray] = None, adaptive_threshold: float = None) -> np.ndarray:
        """Visualize re-identification results with mask overlay and body part weights"""
        vis_frame = frame.copy()
        
        x1, y1, x2, y2, conf = bbox
        
        # Calculate body part weights if masks are available
        part_weights = []
        if masks is not None and person_img is not None:
            try:
                # Convert masks to numpy and remove batch dimension
                if isinstance(masks, torch.Tensor):
                    masks_np = masks.cpu().numpy().squeeze(0)  # Remove batch dimension
                else:
                    masks_np = masks
                
                # Define colors for each body part (BGR format)
                part_colors = [
                    (255, 0, 0),      # Head - Blue
                    (0, 255, 0),      # Upper Body - Green  
                    (0, 0, 255),      # Lower Body - Red
                    (255, 255, 0),    # Upper Legs (upper and lower leg) - Cyan
                    (255, 0, 255),    # Lower Legs (foot) - Magenta
                ]
                
                # Create colored mask overlay
                mask_overlay = np.zeros_like(frame, dtype=np.uint8)
                
                # Get the person region dimensions
                person_h = y2 - y1
                person_w = x2 - x1
                
                # Process each part mask (skip background at index 0)
                for i in range(1, min(6, masks_np.shape[0])):
                    if i - 1 < len(part_colors):
                        # Get the mask for this part
                        part_mask = masks_np[i]
                        
                        # Resize mask to match person region
                        if part_mask.shape != (person_h, person_w):
                            part_mask_resized = cv2.resize(part_mask, (person_w, person_h))
                        else:
                            part_mask_resized = part_mask
                        
                        # Apply threshold to create binary mask
                        binary_mask = (part_mask_resized > 0.2).astype(np.uint8)
                        
                        # Calculate weight (percentage of pixels covered by this part)
                        total_pixels = person_h * person_w
                        covered_pixels = np.sum(binary_mask)
                        weight = (covered_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                        part_weights.append(weight)
                        
                        # Apply color to mask overlay
                        mask_region = np.zeros((person_h, person_w, 3), dtype=np.uint8)
                        mask_region[binary_mask > 0] = part_colors[i - 1]
                        
                        # Add to full frame mask overlay
                        mask_overlay[y1:y2, x1:x2] = np.maximum(mask_overlay[y1:y2, x1:x2], mask_region)
                
                # Blend mask overlay with frame
                alpha = 0.4  # Transparency for mask overlay
                vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, mask_overlay, alpha, 0)
                
                # Add mask type indicator
                cv2.putText(vis_frame, "Mask: YOLO Pose", (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add keypoint visualization
                if person_img is not None:
                    vis_frame = self.visualize_keypoints(vis_frame, person_img, bbox)
                
            except Exception as e:
                print(f"Mask visualization error: {e}")
        
        # Add keypoint visualization even if masks are not available
        if person_img is not None and masks is None:
            vis_frame = self.visualize_keypoints(vis_frame, person_img, bbox)
        
        # Draw bounding box with color based on match
        box_color = (0, 255, 0) if is_match else (0, 0, 255)  # Green if match, Red if no match
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add similarity score with background for better visibility
        similarity_text = f"Sim: {similarity:.3f}"
        text_size = cv2.getTextSize(similarity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(vis_frame, (x1 - 2, y1 - 10 - text_size[1] - 2),
                     (x1 + text_size[0] + 2, y1 - 8), (0, 0, 0), -1)
        cv2.putText(vis_frame, similarity_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add adaptive threshold info if available
        if adaptive_threshold is not None:
            threshold_text = f"Adaptive: {adaptive_threshold:.3f}"
            cv2.putText(vis_frame, threshold_text, (x2, y1 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Yellow color
        
        # Add match status
        status_text = "MATCH" if is_match else "NO MATCH"
        status_color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.putText(vis_frame, status_text, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add detection confidence
        conf_text = f"Det: {conf:.2f}"
        cv2.putText(vis_frame, conf_text, (x2 - 80, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display body part weights on the right of the detection box
        if part_weights:
            part_names = ["Head", "Upper Body", "Lower Body", "Upper Legs", "Foot"]
            start_y = y1
            line_height = 20
            
            # Draw background for weight display
            weight_bg_width = 120
            weight_bg_height = len(part_weights) * line_height + 10
            cv2.rectangle(vis_frame, (x2 + 5, start_y - 5), 
                         (x2 + weight_bg_width, start_y + weight_bg_height), 
                         (0, 0, 0), -1)
            cv2.rectangle(vis_frame, (x2 + 5, start_y - 5), 
                         (x2 + weight_bg_width, start_y + weight_bg_height), 
                         (255, 255, 255), 1)
            
            # Add title
            cv2.putText(vis_frame, "Body Parts:", (x2 + 10, start_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display each body part weight
            for i, (name, weight) in enumerate(zip(part_names, part_weights)):
                y_pos = start_y + 30 + i * line_height
                weight_text = f"{name}: {weight:.1f}%"
                cv2.putText(vis_frame, weight_text, (x2 + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_frame
    
    def process_video(self, video_path: str, gallery_path: str, output_path: str = None, 
                     show_preview: bool = True, save_video: bool = True):
        """Process video with improved BPBreID re-identification"""
        
        # Clear gallery and load new gallery person
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Load gallery person with corrected processing
        self.load_gallery_person_corrected(gallery_path)
        
        # Update test results
        self.test_results['video_path'] = video_path
        self.test_results['gallery_path'] = gallery_path
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.test_results['total_frames'] = total_frames
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if saving
        if save_video:
            if output_path is None:
                output_path = video_path.replace('.MOV', '_improved_reid.mp4').replace('.mp4', '_improved_reid.mp4')
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect persons using YOLO
                results = self.yolo(frame, classes=0, conf=self.person_detection_threshold)
                
                # Process each detected person
                vis_frame = frame.copy()
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            bbox = (x1, y1, x2, y2, conf)
                            
                            # Extract person image (important: crop like in corrected version)
                            person_img = frame[y1:y2, x1:x2]
                            
                            if person_img.size > 0:
                                try:
                                    # Extract features from cropped person with masks
                                    query_features, query_masks = self.extract_features_corrected(person_img, return_masks=True)
                                    
                                    # Calculate part weights for adaptive thresholding
                                    part_weights = []
                                    if query_masks is not None:
                                        try:
                                            masks_np = query_masks.cpu().numpy().squeeze(0)
                                            person_h = y2 - y1
                                            person_w = x2 - x1
                                            
                                            for i in range(1, min(6, masks_np.shape[0])):
                                                part_mask = masks_np[i]
                                                if part_mask.shape != (person_h, person_w):
                                                    part_mask_resized = cv2.resize(part_mask, (person_w, person_h))
                                                else:
                                                    part_mask_resized = part_mask
                                                
                                                binary_mask = (part_mask_resized > 0.2).astype(np.uint8)
                                                total_pixels = person_h * person_w
                                                covered_pixels = np.sum(binary_mask)
                                                weight = (covered_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                                                part_weights.append(weight)
                                        except Exception as e:
                                            print(f"Error calculating part weights: {e}")
                                            part_weights = []
                                    
                                    # Match with gallery using adaptive thresholding
                                    matched_id, similarity, adaptive_threshold = self.match_person_corrected(query_features, part_weights)
                                    
                                    # Determine if it's a match
                                    is_match = matched_id is not None
                                    
                                    # Update statistics
                                    self.test_results['frames_with_detection'] += 1
                                    if is_match:
                                        self.test_results['frames_with_match'] += 1
                                    else:
                                        self.test_results['frames_without_match'] += 1
                                    
                                    self.test_results['similarity_scores'].append(similarity)
                                    
                                    # Visualize result with masks and adaptive threshold
                                    vis_frame = self.visualize_reid_result(
                                        vis_frame, bbox, similarity, is_match, 
                                        masks=query_masks, person_img=person_img, 
                                        adaptive_threshold=adaptive_threshold
                                    )
                                    
                                except Exception as e:
                                    print(f"Processing error for frame {frame_count}: {e}")
                                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(vis_frame, "ERROR", (x1, y1-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Gallery: Person {self.gallery_ids[0] if self.gallery_ids else 'None'}"
                cv2.putText(vis_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                        # Add threshold info
                threshold_text = f"Base Threshold: {self.reid_threshold} | Masking: YOLO Pose"
                cv2.putText(vis_frame, threshold_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame if requested
                if save_video:
                    out.write(vis_frame)
                
                # Show preview if requested
                if show_preview:
                    cv2.imshow('Improved BPBreID ReID', vis_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | FPS: {fps_actual:.1f} | Frame: {frame_count}/{total_frames}")
                
                # Handle keyboard input
                if show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if save_video:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            # Calculate final statistics
            if self.test_results['frames_with_detection'] > 0:
                match_rate = self.test_results['frames_with_match'] / self.test_results['frames_with_detection']
                avg_similarity = np.mean(self.test_results['similarity_scores']) if self.test_results['similarity_scores'] else 0
                
                print(f"\nImproved ReID Results:")
                print(f"  Total frames: {self.test_results['total_frames']}")
                print(f"  Frames with detection: {self.test_results['frames_with_detection']}")
                print(f"  Frames with match: {self.test_results['frames_with_match']}")
                print(f"  Match rate: {match_rate:.3f}")
                print(f"  Average similarity: {avg_similarity:.3f}")
                print(f"  ReID threshold: {self.reid_threshold}")
                
                if self.test_results['similarity_scores']:
                    min_sim = np.min(self.test_results['similarity_scores'])
                    max_sim = np.max(self.test_results['similarity_scores'])
                    print(f"  Similarity range: [{min_sim:.3f}, {max_sim:.3f}]")
            
            if save_video:
                print(f"  Output saved to: {output_path}")
            
            # Save test results
            results_path = output_path.replace('.mp4', '_results.json') if output_path else 'reid_results.json'
            with open(results_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"  Results saved to: {results_path}")
    
    def run_comparison_test(self, gallery_path: str, video1_path: str, video2_path: str, 
                           output_dir: str = "improved_reid_results"):
        """Run comparison test on two videos (similar to corrected version)"""
        print("="*80)
        print("IMPROVED BPBREID COMPARISON TEST")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Test video 1 (should match)
        print("\n1. Testing Video 1 (Same Person - Should Match)...")
        video1_output = os.path.join(output_dir, "video1_improved_reid.mp4")
        
        try:
            self.process_video(
                video_path=video1_path,
                gallery_path=gallery_path,
                output_path=video1_output,
                show_preview=False,
                save_video=True
            )
            results1 = self.test_results.copy()
        except Exception as e:
            print(f"Error testing video 1: {e}")
            return None
        
        # Reset test results
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'video_path': '',
            'gallery_path': '',
            'total_frames': 0,
            'frames_with_detection': 0,
            'frames_with_match': 0,
            'frames_without_match': 0,
            'similarity_scores': [],
            'reid_threshold': self.reid_threshold
        }
        
        # Test video 2 (should not match)
        print("\n2. Testing Video 2 (Different Person - Should NOT Match)...")
        video2_output = os.path.join(output_dir, "video2_improved_reid.mp4")
        
        try:
            self.process_video(
                video_path=video2_path,
                gallery_path=gallery_path,
                output_path=video2_output,
                show_preview=False,
                save_video=True
            )
            results2 = self.test_results.copy()
        except Exception as e:
            print(f"Error testing video 2: {e}")
            return None
        
        # Compile and display results
        print("\n" + "="*80)
        print("COMPARISON TEST RESULTS")
        print("="*80)
        
        # Video 1 analysis
        if results1['frames_with_detection'] > 0:
            v1_match_rate = results1['frames_with_match'] / results1['frames_with_detection']
            v1_avg_sim = np.mean(results1['similarity_scores']) if results1['similarity_scores'] else 0
            
            print(f"\nVideo 1 (Same Person):")
            print(f"  Match rate: {v1_match_rate:.3f} (expected: high)")
            print(f"  Average similarity: {v1_avg_sim:.3f}")
            if results1['similarity_scores']:
                print(f"  Similarity range: [{np.min(results1['similarity_scores']):.3f}, {np.max(results1['similarity_scores']):.3f}]")
        
        # Video 2 analysis
        if results2['frames_with_detection'] > 0:
            v2_match_rate = results2['frames_with_match'] / results2['frames_with_detection']
            v2_avg_sim = np.mean(results2['similarity_scores']) if results2['similarity_scores'] else 0
            
            print(f"\nVideo 2 (Different Person):")
            print(f"  Match rate: {v2_match_rate:.3f} (expected: low)")
            print(f"  Average similarity: {v2_avg_sim:.3f}")
            if results2['similarity_scores']:
                print(f"  Similarity range: [{np.min(results2['similarity_scores']):.3f}, {np.max(results2['similarity_scores']):.3f}]")
        
        # Separation analysis
        if results1['similarity_scores'] and results2['similarity_scores']:
            separation = v1_avg_sim - v2_avg_sim
            print(f"\nSimilarity Separation: {separation:.3f}")
            if separation > 0.1:
                print("✅ Good feature discrimination achieved!")
            else:
                print("⚠️ Poor feature discrimination - consider adjusting threshold")
        
        print("="*80)
        
        # Save combined results
        combined_results = {
            'timestamp': datetime.now().isoformat(),
            'gallery_path': gallery_path,
            'reid_threshold': self.reid_threshold,
            'masking_method': 'YOLO Pose',
            'video1_results': results1,
            'video2_results': results2
        }
        
        results_file = os.path.join(output_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Annotated videos saved in: {output_dir}")
        
        return combined_results


def main():
    """Main function to run improved BPBreID re-identification"""
    
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bpbreid_dir = os.path.dirname(current_dir)
    
    # Configuration
    REID_MODEL_PATH = os.path.join(bpbreid_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
    HRNET_PATH = os.path.join(bpbreid_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")
    YOLO_MODEL = "yolov8n-pose.pt"
    
    # Test dataset paths - you can switch between datasets
    dataset = "dataset-1"  # Change to "dataset-1" to test the other dataset
    
    if dataset == "dataset-2":
        GALLERY_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-2.jpg")
        VIDEO1_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-2-vid.MOV")
        VIDEO2_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1-vid.MOV")
    else:
        GALLERY_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "gallery-person.jpg")
        VIDEO1_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "correct.MOV")
        VIDEO2_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "incorrect.MOV")
    
    # Check if files exist
    missing_files = []
    for path, name in [(REID_MODEL_PATH, "BPBreID model"), (HRNET_PATH, "HRNet model"),
                       (GALLERY_PATH, "Gallery image"), (VIDEO1_PATH, "Video 1"), 
                       (VIDEO2_PATH, "Video 2")]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    try:
        print("="*80)
        print("IMPROVED BPBREID YOLO MASKED REID SYSTEM")
        print("="*80)
        print("Key improvements from corrected version:")
        print("- Gallery image processing with YOLO detection and cropping")
        print("- Corrected feature extraction with proper normalization")
        print("- Combined similarity metrics (70% cosine + 30% Euclidean)")
        print("- Better feature key prioritization")
        print("- YOLO Pose-based masking with 5 body sections:")
        print("  1. Head")
        print("  2. Upper body (upper half of torso + upper arms)")
        print("  3. Lower body (lower half of torso + lower arms)")
        print("  4. Upper legs (thighs and upper calf - stops at 75% to ankle)")
        print("  5. Foot (lower calf from 75% + ankle + foot area)")
        print("="*80)
        print()
        
        # Create improved BPBreID re-identification system
        print("Initializing Improved BPBreID YOLO Masked ReID system...")
        
        reid_system = ImprovedBPBreIDYOLOMaskedReID(
            reid_model_path=REID_MODEL_PATH,
            hrnet_path=HRNET_PATH,
            yolo_model_path=YOLO_MODEL
        )
        
        # Run comparison test
        reid_system.run_comparison_test(
            gallery_path=GALLERY_PATH,
            video1_path=VIDEO1_PATH,
            video2_path=VIDEO2_PATH,
            output_dir=f"improved_reid_results_{dataset}"
        )
        
        print("\n🎉 Improved BPBreID test completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()