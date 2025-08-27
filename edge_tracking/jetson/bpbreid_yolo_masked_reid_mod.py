#!/usr/bin/env python3
"""
Improved BPBreID Re-Identification with YOLO Detection and Pose-Based Masking

This version incorporates the successful techniques from corrected_masked_reid_test_default.py:
1. Proper gallery image processing with YOLO detection and cropping
2. Corrected feature extraction and normalization
3. Combined similarity metrics
4. Better feature key selection
5. Consistent processing for both gallery and query images
6. YOLO Pose-based masking with 5 body sections
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
             keypoint_confidence_threshold=0.5, person_detection_threshold=0.6):
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
            # Build model with original configuration
            model = torchreid.models.build_model(
                name='bpbreid',
                num_classes=751,
                config=self.config,
                pretrained=True
            )
            
            # Load weights
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
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            model = model.to(self.device)
            model.eval()
            
            print("Corrected BPBreid model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading corrected BPBreid model: {e}")
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