"""
YOLO Pose-based mask generation utility for BPBreID

This module provides shared functionality for generating pose-based masks
using YOLO Pose skeleton structure with 5 body sections:
1. Head
2. Upper body (upper half of torso + upper arms)
3. Lower body (lower half of torso + lower arms)
4. Upper legs (thighs and upper calf - stops at 75% to ankle)
5. Foot (lower calf from 75% + ankle + foot area)
"""

from __future__ import division, print_function, absolute_import
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union


__all__ = ['YOLOPoseMaskGenerator', 'generate_yolo_pose_masks']


class YOLOPoseMaskGenerator:
    """
    YOLO Pose-based mask generator for BPBreID
    """
    
    def __init__(self, yolo_model, keypoint_confidence_threshold=0.5, 
                 height=384, width=128, device=None):
        """
        Initialize YOLO Pose mask generator
        
        Args:
            yolo_model: YOLO model instance for pose estimation
            keypoint_confidence_threshold: Confidence threshold for keypoints
            height: Target height for feature maps (default: 384)
            width: Target width for feature maps (default: 128)
            device: Target device for GPU acceleration (default: None for CPU)
        """
        self.yolo = yolo_model
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self.height = height
        self.width = width
        self.device = device if device is not None else torch.device('cpu')
        
    def generate_masks(self, person_img: Union[np.ndarray, torch.Tensor], 
                      device: Optional[torch.device] = None) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """
        Generate pose-based masks using YOLO Pose skeleton structure
        
        Args:
            person_img: Input person image (BGR format for np.ndarray)
            device: Target device for output tensor (if None, returns numpy array)
            
        Returns:
            Generated masks with shape (1, 6, feat_h, feat_w) or None if failed
        """
        try:
            # Ensure person_img is a numpy array, not a tensor
            if isinstance(person_img, torch.Tensor):
                person_img = person_img.cpu().numpy()
            
            # Run YOLO pose estimation directly on BGR image (same as default implementation)
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
            feat_h, feat_w = self.height // 8, self.width // 8
            
            # Initialize temporary masks for each part (before priority assignment)
            temp_masks = torch.zeros(6, feat_h, feat_w, device=self.device)  # 5 parts + will add background later
            
            # Scale factors for keypoint coordinates
            scale_x = feat_w / w
            scale_y = feat_h / h
            
            # GPU-accelerated helper functions
            def draw_skeleton_line_gpu(kp1_idx, kp2_idx, thickness=1):
                """Draw a thick line between two keypoints on GPU tensor"""
                if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints) and 
                    keypoints[kp1_idx, 2] > self.keypoint_confidence_threshold and keypoints[kp2_idx, 2] > self.keypoint_confidence_threshold):
                    
                    x1 = int(keypoints[kp1_idx, 0] * scale_x)
                    y1 = int(keypoints[kp1_idx, 1] * scale_y)
                    x2 = int(keypoints[kp2_idx, 0] * scale_x)
                    y2 = int(keypoints[kp2_idx, 1] * scale_y)
                    
                    # Clip coordinates
                    x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                    y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                    
                    # Create GPU tensor for line drawing
                    temp_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
                    temp_mask = self._draw_line_gpu(temp_mask, x1, y1, x2, y2, thickness)
                    
                    return temp_mask
                return torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
            
            def fill_area_between_gpu(kp1_idx, kp2_idx, width=1):
                """Fill the area around a line between two keypoints using GPU"""
                if (kp1_idx < len(keypoints) and kp2_idx < len(keypoints) and 
                    keypoints[kp1_idx, 2] > self.keypoint_confidence_threshold and keypoints[kp2_idx, 2] > self.keypoint_confidence_threshold):
                    
                    x1 = int(keypoints[kp1_idx, 0] * scale_x)
                    y1 = int(keypoints[kp1_idx, 1] * scale_y)
                    x2 = int(keypoints[kp2_idx, 0] * scale_x)
                    y2 = int(keypoints[kp2_idx, 1] * scale_y)
                    
                    # Clip coordinates
                    x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                    y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                    
                    temp_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
                    temp_mask = self._draw_line_gpu(temp_mask, x1, y1, x2, y2, width)
                    
                    # Apply GPU-based dilation
                    kernel_size = max(1, min(width, 2))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    temp_mask = self._dilate_gpu(temp_mask, kernel_size)
                    
                    return temp_mask
                return torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
            
            # Part 1: Head (index 1)
            try:
                head_mask = self._generate_head_mask_gpu(keypoints, feat_h, feat_w, scale_x, scale_y, draw_skeleton_line_gpu)
                temp_masks[1] = torch.clamp(head_mask, 0, 1)
            except Exception as e:
                print(f"Error generating head mask: {e}")
                temp_masks[1] = torch.zeros((feat_h, feat_w), device=self.device)
            
            # Part 2: Upper body (index 2)
            try:
                upper_body_mask = self._generate_upper_body_mask_gpu(keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between_gpu)
                temp_masks[2] = torch.clamp(upper_body_mask, 0, 1)
            except Exception as e:
                print(f"Error generating upper body mask: {e}")
                temp_masks[2] = torch.zeros((feat_h, feat_w), device=self.device)
            
            # Part 3: Lower body (index 3)
            try:
                lower_body_mask = self._generate_lower_body_mask_gpu(keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between_gpu)
                temp_masks[3] = torch.clamp(lower_body_mask, 0, 1)
            except Exception as e:
                print(f"Error generating lower body mask: {e}")
                temp_masks[3] = torch.zeros((feat_h, feat_w), device=self.device)
            
            # Part 4: Upper legs (index 4)
            try:
                upper_legs_mask = self._generate_upper_legs_mask_gpu(keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between_gpu)
                temp_masks[4] = torch.clamp(upper_legs_mask, 0, 1)
            except Exception as e:
                print(f"Error generating upper legs mask: {e}")
                temp_masks[4] = torch.zeros((feat_h, feat_w), device=self.device)
            
            # Part 5: Lower legs (foot) (index 5)
            try:
                lower_legs_mask = self._generate_lower_legs_mask_gpu(keypoints, feat_h, feat_w, scale_x, scale_y)
                temp_masks[5] = torch.clamp(lower_legs_mask, 0, 1)
            except Exception as e:
                print(f"Error generating lower legs mask: {e}")
                temp_masks[5] = torch.zeros((feat_h, feat_w), device=self.device)
            
            # Apply morphological operations to smooth masks
            temp_masks = self._apply_morphological_smoothing(temp_masks)
            
            # Apply priority-based overlap handling
            final_masks, assignment_map = self._apply_priority_assignment(temp_masks, feat_h, feat_w)
            
            # Apply final smoothing
            final_masks = self._apply_final_smoothing(final_masks)
            
            # Create background mask and normalize
            final_masks = self._finalize_masks(final_masks, assignment_map)
            
            # Return in appropriate format
            if device is not None:
                return final_masks.to(device)
            else:
                return final_masks.cpu().numpy()
                
        except Exception as e:
            print(f"YOLO Pose skeleton mask generation failed: {e}")
            return None
    
    def _generate_head_mask(self, keypoints, feat_h, feat_w, scale_x, scale_y, draw_skeleton_line):
        """Generate head mask"""
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
        
        return head_mask
    
    def _generate_upper_body_mask(self, keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between):
        """Generate upper body mask"""
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
        
        return upper_body_mask
    
    def _generate_lower_body_mask(self, keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between):
        """Generate lower body mask"""
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
        
        return lower_body_mask
    
    def _generate_upper_legs_mask(self, keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between):
        """Generate upper legs mask"""
        upper_legs_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
        
        # Add thighs (hip to knee) - upper leg
        upper_legs_mask += fill_area_between(11, 13, width=2)  # left thigh
        upper_legs_mask += fill_area_between(12, 14, width=2)  # right thigh
        
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
                
                # Draw partial calf
                x1 = int(knee_x * scale_x)
                y1 = int(knee_y * scale_y)
                x2 = int(partial_x * scale_x)
                y2 = int(partial_y * scale_y)
                
                x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                
                temp_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
                cv2.line(temp_mask, (x1, y1), (x2, y2), 1.0, 2)
                kernel = np.ones((2, 2), np.uint8)
                temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)
                
                upper_legs_mask += temp_mask

        # Add knee areas
        for knee_idx in [13, 14]:  # left and right knees
            if knee_idx < len(keypoints) and keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                x = int(keypoints[knee_idx, 0] * scale_x)
                y = int(keypoints[knee_idx, 1] * scale_y)
                x = np.clip(x, 0, feat_w - 1)
                y = np.clip(y, 0, feat_h - 1)
                cv2.circle(upper_legs_mask, (x, y), 2, 1.0, -1)

        return upper_legs_mask
    
    def _generate_lower_legs_mask(self, keypoints, feat_h, feat_w, scale_x, scale_y):
        """Generate lower legs (foot) mask"""
        lower_legs_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
        
        # Add foot areas around ankles
        for ankle_idx, knee_idx in [(15, 13), (16, 14)]:  # left and right ankles with corresponding knees
            if ankle_idx < len(keypoints) and keypoints[ankle_idx, 2] > self.keypoint_confidence_threshold:
                ankle_x = int(keypoints[ankle_idx, 0] * scale_x)
                ankle_y = int(keypoints[ankle_idx, 1] * scale_y)
                ankle_x = np.clip(ankle_x, 0, feat_w - 1)
                ankle_y = np.clip(ankle_y, 0, feat_h - 1)
                
                # Keep ankle circles thin
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
                    
                    cv2.line(lower_legs_mask, (x1, y1), (x2, y2), 1.0, 1)
                
                # Extend foot area below ankle (simulate actual foot)
                if ankle_y < feat_h - 4:  # Make sure there's room below
                    center = (ankle_x, min(ankle_y + 2, feat_h - 1))
                    axes = (1, 1)  # width, height of ellipse
                    angle = 0
                    startAngle = 0
                    endAngle = 360
                    cv2.ellipse(lower_legs_mask, center, axes, angle, startAngle, endAngle, 1.0, -1)
                
                # Add area extending downward from ankle
                for dy in range(1, 3):  # Extend 2 pixels down
                    y_pos = ankle_y + dy
                    if y_pos < feat_h:
                        width = 1  # Keep narrow
                        x_start = max(0, ankle_x - width)
                        x_end = min(feat_w - 1, ankle_x + width)
                        cv2.line(lower_legs_mask, (x_start, y_pos), (x_end, y_pos), 1.0, 1)
        
        return lower_legs_mask
    
    def _apply_morphological_smoothing(self, temp_masks):
        """Apply morphological operations to smooth masks using GPU operations"""
        for i in range(1, 6):
            if temp_masks[i].max() > 0:
                # Convert to numpy for OpenCV operations to match CPU version exactly
                mask_np = temp_masks[i].cpu().numpy()
                kernel = np.ones((3, 3), np.uint8)
                mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                mask_np = cv2.erode(mask_np, kernel, iterations=1)
                mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.5)
                temp_masks[i] = torch.from_numpy(mask_np).to(self.device)
        return temp_masks
    
    def _apply_priority_assignment(self, temp_masks, feat_h, feat_w):
        """Apply priority-based overlap handling using GPU operations"""
        # Priority order (higher number = higher priority) - match CPU version exactly:
        # Head: 5 (highest), Upper body: 4, Lower body: 3, Foot: 2, Upper legs: 1 (lowest)
        part_priorities = {
            1: 5,  # Head - highest priority
            2: 4,  # Upper body - second highest priority
            3: 3,  # Lower body - middle priority
            4: 1,  # Upper legs (thighs and upper calf) - lowest priority
            5: 2,  # Foot (lower calf + ankle + foot area) - second lowest priority
        }
        
        # Create final masks with priority-based assignment
        final_masks = torch.zeros(1, 6, feat_h, feat_w, device=self.device)
        
        # Create assignment map
        assignment_map = torch.zeros(feat_h, feat_w, device=self.device, dtype=torch.long)
        priority_map = torch.zeros(feat_h, feat_w, device=self.device)
        
        # Threshold for considering a pixel as part of a mask
        activation_threshold = 0.2
        
        # Assign each pixel to the highest priority part - match CPU logic exactly
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
        
        return final_masks, assignment_map
    
    def _apply_final_smoothing(self, final_masks):
        """Apply final smoothing to reduce harsh boundaries using GPU operations"""
        # Use OpenCV operations to match CPU version exactly
        smooth_kernel = np.array([[1, 1, 1],
                                [1, 2, 1],
                                [1, 1, 1]], dtype=np.float32) / 10.0
        
        for i in range(1, 6):
            if final_masks[0, i].max() > 0:
                # Convert to numpy for OpenCV operations
                mask_np = final_masks[0, i].cpu().numpy()
                mask_np = cv2.filter2D(mask_np, -1, smooth_kernel)
                mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.3)
                final_masks[0, i] = torch.from_numpy(mask_np).to(self.device)
        
        return final_masks
    
    def _finalize_masks(self, final_masks, assignment_map):
        """Create background mask and normalize"""
        # Create background mask (pixels not assigned to any part)
        # Use the same logic as the default implementation
        final_masks[0, 0] = (assignment_map == 0).float()
        
        # Ensure each pixel sums to 1 (normalization)
        mask_sum = final_masks.sum(dim=1, keepdim=True)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        final_masks = final_masks / mask_sum
        
        return final_masks
    
    def _draw_line_gpu(self, mask, x1, y1, x2, y2, thickness):
        """Draw a line on GPU tensor using PyTorch operations"""
        if x1 == x2 and y1 == y2:
            # Single point
            if 0 <= x1 < mask.shape[1] and 0 <= y1 < mask.shape[0]:
                mask[y1, x1] = 1.0
            return mask
        
        # Use OpenCV for line drawing to match CPU version exactly
        mask_np = mask.cpu().numpy()
        cv2.line(mask_np, (x1, y1), (x2, y2), 1.0, thickness)
        mask.copy_(torch.from_numpy(mask_np).to(self.device))
        
        return mask
    
    def _dilate_gpu(self, mask, kernel_size):
        """Apply dilation using GPU-accelerated convolution"""
        if kernel_size <= 1:
            return mask
        
        # Store original shape
        original_shape = mask.shape
        
        # Create dilation kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device, dtype=torch.float32)
        
        # Add batch and channel dimensions for convolution
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply dilation using max pooling (equivalent to dilation)
        # Use padding=0 to avoid size changes, then crop if needed
        dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=0)
        
        # Ensure output has the same shape as input
        result = dilated.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        
        # If the result is smaller, pad it back to original size
        if result.shape != original_shape:
            # Calculate padding needed
            pad_h = original_shape[0] - result.shape[0]
            pad_w = original_shape[1] - result.shape[1]
            
            if pad_h > 0 or pad_w > 0:
                # Pad with zeros
                result = F.pad(result, (0, max(0, pad_w), 0, max(0, pad_h)), mode='constant', value=0)
        
        return result
    
    def _erode_gpu(self, mask, kernel_size):
        """Apply erosion using GPU-accelerated convolution"""
        if kernel_size <= 1:
            return mask
        
        # Store original shape
        original_shape = mask.shape
        
        # Create erosion kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device, dtype=torch.float32)
        
        # Add batch and channel dimensions for convolution
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply erosion using min pooling (equivalent to erosion)
        # Use padding=0 to avoid size changes
        eroded = -F.max_pool2d(-mask_4d, kernel_size=kernel_size, stride=1, padding=0)
        
        # Ensure output has the same shape as input
        result = eroded.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
        
        # If the result is smaller, pad it back to original size
        if result.shape != original_shape:
            # Calculate padding needed
            pad_h = original_shape[0] - result.shape[0]
            pad_w = original_shape[1] - result.shape[1]
            
            if pad_h > 0 or pad_w > 0:
                # Pad with zeros
                result = F.pad(result, (0, max(0, pad_w), 0, max(0, pad_h)), mode='constant', value=0)
        
        return result
    
    def _gaussian_blur_gpu(self, mask, kernel_size=3, sigma=0.5):
        """Apply Gaussian blur using GPU-accelerated convolution"""
        if kernel_size <= 1:
            return mask
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
        # Add batch and channel dimensions
        mask_4d = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply convolution
        padding = kernel_size // 2
        blurred = F.conv2d(mask_4d, kernel, padding=padding)
        
        return blurred.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create Gaussian kernel for convolution"""
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Add batch and channel dimensions
        kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return kernel_4d
    
    def _generate_head_mask_gpu(self, keypoints, feat_h, feat_w, scale_x, scale_y, draw_skeleton_line_gpu):
        """Generate head mask using GPU operations"""
        head_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
        # Create head area around nose keypoint
        if keypoints[0, 2] > self.keypoint_confidence_threshold:  # Nose
            nose_x = int(keypoints[0, 0] * scale_x)
            nose_y = int(keypoints[0, 1] * scale_y)
            nose_x = np.clip(nose_x, 0, feat_w - 1)
            nose_y = np.clip(nose_y, 0, feat_h - 1)
            
            # Create circular head area using GPU operations
            head_radius = 4
            y_coords, x_coords = torch.meshgrid(torch.arange(feat_h, device=self.device), 
                                              torch.arange(feat_w, device=self.device), indexing='ij')
            head_mask = ((x_coords - nose_x)**2 + (y_coords - nose_y)**2 <= head_radius**2).float()
            
            # Add connections to eyes and ears if available
            head_mask += draw_skeleton_line_gpu(0, 1, thickness=1)  # nose to left eye
            head_mask += draw_skeleton_line_gpu(0, 2, thickness=1)  # nose to right eye
            head_mask += draw_skeleton_line_gpu(1, 3, thickness=1)  # left eye to left ear
            head_mask += draw_skeleton_line_gpu(2, 4, thickness=1)  # right eye to right ear
        
        return head_mask
    
    def _generate_upper_body_mask_gpu(self, keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between_gpu):
        """Generate upper body mask using GPU operations"""
        upper_body_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
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
            
            # Create upper torso area using GPU polygon filling
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
                # Use GPU-based polygon filling
                upper_body_mask = self._fill_polygon_gpu(upper_body_mask, scaled_points)
        
        # Add upper arms (shoulder to elbow)
        upper_body_mask += fill_area_between_gpu(5, 7, width=2)  # left upper arm
        upper_body_mask += fill_area_between_gpu(6, 8, width=2)  # right upper arm
        
        return upper_body_mask
    
    def _generate_lower_body_mask_gpu(self, keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between_gpu):
        """Generate lower body mask using GPU operations"""
        lower_body_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
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
                # Use GPU-based polygon filling
                lower_body_mask = self._fill_polygon_gpu(lower_body_mask, scaled_points)
        
        # Add lower arms (elbow to wrist)
        lower_body_mask += fill_area_between_gpu(7, 9, width=2)  # left lower arm
        lower_body_mask += fill_area_between_gpu(8, 10, width=2)  # right lower arm
        
        return lower_body_mask
    
    def _generate_upper_legs_mask_gpu(self, keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between_gpu):
        """Generate upper legs mask using GPU operations"""
        upper_legs_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
        # Add thighs (hip to knee) - upper leg
        upper_legs_mask += fill_area_between_gpu(11, 13, width=2)  # left thigh
        upper_legs_mask += fill_area_between_gpu(12, 14, width=2)  # right thigh
        
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
                
                # Draw partial calf using GPU line drawing
                x1 = int(knee_x * scale_x)
                y1 = int(knee_y * scale_y)
                x2 = int(partial_x * scale_x)
                y2 = int(partial_y * scale_y)
                
                x1, x2 = np.clip([x1, x2], 0, feat_w - 1)
                y1, y2 = np.clip([y1, y2], 0, feat_h - 1)
                
                temp_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
                temp_mask = self._draw_line_gpu(temp_mask, x1, y1, x2, y2, 2)
                temp_mask = self._dilate_gpu(temp_mask, 2)
                upper_legs_mask += temp_mask

        # Add knee areas
        for knee_idx in [13, 14]:  # left and right knees
            if knee_idx < len(keypoints) and keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                x = int(keypoints[knee_idx, 0] * scale_x)
                y = int(keypoints[knee_idx, 1] * scale_y)
                x = np.clip(x, 0, feat_w - 1)
                y = np.clip(y, 0, feat_h - 1)
                
                # Create circular knee area using GPU operations
                y_coords, x_coords = torch.meshgrid(torch.arange(feat_h, device=self.device), 
                                                   torch.arange(feat_w, device=self.device), indexing='ij')
                knee_mask = ((x_coords - x)**2 + (y_coords - y)**2 <= 4).float()  # radius=2
                upper_legs_mask += knee_mask

        return upper_legs_mask
    
    def _generate_lower_legs_mask_gpu(self, keypoints, feat_h, feat_w, scale_x, scale_y):
        """Generate lower legs (foot) mask using GPU operations"""
        lower_legs_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
        # Add foot areas around ankles
        for ankle_idx, knee_idx in [(15, 13), (16, 14)]:  # left and right ankles with corresponding knees
            if ankle_idx < len(keypoints) and keypoints[ankle_idx, 2] > self.keypoint_confidence_threshold:
                ankle_x = int(keypoints[ankle_idx, 0] * scale_x)
                ankle_y = int(keypoints[ankle_idx, 1] * scale_y)
                ankle_x = np.clip(ankle_x, 0, feat_w - 1)
                ankle_y = np.clip(ankle_y, 0, feat_h - 1)
                
                # Keep ankle circles thin
                y_coords, x_coords = torch.meshgrid(torch.arange(feat_h, device=self.device), 
                                                   torch.arange(feat_w, device=self.device), indexing='ij')
                ankle_mask = ((x_coords - ankle_x)**2 + (y_coords - ankle_y)**2 <= 1).float()  # radius=1
                lower_legs_mask += ankle_mask
                
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
                    
                    temp_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
                    temp_mask = self._draw_line_gpu(temp_mask, x1, y1, x2, y2, 1)
                    lower_legs_mask += temp_mask
                
                # Extend foot area below ankle (simulate actual foot)
                if ankle_y < feat_h - 4:  # Make sure there's room below
                    center_y = min(ankle_y + 2, feat_h - 1)
                    # Create elliptical foot area using GPU operations
                    y_coords, x_coords = torch.meshgrid(torch.arange(feat_h, device=self.device), 
                                                       torch.arange(feat_w, device=self.device), indexing='ij')
                    foot_mask = ((x_coords - ankle_x)**2 + (y_coords - center_y)**2 <= 1).float()  # radius=1
                    lower_legs_mask += foot_mask
                
                # Add area extending downward from ankle
                for dy in range(1, 3):  # Extend 2 pixels down
                    y_pos = ankle_y + dy
                    if y_pos < feat_h:
                        width = 1  # Keep narrow
                        x_start = max(0, ankle_x - width)
                        x_end = min(feat_w - 1, ankle_x + width)
                        
                        temp_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
                        temp_mask = self._draw_line_gpu(temp_mask, x_start, y_pos, x_end, y_pos, 1)
                        lower_legs_mask += temp_mask
        
        return lower_legs_mask
    
    def _fill_polygon_gpu(self, mask, points):
        """Fill polygon using GPU operations"""
        if len(points) < 3:
            return mask
        
        # Use OpenCV for polygon filling to match CPU version exactly
        mask_np = mask.cpu().numpy()
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask_np, [points_array], 1.0)
        mask.copy_(torch.from_numpy(mask_np).to(self.device))
        
        return mask


def generate_yolo_pose_masks(yolo_model, person_img: Union[np.ndarray, torch.Tensor],
                           keypoint_confidence_threshold: float = 0.5,
                           height: int = 384, width: int = 128,
                           device: Optional[torch.device] = None) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Convenience function to generate YOLO pose masks
    
    Args:
        yolo_model: YOLO model instance for pose estimation
        person_img: Input person image (BGR format for np.ndarray)
        keypoint_confidence_threshold: Confidence threshold for keypoints
        height: Target height for feature maps
        width: Target width for feature maps
        device: Target device for GPU acceleration (if None, uses CPU)
        
    Returns:
        Generated masks with shape (1, 6, feat_h, feat_w) or None if failed
    """
    generator = YOLOPoseMaskGenerator(
        yolo_model=yolo_model,
        keypoint_confidence_threshold=keypoint_confidence_threshold,
        height=height,
        width=width,
        device=device
    )
    return generator.generate_masks(person_img, device)
