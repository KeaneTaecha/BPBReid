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
from typing import Optional, Union


__all__ = ['YOLOPoseMaskGenerator', 'generate_yolo_pose_masks']


class YOLOPoseMaskGenerator:
    """
    YOLO Pose-based mask generator for BPBreID
    """
    
    def __init__(self, yolo_model, keypoint_confidence_threshold=0.5, 
                 height=384, width=128):
        """
        Initialize YOLO Pose mask generator
        
        Args:
            yolo_model: YOLO model instance for pose estimation
            keypoint_confidence_threshold: Confidence threshold for keypoints
            height: Target height for feature maps (default: 384)
            width: Target width for feature maps (default: 128)
        """
        self.yolo = yolo_model
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        self.height = height
        self.width = width
        
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
            temp_masks = torch.zeros(6, feat_h, feat_w)  # 5 parts + will add background later
            
            # Scale factors for keypoint coordinates
            scale_x = feat_w / w
            scale_y = feat_h / h
            
            # Helper functions
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
            
            def fill_area_between(kp1_idx, kp2_idx, width=1):
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
            head_mask = self._generate_head_mask(keypoints, feat_h, feat_w, scale_x, scale_y, draw_skeleton_line)
            temp_masks[1] = torch.from_numpy(np.clip(head_mask, 0, 1))
            
            # Part 2: Upper body (index 2)
            upper_body_mask = self._generate_upper_body_mask(keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between)
            temp_masks[2] = torch.from_numpy(np.clip(upper_body_mask, 0, 1))
            
            # Part 3: Lower body (index 3)
            lower_body_mask = self._generate_lower_body_mask(keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between)
            temp_masks[3] = torch.from_numpy(np.clip(lower_body_mask, 0, 1))
            
            # Part 4: Upper legs (index 4)
            upper_legs_mask = self._generate_upper_legs_mask(keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between)
            temp_masks[4] = torch.from_numpy(np.clip(upper_legs_mask, 0, 1))
            
            # Part 5: Lower legs (foot) (index 5)
            lower_legs_mask = self._generate_lower_legs_mask(keypoints, feat_h, feat_w, scale_x, scale_y)
            temp_masks[5] = torch.from_numpy(np.clip(lower_legs_mask, 0, 1))
            
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
        """Apply morphological operations to smooth masks"""
        for i in range(1, 6):
            if temp_masks[i].max() > 0:
                mask_np = temp_masks[i].numpy()
                kernel = np.ones((3, 3), np.uint8)
                mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                mask_np = cv2.erode(mask_np, kernel, iterations=1)
                mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.5)
                temp_masks[i] = torch.from_numpy(mask_np)
        return temp_masks
    
    def _apply_priority_assignment(self, temp_masks, feat_h, feat_w):
        """Apply priority-based overlap handling"""
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
        
        return final_masks, assignment_map
    
    def _apply_final_smoothing(self, final_masks):
        """Apply final smoothing to reduce harsh boundaries"""
        smooth_kernel = np.array([[1, 1, 1],
                                [1, 2, 1],
                                [1, 1, 1]], dtype=np.float32) / 10.0
        
        for i in range(1, 6):
            if final_masks[0, i].max() > 0:
                mask_np = final_masks[0, i].numpy()
                mask_np = cv2.filter2D(mask_np, -1, smooth_kernel)
                mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.3)
                final_masks[0, i] = torch.from_numpy(mask_np)
        
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
        device: Target device for output tensor (if None, returns numpy array)
        
    Returns:
        Generated masks with shape (1, 6, feat_h, feat_w) or None if failed
    """
    generator = YOLOPoseMaskGenerator(
        yolo_model=yolo_model,
        keypoint_confidence_threshold=keypoint_confidence_threshold,
        height=height,
        width=width
    )
    return generator.generate_masks(person_img, device)
