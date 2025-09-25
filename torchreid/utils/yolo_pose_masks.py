"""
YOLO Pose-based mask generation utility for BPBreID (GPU-only)

This module provides GPU-accelerated functionality for generating pose-based masks
using YOLO Pose skeleton structure with 5 body sections:
1. Head
2. Upper body (upper half of torso + upper arms)
3. Lower body (lower half of torso + lower arms)
4. Upper legs (thighs and upper calf - stops at 75% to ankle)
5. Foot (lower calf from 75% + ankle + foot area)

All operations are performed on GPU tensors for maximum performance.
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union


__all__ = ['YOLOPoseMaskGenerator', 'generate_yolo_pose_masks']


class YOLOPoseMaskGenerator:
    """
    GPU-accelerated YOLO Pose-based mask generator for BPBreID
    
    All mask generation operations are performed on GPU tensors for maximum performance.
    """
    
    def __init__(self, yolo_model, keypoint_confidence_threshold=0.5, 
                 height=384, width=128, device=None):
        """
        Initialize GPU-accelerated YOLO Pose mask generator
        
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
        Generate pose-based masks using GPU-accelerated YOLO Pose skeleton structure
        
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
            

            
            # Check temp_masks shape before priority assignment
            
            # Apply priority-based overlap handling
            try:
                final_masks, assignment_map = self._apply_priority_assignment(temp_masks, feat_h, feat_w)
            except Exception as e:
                print(f"Error in priority assignment: {e}")
                # Create fallback masks
                final_masks = torch.zeros(1, 6, feat_h, feat_w, device=self.device)
                assignment_map = torch.zeros(feat_h, feat_w, device=self.device, dtype=torch.long)
            
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
    
    
    
    
    
    
    
    def _apply_priority_assignment(self, temp_masks, feat_h, feat_w):
        """Apply priority-based overlap handling using GPU operations"""
        # Priority order (higher number = higher priority):
        # Head: 5 (highest), Upper body: 4, Lower body: 3, Foot: 2, Upper legs: 1 (lowest)
        part_priorities = torch.tensor([0, 4, 5, 3, 1, 2], device=self.device, dtype=torch.float32)  # 0 for background
        
        # Create final masks with priority-based assignment
        final_masks = torch.zeros(1, 6, feat_h, feat_w, device=self.device)
        
        # Threshold for considering a pixel as part of a mask
        activation_threshold = 0.2
        
        # Create activation masks for each part
        activation_masks = (temp_masks > activation_threshold).float()
        
        # Apply priorities to activation masks - ensure proper broadcasting
        # temp_masks shape: [6, feat_h, feat_w]
        # part_priorities shape: [6] -> need to reshape to [6, 1, 1] for broadcasting
        priority_masks = activation_masks * part_priorities.view(6, 1, 1)
        
        # Find the part with highest priority for each pixel
        max_priorities, assignment_map = torch.max(priority_masks, dim=0)
        
        # Create hard masks based on assignment
        for part_idx in range(1, 6):
            final_masks[0, part_idx] = (assignment_map == part_idx).float()
        
        return final_masks, assignment_map
    
    def _apply_final_smoothing(self, final_masks):
        """Apply final smoothing to reduce harsh boundaries using GPU operations"""
        # Create smooth kernel for GPU operations
        smooth_kernel = torch.tensor([[1, 1, 1],
                                     [1, 2, 1],
                                     [1, 1, 1]], device=self.device, dtype=torch.float32) / 10.0
        
        for i in range(1, 6):
            if final_masks[0, i].max() > 0:
                # Apply custom smoothing kernel
                mask_4d = final_masks[0, i].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                kernel_4d = smooth_kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
                
                # Apply convolution
                smoothed = F.conv2d(mask_4d, kernel_4d, padding=1)
                
                
                final_masks[0, i] = smoothed
        
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
        
        # Create coordinate grids
        h, w = mask.shape
        y_coords, x_coords = torch.meshgrid(torch.arange(h, device=self.device), 
                                           torch.arange(w, device=self.device), indexing='ij')
        
        # Calculate distance from line
        if x2 != x1:
            # Non-vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
            # Line equation: y = slope * x + intercept => slope * x - y + intercept = 0
            # So a = slope, b = -1, c = intercept
            a, b, c = slope, -1, intercept
            
            # Convert to tensors to ensure proper operations
            a_tensor = torch.tensor(a, device=self.device, dtype=torch.float32)
            b_tensor = torch.tensor(b, device=self.device, dtype=torch.float32)
            c_tensor = torch.tensor(c, device=self.device, dtype=torch.float32)
            
            distance = torch.abs(a_tensor * x_coords + b_tensor * y_coords + c_tensor) / torch.sqrt(a_tensor**2 + b_tensor**2)
        else:
            # Vertical line
            distance = torch.abs(x_coords - x1)
        
        # Create line mask
        thickness_tensor = torch.tensor(thickness / 2, device=self.device, dtype=torch.float32)
        line_mask = distance <= thickness_tensor
        
        # Ensure we're within the line segment bounds
        if x1 != x2:
            # For non-vertical lines, check if point is within x bounds
            x_min, x_max = min(x1, x2), max(x1, x2)
            within_bounds = (x_coords >= x_min) & (x_coords <= x_max)
        else:
            # For vertical lines, check if point is within y bounds
            y_min, y_max = min(y1, y2), max(y1, y2)
            within_bounds = (y_coords >= y_min) & (y_coords <= y_max)
        
        line_mask = line_mask & within_bounds
        mask[line_mask] = 1.0
        
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
    
    
    def _generate_head_mask_gpu(self, keypoints, feat_h, feat_w, scale_x, scale_y, draw_skeleton_line_gpu):
        """Generate head mask using GPU operations - oval shape with more height than width"""
        head_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
        # Create head area around nose keypoint
        if keypoints[0, 2] > self.keypoint_confidence_threshold:  # Nose
            nose_x = int(keypoints[0, 0] * scale_x)
            nose_y = int(keypoints[0, 1] * scale_y)
            nose_x = np.clip(nose_x, 0, feat_w - 1)
            nose_y = np.clip(nose_y, 0, feat_h - 1)
            
            # Create oval head area using GPU operations (more height than width)
            y_coords, x_coords = torch.meshgrid(torch.arange(feat_h, device=self.device), 
                                              torch.arange(feat_w, device=self.device), indexing='ij')
            
            # Oval parameters: height > width for more natural head shape
            head_height = 5 # Vertical radius (height)
            head_width = 3   # Horizontal radius (width)
            
            # Create oval shape: (x-center)^2/a^2 + (y-center)^2/b^2 <= 1
            # where a = width, b = height
            oval_mask = ((x_coords - nose_x)**2 / (head_width**2) + 
                        (y_coords - nose_y)**2 / (head_height**2)) <= 1.0
            head_mask = oval_mask.float()
            
            # Add connections to eyes and ears if available (thinner lines to maintain oval shape)
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
        
        # Add upper arms as more natural shapes instead of thick lines
        # Left upper arm (shoulder to elbow)
        if (keypoints[5, 2] > self.keypoint_confidence_threshold and keypoints[7, 2] > self.keypoint_confidence_threshold):
            left_arm_mask = self._create_arm_mask_gpu(keypoints[5], keypoints[7], feat_h, feat_w, scale_x, scale_y)
            upper_body_mask += left_arm_mask
        
        # Right upper arm (shoulder to elbow)
        if (keypoints[6, 2] > self.keypoint_confidence_threshold and keypoints[8, 2] > self.keypoint_confidence_threshold):
            right_arm_mask = self._create_arm_mask_gpu(keypoints[6], keypoints[8], feat_h, feat_w, scale_x, scale_y)
            upper_body_mask += right_arm_mask
        
        return upper_body_mask
    
    def _create_arm_mask_gpu(self, shoulder_kp, elbow_kp, feat_h, feat_w, scale_x, scale_y):
        """Create a more natural arm mask between shoulder and elbow keypoints"""
        arm_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
        # Scale keypoints
        shoulder_x = int(shoulder_kp[0] * scale_x)
        shoulder_y = int(shoulder_kp[1] * scale_y)
        elbow_x = int(elbow_kp[0] * scale_x)
        elbow_y = int(elbow_kp[1] * scale_y)
        
        # Clip coordinates
        shoulder_x = np.clip(shoulder_x, 0, feat_w - 1)
        shoulder_y = np.clip(shoulder_y, 0, feat_h - 1)
        elbow_x = np.clip(elbow_x, 0, feat_w - 1)
        elbow_y = np.clip(elbow_y, 0, feat_h - 1)
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(torch.arange(feat_h, device=self.device), 
                                           torch.arange(feat_w, device=self.device), indexing='ij')
        
        # Calculate arm direction and length
        dx = elbow_x - shoulder_x
        dy = elbow_y - shoulder_y
        arm_length = torch.sqrt(torch.tensor(dx**2 + dy**2, device=self.device, dtype=torch.float32))
        
        if arm_length > 0:
            # Normalize direction vector
            dx_norm = dx / arm_length
            dy_norm = dy / arm_length
            
            # Create perpendicular vector for arm width
            perp_x = -dy_norm
            perp_y = dx_norm
            
            # Calculate distance from each point to the arm line
            # Vector from shoulder to point
            vec_x = x_coords - shoulder_x
            vec_y = y_coords - shoulder_y
            
            # Project onto arm direction to get position along arm
            t = dx_norm * vec_x + dy_norm * vec_y
            
            # Distance from arm line (perpendicular distance)
            perp_dist = torch.abs(perp_x * vec_x + perp_y * vec_y)
            
            # Arm width varies along the length (thicker at shoulder, thinner at elbow)
            # Use a linear interpolation for width
            max_width = 3.0  # Maximum width at shoulder
            min_width = 1.5  # Minimum width at elbow
            arm_width = max_width - (max_width - min_width) * torch.clamp(t / arm_length, 0, 1)
            
            # Create arm mask: points that are close to the line and within the arm segment
            on_arm = (perp_dist <= arm_width) & (t >= 0) & (t <= arm_length)
            arm_mask[on_arm] = 1.0
        
        return arm_mask
    
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
        
        # Add lower arms (elbow to wrist) using the same method as upper arms
        # Left lower arm (elbow to wrist)
        if (keypoints[7, 2] > self.keypoint_confidence_threshold and keypoints[9, 2] > self.keypoint_confidence_threshold):
            left_lower_arm_mask = self._create_arm_mask_gpu(keypoints[7], keypoints[9], feat_h, feat_w, scale_x, scale_y)
            lower_body_mask += left_lower_arm_mask
        
        # Right lower arm (elbow to wrist)
        if (keypoints[8, 2] > self.keypoint_confidence_threshold and keypoints[10, 2] > self.keypoint_confidence_threshold):
            right_lower_arm_mask = self._create_arm_mask_gpu(keypoints[8], keypoints[10], feat_h, feat_w, scale_x, scale_y)
            lower_body_mask += right_lower_arm_mask
        
        return lower_body_mask
    
    def _generate_upper_legs_mask_gpu(self, keypoints, feat_h, feat_w, scale_x, scale_y, fill_area_between_gpu):
        """Generate upper legs mask using GPU operations"""
        upper_legs_mask = torch.zeros((feat_h, feat_w), device=self.device, dtype=torch.float32)
        
        # Add thighs (hip to knee) - upper leg using the same method as upper arms
        # Left thigh (hip to knee)
        if (keypoints[11, 2] > self.keypoint_confidence_threshold and keypoints[13, 2] > self.keypoint_confidence_threshold):
            left_thigh_mask = self._create_arm_mask_gpu(keypoints[11], keypoints[13], feat_h, feat_w, scale_x, scale_y)
            upper_legs_mask += left_thigh_mask
        
        # Right thigh (hip to knee)
        if (keypoints[12, 2] > self.keypoint_confidence_threshold and keypoints[14, 2] > self.keypoint_confidence_threshold):
            right_thigh_mask = self._create_arm_mask_gpu(keypoints[12], keypoints[14], feat_h, feat_w, scale_x, scale_y)
            upper_legs_mask += right_thigh_mask
        
        # Add partial calves (knee to 75% toward ankle) using the same method as upper arms
        # Left calf (knee to 75% toward ankle)
        if (keypoints[13, 2] > self.keypoint_confidence_threshold and keypoints[15, 2] > self.keypoint_confidence_threshold):
            # Calculate point 75% of the way from knee to ankle
            knee_x = keypoints[13, 0]
            knee_y = keypoints[13, 1]
            ankle_x = keypoints[15, 0]
            ankle_y = keypoints[15, 1]
            
            partial_x = knee_x + 0.75 * (ankle_x - knee_x)
            partial_y = knee_y + 0.75 * (ankle_y - knee_y)
            
            # Create partial calf using the same method as arms
            partial_calf_kp = np.array([partial_x, partial_y, keypoints[15, 2]])  # Use ankle confidence
            left_calf_mask = self._create_arm_mask_gpu(keypoints[13], partial_calf_kp, feat_h, feat_w, scale_x, scale_y)
            upper_legs_mask += left_calf_mask
        
        # Right calf (knee to 75% toward ankle)
        if (keypoints[14, 2] > self.keypoint_confidence_threshold and keypoints[16, 2] > self.keypoint_confidence_threshold):
            # Calculate point 75% of the way from knee to ankle
            knee_x = keypoints[14, 0]
            knee_y = keypoints[14, 1]
            ankle_x = keypoints[16, 0]
            ankle_y = keypoints[16, 1]
            
            partial_x = knee_x + 0.75 * (ankle_x - knee_x)
            partial_y = knee_y + 0.75 * (ankle_y - knee_y)
            
            # Create partial calf using the same method as arms
            partial_calf_kp = np.array([partial_x, partial_y, keypoints[16, 2]])  # Use ankle confidence
            right_calf_mask = self._create_arm_mask_gpu(keypoints[14], partial_calf_kp, feat_h, feat_w, scale_x, scale_y)
            upper_legs_mask += right_calf_mask

        # Add knee areas
        for knee_idx in [13, 14]:  # left and right knees
            if knee_idx < len(keypoints) and keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                x = int(keypoints[knee_idx, 0] * scale_x)
                y = int(keypoints[knee_idx, 1] * scale_y)
                x = np.clip(x, 0, feat_w - 1)
                y = np.clip(y, 0, feat_h - 1)
                
                # Create circular knee area using GPU operations
                try:
                    y_coords, x_coords = torch.meshgrid(torch.arange(feat_h, device=self.device), 
                                                       torch.arange(feat_w, device=self.device), indexing='ij')
                    knee_mask = ((x_coords - x)**2 + (y_coords - y)**2 <= 4).float()  # radius=2
                    upper_legs_mask += knee_mask
                except Exception as e:
                    print(f"Error creating knee mask: {e}")

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
        """Fill polygon using simple and fast GPU operations"""
        if len(points) < 3:
            return mask
        
        h, w = mask.shape
        
        # Convert points to tensor once
        points_tensor = torch.tensor(points, device=self.device, dtype=torch.float32)
        
        # Simple approach: use scanline algorithm optimized for GPU
        # Find min/max y coordinates
        y_min = int(torch.min(points_tensor[:, 1]).item())
        y_max = int(torch.max(points_tensor[:, 1]).item())
        
        # Clip to mask bounds
        y_min = max(0, min(y_min, h - 1))
        y_max = max(0, min(y_max, h - 1))
        
        if y_min >= y_max:
            return mask
        
        # For each scanline, find intersections with polygon edges
        for y in range(y_min, y_max + 1):
            intersections = []
            
            # Check each edge
            for i in range(len(points_tensor)):
                j = (i + 1) % len(points_tensor)
                xi, yi = points_tensor[i]
                xj, yj = points_tensor[j]
                
                # Check if edge intersects with scanline y
                if (yi <= y < yj) or (yj <= y < yi):
                    if yi != yj:  # Avoid division by zero
                        # Calculate x intersection
                        x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi)
                        intersections.append(x_intersect)
            
            # Sort intersections
            intersections.sort()
            
            # Fill between pairs of intersections
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = max(0, int(intersections[i]))
                    x_end = min(w - 1, int(intersections[i + 1]))
                    
                    if x_start <= x_end:
                        mask[y, x_start:x_end + 1] = 1.0
        
        return mask


def generate_yolo_pose_masks(yolo_model, person_img: Union[np.ndarray, torch.Tensor],
                           keypoint_confidence_threshold: float = 0.5,
                           height: int = 384, width: int = 128,
                           device: Optional[torch.device] = None) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Convenience function to generate GPU-accelerated YOLO pose masks
    
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