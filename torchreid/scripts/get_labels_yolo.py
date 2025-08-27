import argparse
import glob
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset


def get_image_paths(source, path_format=False):
    """
    Get the paths of all image files in a directory.

    Args:
        source (str): Directory path.
        path_format (bool, optional): Return paths as Path objects if True, otherwise as strings. Default is False.

    Returns:
        image_paths (List[str or Path]): List of image file paths.
    """
    image_paths = glob.glob(f"{source}/**/*.[jJ][pP][gG]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[pP][nN][gG]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[jJ][pP][eE][gG]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[tT][iI][fF]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[tT][iI][fF][fF]", recursive=True)
    if path_format:
        image_paths = [Path(path_str) for path_str in image_paths]
    return image_paths


def format_path(img_path, dataset_dir):
    """
    Formats the given image path based on the dataset directory.

    Args:
        img_path (str): The path of the image file.
        dataset_dir (str): The directory path of the dataset.

    Returns:
        str: The formatted path of the image file.
    """
    if "occluded_reid" in dataset_dir.lower() or "occluded-reid" in dataset_dir.lower():
        return os.path.join(os.path.basename(os.path.dirname(os.path.dirname(img_path))), os.path.basename(img_path))
    elif "p-dukemtmc_reid" in dataset_dir.lower() or "p-dukemtmc-reid" in dataset_dir.lower():
        return os.path.join(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path)))),
                            os.path.basename(os.path.dirname(os.path.dirname(img_path))), os.path.basename(img_path))
    return os.path.relpath(img_path, dataset_dir)


def get_label_paths(img_paths, dataset_dir):
    """
    Get the paths of label files corresponding to the image paths.

    Args:
        img_paths (List[str]): List of image file paths.
        dataset_dir (str): Directory path of the dataset.

    Returns:
        relative_paths (List[str]): List of relative paths of the image files.
        file_paths (List[str]): List of label file paths.
    """
    relative_paths, file_paths = [], []
    for img_name in img_paths:
        relative_path = format_path(img_name, dataset_dir)
        file_path = os.path.join(dataset_dir, "masks", "yolo_pose", relative_path + ".npy")
        relative_paths.append(relative_path)
        file_paths.append(file_path)
    return relative_paths, file_paths


def skip_existing(imagery, dataset_dir):
    """
    Filter out image paths for which label files already exist.

    Args:
        imagery (List[str]): List of image file paths.
        dataset_dir (str): Directory path of the dataset.

    Returns:
        new_imagery (List[str]): List of image file paths for which label files do not exist.
    """
    relative_paths, file_paths = get_label_paths(img_paths=imagery, dataset_dir=dataset_dir)
    new_imagery = []
    for index, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            new_imagery.append(imagery[index])
    return new_imagery


def save_files(files, files_path, verbose=True):
    """
    Save files to specified paths.

    Args:
        files (List[object]): List of files to be saved.
        files_path (List[str]): List of paths where files will be saved.
        verbose (bool, optional): Print progress if True. Default is True.
    """
    for file, file_path in zip(files, files_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, file)
        if verbose:
            print(f"Processed {os.path.basename(file_path)}")


class ImageDataset(Dataset):
    """
    Custom dataset class for loading images.

    Args:
        imagery (List[Path]): List of image file paths.

    Returns:
        (str, np.ndarray): Tuple containing the image file path and the loaded image.
    """

    def __init__(self, imagery: List[Path]):
        self.imagery = imagery

    def __getitem__(self, index):
        return self.imagery[index], cv2.imread(str(self.imagery[index]))

    def __len__(self):
        return len(self.imagery)


class YOLOPoseMaskGenerator:
    """
    YOLO Pose-based mask generator for BPBreID
    Adapted from bpbreid_yolo_masked_reid_fin2.py
    """
    
    def __init__(self, yolo_model_path='yolov8n-pose.pt', keypoint_confidence_threshold=0.5):
        """
        Initialize YOLO Pose mask generator

        Args:
            yolo_model_path: Path to YOLO model weights
            keypoint_confidence_threshold: Confidence threshold for keypoints
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.keypoint_confidence_threshold = keypoint_confidence_threshold
        
        # Load YOLO for pose estimation
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        
        # BPBreID configuration (standard dimensions)
        self.config = type('Config', (), {
            'data': type('Data', (), {
                'height': 384,
                'width': 128
            })()
        })()
        
        print("YOLO Pose mask generator initialized successfully")
    
    def generate_yolo_pose_masks(self, person_img):
        """Generate pose-based masks using YOLO Pose skeleton structure with 5 body sections:
        1. Head
        2. Upper body (upper half of torso + upper arms)
        3. Lower body (lower half of torso + lower arms)
        4. Upper legs (thighs and upper calf - stops at 75% to ankle)
        5. Foot (lower calf from 75% + ankle + foot area)
        """
        
        try:
            # Ensure person_img is a numpy array, not a tensor
            if isinstance(person_img, torch.Tensor):
                person_img = person_img.cpu().numpy()
            
            # Ensure image is in BGR format for YOLO
            if len(person_img.shape) == 3 and person_img.shape[2] == 3:
                # Already BGR from cv2.imread
                pass
            else:
                print(f"Warning: Unexpected image format with shape {person_img.shape}")
                return None
            
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
            
            temp_masks[5] = torch.from_numpy(np.clip(lower_legs_mask, 0, 1))
            
            # Apply morphological operations to smooth masks
            for i in range(1, 6):
                if temp_masks[i].max() > 0:
                    mask_np = temp_masks[i].numpy()
                    kernel = np.ones((3, 3), np.uint8)
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)
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
                    mask_np = cv2.filter2D(mask_np, -1, smooth_kernel)
                    mask_np = cv2.GaussianBlur(mask_np, (3, 3), 0.3)
                    final_masks[0, i] = torch.from_numpy(mask_np)
            
            # Create background mask (pixels not assigned to any part)
            final_masks[0, 0] = (assignment_map == 0).float()
            
            # Ensure each pixel sums to 1 (normalization)
            mask_sum = final_masks.sum(dim=1, keepdim=True)
            mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
            final_masks = final_masks / mask_sum
            
            return final_masks.cpu().numpy()
            
        except Exception as e:
            print(f"YOLO Pose skeleton mask generation failed: {e}")
            return None


class BatchYOLOPose:
    def __init__(self, yolo_model_path: str = "yolov8n-pose.pt", batch_size: int = 1, workers: int = 0):
        """
        Initialize the BatchYOLOPose class for performing batched YOLO pose estimation.

        Args:
            yolo_model_path (str): Path to YOLO model weights.
            batch_size (int, optional): Batch size for processing images. Defaults to 1.
            workers (int, optional): Number of worker processes for data loading. Defaults to 0.
        """
        print(f"* YOLO Pose model -> {yolo_model_path}")
        
        # Initialize YOLO pose mask generator
        self.mask_generator = YOLOPoseMaskGenerator(yolo_model_path)
        
        # Set the batch size for processing images
        self.batch_size = batch_size
        
        # Set the number of worker processes for data loading
        self.workers = workers
        
        # Timing statistics
        self.timing_stats = {
            'total_time': 0.0,
            'yolo_inference_time': 0.0,
            'mask_generation_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': [],
            'per_frame_times': []  # New: track time per individual frame
        }

    def __call__(self, imagery: List[Path] or List[str], dataset_dir: List[Path] or List[str],
                 is_overwrite: bool = False, verbose: bool = False):
        """
        Perform the batch processing of imagery to generate and save YOLO pose mask files.

        Args:
            imagery (List[Path] or List[str]): A list of image paths or image filenames.
            dataset_dir (List[Path] or List[str]): A list of dataset directories.
            is_overwrite (bool, optional): Whether to overwrite existing mask files. Defaults to False.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        """
        assert len(imagery) > 0, "No images found in imagery."

        if not is_overwrite:
            # Skip existing images if overwrite is disabled
            imagery = skip_existing(imagery, dataset_dir)

        # Create an instance of the ImageDataset class
        dataset = ImageDataset(imagery)

        # Create a data loader for batch processing
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True
        )

        total_batches = len(loader)
        progress_bar = tqdm.tqdm(total=total_batches, desc="Processing YOLO Pose", unit="batch")
        
        # Reset timing stats
        self.timing_stats = {
            'total_time': 0.0,
            'yolo_inference_time': 0.0,
            'mask_generation_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': [],
            'per_frame_times': []  # Add this missing key
        }

        with torch.no_grad():
            for batch_idx, (paths, images) in enumerate(loader):
                batch_start_time = time.time()

                # Get the file paths for saving the mask files
                relative_paths, mask_file_paths = get_label_paths(img_paths=paths, dataset_dir=dataset_dir)
                
                # Process each image in the batch
                batch_masks = []
                for img_path, image in zip(paths, images):
                    if image is None:
                        print(f"Warning: Could not load image {img_path}")
                        # Create empty mask for failed images
                        empty_mask = np.zeros((1, 6, 48, 16))  # Standard BPBreID mask size
                        batch_masks.append(empty_mask)
                        continue
                    
                    # Time YOLO pose inference and mask generation per frame
                    frame_start = time.time()
                    masks = self.mask_generator.generate_yolo_pose_masks(image)
                    frame_end = time.time()
                    frame_time = frame_end - frame_start
                    
                    # Record per-frame timing
                    self.timing_stats['per_frame_times'].append(frame_time)
                    
                    if masks is None:
                        print(f"Warning: No pose detected in {img_path} (took {frame_time:.3f}s)")
                        # Create empty mask for images without pose
                        empty_mask = np.zeros((1, 6, 48, 16))  # Standard BPBreID mask size
                        batch_masks.append(empty_mask)
                    else:
                        print(f"Successfully processed {os.path.basename(img_path)} (took {frame_time:.3f}s)")
                        batch_masks.append(masks)
                    
                    # Update timing stats
                    self.timing_stats['yolo_inference_time'] += frame_time

                # Time saving
                save_start = time.time()
                save_files(batch_masks, mask_file_paths, verbose)
                save_end = time.time()
                save_time = save_end - save_start
                
                batch_end_time = time.time()
                batch_total_time = batch_end_time - batch_start_time
                
                # Update timing stats
                self.timing_stats['total_time'] += batch_total_time
                self.timing_stats['saving_time'] += save_time
                self.timing_stats['num_images'] += len(paths)
                
                # Record per-batch timing
                per_image_time = batch_total_time / len(paths)
                self.timing_stats['per_image_times'].extend([per_image_time] * len(paths))
                
                if verbose:
                    print(f"Batch {batch_idx + 1}/{total_batches}: "
                          f"Save: {save_time:.3f}s, "
                          f"Total: {batch_total_time:.3f}s "
                          f"({per_image_time:.3f}s per image)")

                progress_bar.update(1)

            progress_bar.close()
            
        # Print final timing statistics
        self._print_timing_stats("YOLO Pose Processing")
    
    def _print_timing_stats(self, stage_name):
        """Print detailed timing statistics"""
        stats = self.timing_stats
        print(f"\n=== {stage_name} Timing Statistics ===")
        print(f"Total images processed: {stats['num_images']}")
        print(f"Total time: {stats['total_time']:.3f}s")
        print(f"Average time per image: {stats['total_time']/stats['num_images']:.3f}s")
        print(f"YOLO inference time: {stats['yolo_inference_time']:.3f}s ({stats['yolo_inference_time']/stats['total_time']*100:.1f}%)")
        print(f"Saving time: {stats['saving_time']:.3f}s ({stats['saving_time']/stats['total_time']*100:.1f}%)")
        
        # Per-frame timing statistics
        if stats['per_frame_times']:
            print(f"\nPer-frame timing statistics:")
            print(f"  Average time per frame: {np.mean(stats['per_frame_times']):.3f}s")
            print(f"  Min time per frame: {min(stats['per_frame_times']):.3f}s")
            print(f"  Max time per frame: {max(stats['per_frame_times']):.3f}s")
            print(f"  Std dev per frame: {np.std(stats['per_frame_times']):.3f}s")
            print(f"  Frames per second: {1.0/np.mean(stats['per_frame_times']):.2f}")
        
        if stats['per_image_times']:
            print(f"\nPer-batch timing statistics:")
            print(f"  Min time per image: {min(stats['per_image_times']):.3f}s")
            print(f"  Max time per image: {max(stats['per_image_times']):.3f}s")
            print(f"  Std dev per image: {np.std(stats['per_image_times']):.3f}s")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Source dataset containing image files')
    parser.add_argument('--yolo-model-path', type=str, default="yolov8n-pose.pt",
                        help='Path to YOLO model weights')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size for processing images')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    parser.add_argument('--single-image', type=str,
                        help='Process only a single image for timing measurement')
    args = parser.parse_args()

    # Get image paths
    if args.single_image:
        # Process only a single image for timing measurement
        if os.path.exists(args.single_image):
            img_paths = [args.single_image]
            print(f"Processing single image: {args.single_image}")
        else:
            print(f"Error: Image file {args.single_image} not found!")
            return
    else:
        img_paths = get_image_paths(args.source)
        print(f"Found {len(img_paths)} images to process")

    # Overall timing
    total_start_time = time.time()
    
    # Perform YOLO Pose processing
    print("\n" + "="*50)
    print("YOLO POSE MASK GENERATION")
    print("="*50)
    print("Generating 5-part body masks:")
    print("1. Head")
    print("2. Upper body (upper half of torso + upper arms)")
    print("3. Lower body (lower half of torso + lower arms)")
    print("4. Upper legs (thighs and upper calf - stops at 75% to ankle)")
    print("5. Foot (lower calf from 75% + ankle + foot area)")
    print("="*50)
    
    yolo_model = BatchYOLOPose(yolo_model_path=args.yolo_model_path,
                               batch_size=args.batch_size,
                               workers=args.num_workers)
    yolo_model(imagery=img_paths, dataset_dir=args.source, is_overwrite=False, verbose=True)
    
    # Overall timing summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "="*50)
    print("OVERALL TIMING SUMMARY")
    print("="*50)
    print(f"Total processing time: {total_time:.3f}s")
    if args.single_image:
        print(f"Time for single image: {total_time:.3f}s")
    else:
        print(f"Average time per image: {total_time/len(img_paths):.3f}s")
        print(f"Images per second: {len(img_paths)/total_time:.2f}")
    
    print(f"\nMasks saved to: {os.path.join(args.source, 'masks', 'yolo_pose')}")


if __name__ == '__main__':
    main()
