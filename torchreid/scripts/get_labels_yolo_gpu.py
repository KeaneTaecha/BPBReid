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

import torch.nn.functional as F


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
    GPU-optimized YOLO Pose-based mask generator for BPBreID
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
        
        print(f"YOLO Pose mask generator initialized successfully on {self.device}")
    
    def draw_line_gpu(self, mask, x1, y1, x2, y2, thickness=1):
        """
        Draw a line on GPU tensor using distance field approach
        
        Args:
            mask: torch tensor on GPU (H, W)
            x1, y1, x2, y2: line coordinates
            thickness: line thickness
        """
        h, w = mask.shape
        
        # Create coordinate grids on GPU
        y_coords = torch.arange(h, device=self.device).float().unsqueeze(1)
        x_coords = torch.arange(w, device=self.device).float().unsqueeze(0)
        
        # Vector from point 1 to point 2
        dx = x2 - x1
        dy = y2 - y1
        
        # Length of the line
        line_length = torch.sqrt(dx**2 + dy**2)
        
        if line_length > 0:
            # Normalize direction vector
            dx = dx / line_length
            dy = dy / line_length
            
            # Vector from point 1 to each pixel
            px = x_coords - x1
            py = y_coords - y1
            
            # Project onto line direction
            t = torch.clamp((px * dx + py * dy), 0, line_length)
            
            # Closest point on line segment
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            
            # Distance from pixel to line
            dist = torch.sqrt((x_coords - closest_x)**2 + (y_coords - closest_y)**2)
            
            # Create line mask
            line_mask = (dist <= thickness / 2).float()
            
            return torch.maximum(mask, line_mask)
        
        return mask
    
    def fill_polygon_gpu(self, h, w, points):
        """
        Fill a polygon on GPU using point-in-polygon test
        
        Args:
            h, w: height and width of the mask
            points: list of [x, y] coordinates
        
        Returns:
            torch tensor mask on GPU
        """
        mask = torch.zeros((h, w), device=self.device)
        
        if len(points) < 3:
            return mask
        
        # Convert points to GPU tensor
        points_tensor = torch.tensor(points, device=self.device, dtype=torch.float32)
        
        # Create coordinate grids
        y_coords = torch.arange(h, device=self.device).float().unsqueeze(1)
        x_coords = torch.arange(w, device=self.device).float().unsqueeze(0)
        
        # Point-in-polygon test using ray casting
        n_points = len(points_tensor)
        inside = torch.zeros((h, w), device=self.device, dtype=torch.bool)
        
        for i in range(n_points):
            j = (i + 1) % n_points
            xi, yi = points_tensor[i]
            xj, yj = points_tensor[j]
            
            # Check if ray from point crosses edge
            intersect = ((yi > y_coords) != (yj > y_coords)) & \
                       (x_coords < (xj - xi) * (y_coords - yi) / (yj - yi) + xi)
            
            inside = inside ^ intersect
        
        return inside.float()
    
    def gaussian_blur_gpu(self, tensor, kernel_size=3, sigma=0.5):
        """
        Apply Gaussian blur on GPU tensor
        
        Args:
            tensor: input tensor on GPU (H, W)
            kernel_size: size of Gaussian kernel
            sigma: standard deviation
        
        Returns:
            blurred tensor on GPU
        """
        # Add batch and channel dimensions
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, device=self.device).float() - (kernel_size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Separable convolution for efficiency
        g = g.view(1, 1, 1, kernel_size)
        tensor = F.conv2d(tensor, g, padding=(0, kernel_size // 2))
        g = g.transpose(2, 3)
        tensor = F.conv2d(tensor, g, padding=(kernel_size // 2, 0))
        
        return tensor.squeeze(0).squeeze(0)
    
    def morphology_gpu(self, mask, operation='dilate', kernel_size=3, iterations=1):
        """
        Morphological operations on GPU
        
        Args:
            mask: input tensor on GPU (H, W)
            operation: 'dilate' or 'erode'
            kernel_size: size of structuring element
            iterations: number of iterations
        
        Returns:
            processed mask on GPU
        """
        # Add batch and channel dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Create structuring element
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device)
        
        for _ in range(iterations):
            if operation == 'dilate':
                mask = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
            else:  # erode
                mask = -F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size//2)
        
        return mask.squeeze(0).squeeze(0)
    
    def generate_yolo_pose_masks(self, person_img):
        """Generate pose-based masks using YOLO Pose skeleton structure with 5 body sections - GPU optimized"""
        
        try:
            # Ensure person_img is a numpy array for YOLO
            if isinstance(person_img, torch.Tensor):
                person_img = person_img.cpu().numpy()
            
            # Run YOLO pose estimation (this runs on GPU internally)
            results = self.yolo(person_img, task='pose')
            
            # Check if results exist and have keypoints
            if len(results) == 0:
                return None
            
            if not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
                return None
            
            if len(results[0].keypoints.data) == 0:
                return None
            
            # Keep keypoints on GPU
            keypoints = results[0].keypoints.data[0]  # Keep on GPU, shape: [17, 3]
            
            # Validate keypoints shape
            if keypoints.shape[0] != 17:
                return None
            
            # Create 5-part masks from skeleton
            h, w = person_img.shape[:2]
            feat_h, feat_w = self.config.data.height // 8, self.config.data.width // 8
            
            # Initialize masks on GPU
            temp_masks = torch.zeros(6, feat_h, feat_w, device=self.device)
            
            # Scale factors
            scale_x = feat_w / w
            scale_y = feat_h / h
            
            # Scale keypoints to feature map size (on GPU)
            scaled_keypoints = keypoints.clone()
            scaled_keypoints[:, 0] *= scale_x
            scaled_keypoints[:, 1] *= scale_y
            scaled_keypoints[:, :2] = torch.clamp(scaled_keypoints[:, :2], 
                                                   min=0, 
                                                   max=torch.tensor([feat_w-1, feat_h-1], device=self.device))
            
            # Part 1: Head (index 1) - GPU optimized
            head_mask = torch.zeros((feat_h, feat_w), device=self.device)
            
            if keypoints[0, 2] > self.keypoint_confidence_threshold:  # Nose
                nose_x = scaled_keypoints[0, 0]
                nose_y = scaled_keypoints[0, 1]
                
                # Create circular head area using GPU operations
                y_coords = torch.arange(feat_h, device=self.device).float().unsqueeze(1)
                x_coords = torch.arange(feat_w, device=self.device).float().unsqueeze(0)
                
                head_radius = 4
                head_mask = ((x_coords - nose_x)**2 + (y_coords - nose_y)**2 <= head_radius**2).float()
                
                # Add connections to eyes and ears
                for (i, j) in [(0, 1), (0, 2), (1, 3), (2, 4)]:
                    if keypoints[i, 2] > self.keypoint_confidence_threshold and keypoints[j, 2] > self.keypoint_confidence_threshold:
                        head_mask = self.draw_line_gpu(head_mask, 
                                                       scaled_keypoints[i, 0], scaled_keypoints[i, 1],
                                                       scaled_keypoints[j, 0], scaled_keypoints[j, 1], 
                                                       thickness=1)
            
            temp_masks[1] = head_mask
            
            # Part 2: Upper body (upper half of torso + upper arms) - GPU optimized
            upper_body_mask = torch.zeros((feat_h, feat_w), device=self.device)
            
            # Check if all torso keypoints are valid
            torso_valid = all(keypoints[idx, 2] > self.keypoint_confidence_threshold for idx in [5, 6, 11, 12])
            
            if torso_valid:
                # Calculate torso points on GPU
                left_shoulder = scaled_keypoints[5, :2]
                right_shoulder = scaled_keypoints[6, :2]
                left_hip = scaled_keypoints[11, :2]
                right_hip = scaled_keypoints[12, :2]
                
                shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_mid_y = (left_hip[1] + right_hip[1]) / 2
                upper_torso_y = (shoulder_mid_y + hip_mid_y) / 2
                
                # Create upper torso area
                upper_torso_points = [
                    [left_shoulder[0].item(), left_shoulder[1].item()],
                    [right_shoulder[0].item(), right_shoulder[1].item()],
                    [right_shoulder[0].item(), upper_torso_y.item()],
                    [left_shoulder[0].item(), upper_torso_y.item()]
                ]
                
                upper_body_mask = self.fill_polygon_gpu(feat_h, feat_w, upper_torso_points)
            
            # Add upper arms
            for (shoulder_idx, elbow_idx) in [(5, 7), (6, 8)]:
                if keypoints[shoulder_idx, 2] > self.keypoint_confidence_threshold and keypoints[elbow_idx, 2] > self.keypoint_confidence_threshold:
                    upper_body_mask = self.draw_line_gpu(upper_body_mask,
                                                         scaled_keypoints[shoulder_idx, 0], scaled_keypoints[shoulder_idx, 1],
                                                         scaled_keypoints[elbow_idx, 0], scaled_keypoints[elbow_idx, 1],
                                                         thickness=2)
            
            temp_masks[2] = upper_body_mask
            
            # Part 3: Lower body (lower half of torso + lower arms) - GPU optimized
            lower_body_mask = torch.zeros((feat_h, feat_w), device=self.device)
            
            if torso_valid:
                left_shoulder = scaled_keypoints[5, :2]
                right_shoulder = scaled_keypoints[6, :2]
                left_hip = scaled_keypoints[11, :2]
                right_hip = scaled_keypoints[12, :2]
                
                shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_mid_y = (left_hip[1] + right_hip[1]) / 2
                upper_torso_y = (shoulder_mid_y + hip_mid_y) / 2
                
                lower_torso_points = [
                    [left_shoulder[0].item(), upper_torso_y.item()],
                    [right_shoulder[0].item(), upper_torso_y.item()],
                    [right_hip[0].item(), right_hip[1].item()],
                    [left_hip[0].item(), left_hip[1].item()]
                ]
                
                lower_body_mask = self.fill_polygon_gpu(feat_h, feat_w, lower_torso_points)
            
            # Add lower arms
            for (elbow_idx, wrist_idx) in [(7, 9), (8, 10)]:
                if keypoints[elbow_idx, 2] > self.keypoint_confidence_threshold and keypoints[wrist_idx, 2] > self.keypoint_confidence_threshold:
                    lower_body_mask = self.draw_line_gpu(lower_body_mask,
                                                         scaled_keypoints[elbow_idx, 0], scaled_keypoints[elbow_idx, 1],
                                                         scaled_keypoints[wrist_idx, 0], scaled_keypoints[wrist_idx, 1],
                                                         thickness=2)
            
            temp_masks[3] = lower_body_mask
            
            # Part 4: Upper legs - GPU optimized
            upper_legs_mask = torch.zeros((feat_h, feat_w), device=self.device)
            
            # Process both legs
            for (hip_idx, knee_idx, ankle_idx) in [(11, 13, 15), (12, 14, 16)]:
                # Thigh
                if keypoints[hip_idx, 2] > self.keypoint_confidence_threshold and keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                    upper_legs_mask = self.draw_line_gpu(upper_legs_mask,
                                                         scaled_keypoints[hip_idx, 0], scaled_keypoints[hip_idx, 1],
                                                         scaled_keypoints[knee_idx, 0], scaled_keypoints[knee_idx, 1],
                                                         thickness=2)
                
                # Partial calf (75% from knee to ankle)
                if keypoints[knee_idx, 2] > self.keypoint_confidence_threshold and keypoints[ankle_idx, 2] > self.keypoint_confidence_threshold:
                    knee_pos = scaled_keypoints[knee_idx, :2]
                    ankle_pos = scaled_keypoints[ankle_idx, :2]
                    partial_pos = knee_pos + 0.75 * (ankle_pos - knee_pos)
                    
                    upper_legs_mask = self.draw_line_gpu(upper_legs_mask,
                                                         knee_pos[0], knee_pos[1],
                                                         partial_pos[0], partial_pos[1],
                                                         thickness=2)
                
                # Knee area
                if keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                    knee_x = scaled_keypoints[knee_idx, 0]
                    knee_y = scaled_keypoints[knee_idx, 1]
                    
                    y_coords = torch.arange(feat_h, device=self.device).float().unsqueeze(1)
                    x_coords = torch.arange(feat_w, device=self.device).float().unsqueeze(0)
                    circle_mask = ((x_coords - knee_x)**2 + (y_coords - knee_y)**2 <= 4).float()
                    upper_legs_mask = torch.maximum(upper_legs_mask, circle_mask)
            
            temp_masks[4] = upper_legs_mask
            
            # Part 5: Lower legs (foot) - GPU optimized
            lower_legs_mask = torch.zeros((feat_h, feat_w), device=self.device)
            
            for (ankle_idx, knee_idx) in [(15, 13), (16, 14)]:
                if keypoints[ankle_idx, 2] > self.keypoint_confidence_threshold:
                    ankle_x = scaled_keypoints[ankle_idx, 0]
                    ankle_y = scaled_keypoints[ankle_idx, 1]
                    
                    # Ankle circle
                    y_coords = torch.arange(feat_h, device=self.device).float().unsqueeze(1)
                    x_coords = torch.arange(feat_w, device=self.device).float().unsqueeze(0)
                    circle_mask = ((x_coords - ankle_x)**2 + (y_coords - ankle_y)**2 <= 1).float()
                    lower_legs_mask = torch.maximum(lower_legs_mask, circle_mask)
                    
                    # Lower calf (from 75% to ankle)
                    if keypoints[knee_idx, 2] > self.keypoint_confidence_threshold:
                        knee_pos = scaled_keypoints[knee_idx, :2]
                        ankle_pos = scaled_keypoints[ankle_idx, :2]
                        start_pos = knee_pos + 0.75 * (ankle_pos - knee_pos)
                        
                        lower_legs_mask = self.draw_line_gpu(lower_legs_mask,
                                                            start_pos[0], start_pos[1],
                                                            ankle_pos[0], ankle_pos[1],
                                                            thickness=1)
                    
                    # Foot area below ankle
                    if ankle_y < feat_h - 4:
                        foot_center_y = torch.min(ankle_y + 2, torch.tensor(feat_h - 1, device=self.device, dtype=torch.float))
                        ellipse_mask = ((x_coords - ankle_x)**2 + ((y_coords - foot_center_y)**2) * 0.5 <= 1).float()
                        lower_legs_mask = torch.maximum(lower_legs_mask, ellipse_mask)
            
            temp_masks[5] = lower_legs_mask
            
            # Apply morphological operations on GPU
            for i in range(1, 6):
                if temp_masks[i].max() > 0:
                    temp_masks[i] = self.morphology_gpu(temp_masks[i], 'dilate', kernel_size=3, iterations=1)
                    temp_masks[i] = self.morphology_gpu(temp_masks[i], 'erode', kernel_size=3, iterations=1)
                    temp_masks[i] = self.gaussian_blur_gpu(temp_masks[i], kernel_size=3, sigma=0.5)
            
            # PRIORITY-BASED OVERLAP HANDLING - GPU optimized
            part_priorities = torch.tensor([0, 5, 4, 3, 1, 2], device=self.device)  # 0 for background
            
            # Create final masks with priority-based assignment
            final_masks = torch.zeros(1, 6, feat_h, feat_w, device=self.device)
            
            # Threshold for considering a pixel as part of a mask
            activation_threshold = 0.2
            
            # Create activation masks
            active_masks = temp_masks > activation_threshold
            
            # For each pixel, find the part with highest priority
            # Stack all masks and their priorities
            stacked_masks = active_masks.unsqueeze(-1)  # [6, H, W, 1]
            priorities_expanded = part_priorities.view(6, 1, 1, 1).expand(6, feat_h, feat_w, 1)
            
            # Mask out inactive parts
            priorities_masked = torch.where(stacked_masks, priorities_expanded, torch.zeros_like(priorities_expanded))
            
            # Find maximum priority for each pixel
            max_priorities, _ = priorities_masked.max(dim=0)  # [H, W, 1]
            
            # Assign each pixel to the highest priority part
            for part_idx in range(6):
                part_priority = part_priorities[part_idx]
                final_masks[0, part_idx] = ((priorities_masked[part_idx] == max_priorities) & 
                                           (max_priorities > 0)).squeeze(-1).float()
            
            # Apply smoothing on GPU
            for i in range(1, 6):
                if final_masks[0, i].max() > 0:
                    final_masks[0, i] = self.gaussian_blur_gpu(final_masks[0, i], kernel_size=3, sigma=0.3)
            
            # Create background mask
            final_masks[0, 0] = 1.0 - final_masks[0, 1:].sum(dim=0)
            final_masks[0, 0] = torch.clamp(final_masks[0, 0], min=0, max=1)
            
            # Ensure each pixel sums to 1 (normalization)
            mask_sum = final_masks.sum(dim=1, keepdim=True)
            mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
            final_masks = final_masks / mask_sum
            
            # Return as numpy array for compatibility
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
        
        # Print device information for BatchYOLOPose
        print(f"* BatchYOLOPose Device: {str(self.mask_generator.device).upper()}")
        
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

    # Print device information
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"* Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"* CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"* CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

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
