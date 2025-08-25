#!/usr/bin/env python3
"""
BPBreID Re-Identification with YOLO Detection and Pose-Based Masking

This script combines:
1. YOLO person detection
2. OpenPifPaf pose-based masking (from yolo_mask_video_processor.py)
3. BPBreID re-identification using official testing pipeline
4. Real-time visualization with confidence scores

Uses the official BPBreID model and testing infrastructure without additional dependencies.
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
from typing import List, Tuple, Dict, Any
import openpifpaf
import torchreid
from torchvision import transforms
import json
from datetime import datetime

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

class BPBreIDYOLOMaskedReID:
    """
    Complete BPBreID re-identification system with YOLO detection and pose-based masking
    """
    
    def __init__(self, reid_model_path, hrnet_path, yolo_model_path='yolov8n.pt', pifpaf_model='shufflenetv2k16'):
        """
        Initialize the complete BPBreID re-identification system
        
        Args:
            reid_model_path: Path to BPBreID model weights
            hrnet_path: Path to HRNet pretrained weights
            yolo_model_path: Path to YOLO model weights
            pifpaf_model: OpenPifPaf model name
        """
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO for person detection
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize OpenPifPaf for pose estimation
        print("Loading OpenPifPaf model for pose-based masking...")
        try:
            self.pifpaf_predictor = openpifpaf.Predictor(
                checkpoint=pifpaf_model, 
                visualize_image=False, 
                visualize_processed_image=False
            )
            self.pifpaf_available = True
            print("OpenPifPaf model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load OpenPifPaf model: {e}")
            print("Will use YOLO bounding boxes as masks instead")
            self.pifpaf_available = False
        
        # Initialize BPBreID model using official pipeline
        print("Loading BPBreID model...")
        self.config = self._create_bpbreid_config(reid_model_path, hrnet_path)
        self.model = self._load_bpbreid_model()
        
        # Setup transforms for BPBreID
        self.setup_transforms()
        
        # Gallery storage
        self.gallery_features = None
        self.gallery_person_id = None
        self.gallery_image = None
        
        # ReID threshold
        self.reid_threshold = 0.4
        
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
        
        print("BPBreID YOLO Masked ReID system initialized successfully")
    
    def _create_bpbreid_config(self, model_path, hrnet_path):
        """Create BPBreID configuration using official settings"""
        from types import SimpleNamespace
        
        config = SimpleNamespace()
        
        # Model configuration
        config.model = SimpleNamespace()
        config.model.name = 'bpbreid'
        config.model.load_weights = model_path
        config.model.pretrained = True
        
        # BPBreid configuration optimized for masking
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        
        # Enhanced attention and masking
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.test_use_target_segmentation = 'soft'
        config.model.bpbreid.testing_binary_visibility_score = False
        config.model.bpbreid.training_binary_visibility_score = False
        config.model.bpbreid.mask_filtering_testing = True
        config.model.bpbreid.mask_filtering_training = True
        
        # Advanced mask configuration
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.masks.preprocess = 'five_v'
        config.model.bpbreid.masks.softmax_weight = 2.0
        config.model.bpbreid.masks.background_computation_strategy = 'diff_from_max'
        config.model.bpbreid.masks.mask_filtering_threshold = 0.2
        
        # Data configuration
        config.data = SimpleNamespace()
        config.data.height = 384
        config.data.width = 128
        config.data.norm_mean = [0.485, 0.456, 0.406]
        config.data.norm_std = [0.229, 0.224, 0.225]
        
        # Test configuration
        config.test = SimpleNamespace()
        config.test.dist_metric = 'euclidean'
        config.test.normalize_feature = True
        config.test.part_based = SimpleNamespace()
        config.test.part_based.dist_combine_strat = 'mean'
        config.test.batch_size_pairwise_dist_matrix = 500
        
        return config
    
    def _load_bpbreid_model(self):
        """Load BPBreID model using official torchreid infrastructure"""
        # Build model directly without datamanager for inference
        model = torchreid.models.build_model(
            name=self.config.model.name,
            num_classes=751,  # Market-1501 has 751 training identities
            loss='part_based',
            pretrained=self.config.model.pretrained,
            use_gpu=self.device.type == 'cuda',
            config=self.config
        )
        
        # Load pretrained weights
        if os.path.exists(self.config.model.load_weights):
            print(f"Loading BPBreID weights from: {self.config.model.load_weights}")
            checkpoint = torch.load(self.config.model.load_weights, map_location=self.device)
            
            # Handle state dict loading
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (DataParallel to single GPU)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                else:
                    new_state_dict[k] = v
            
            # Load the cleaned state dict
            model.load_state_dict(new_state_dict)
            print("BPBreID weights loaded successfully")
        else:
            print(f"Warning: BPBreID weights not found at {self.config.model.load_weights}")
        
        model.eval()
        if self.device.type == 'cuda':
            model = model.cuda()
        
        return model
    
    def setup_transforms(self):
        """Setup transforms for BPBreID input preprocessing"""
        self.transform = transforms.Compose([
            transforms.Resize((self.config.data.height, self.config.data.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.data.norm_mean, std=self.config.data.norm_std)
        ])
    
    def detect_persons_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect persons in frame using YOLO"""
        results = self.yolo(frame, classes=0, conf=0.5)  # class 0 is person
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    detections.append((x1, y1, x2, y2, conf))
        
        return detections
    
    def create_yolo_mask(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Create a simple mask from YOLO bounding box"""
        x1, y1, x2, y2 = bbox
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        return mask
    
    def create_pifpaf_mask(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Create pose-based confidence fields using OpenPifPaf"""
        if not self.pifpaf_available:
            simple_mask = self.create_yolo_mask(frame, bbox)
            return simple_mask, [simple_mask]
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop the person region with padding
            padding = 20
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame.shape[1], x2 + padding)
            crop_y2 = min(frame.shape[0], y2 + padding)
            
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if person_crop.size == 0:
                simple_mask = self.create_yolo_mask(frame, bbox)
                return simple_mask, [simple_mask]
            
            # Convert to PIL Image for OpenPifPaf
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            # Run OpenPifPaf pose estimation
            predictions, gt_anns, image_meta = self.pifpaf_predictor.pil_image(pil_image)
            
            if len(predictions) == 0:
                simple_mask = self.create_yolo_mask(frame, bbox)
                return simple_mask, [simple_mask]
            
            # Use the first pose detection
            pose = predictions[0]
            keypoints = pose.data  # Shape: [17, 3] - (x, y, confidence)
            
            # Create 5-part confidence fields
            part_masks, combined_mask = self._create_confidence_fields(person_crop, keypoints)
            
            # Create full-frame masks
            full_part_masks = []
            full_combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            
            for part_mask in part_masks:
                full_part_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                if part_mask.shape == person_crop.shape[:2]:
                    full_part_mask[crop_y1:crop_y2, crop_x1:crop_x2] = part_mask
                else:
                    mask_resized = cv2.resize(part_mask, (person_crop.shape[1], person_crop.shape[0]))
                    full_part_mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask_resized
                full_part_masks.append(full_part_mask)
            
            # Place combined mask
            if combined_mask.shape == person_crop.shape[:2]:
                full_combined_mask[crop_y1:crop_y2, crop_x1:crop_x2] = combined_mask
            else:
                mask_resized = cv2.resize(combined_mask, (person_crop.shape[1], person_crop.shape[0]))
                full_combined_mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask_resized
            
            return full_combined_mask, full_part_masks
            
        except Exception as e:
            print(f"OpenPifPaf processing failed: {e}")
            simple_mask = self.create_yolo_mask(frame, bbox)
            return simple_mask, [simple_mask]
    
    def _create_confidence_fields(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Create 5-part confidence fields from keypoints"""
        h, w = image.shape[:2]
        
        # Define 5-part body segmentation
        part_definitions = {
            'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
            'upper_torso_arms': [5, 6, 7, 8],  # shoulders, elbows
            'lower_torso_arms': [9, 10, 11, 12],  # wrists, hips
            'legs': [13, 14],  # knees
            'feet': [15, 16]  # ankles
        }
        
        part_masks = []
        
        # Create confidence field for each body part
        for part_name, keypoint_indices in part_definitions.items():
            part_mask = np.zeros((h, w), dtype=np.float32)
            
            # Get valid keypoints for this part
            valid_kps = []
            for idx in keypoint_indices:
                if idx < len(keypoints) and keypoints[idx, 2] > 0.2:
                    valid_kps.append(keypoints[idx])
            
            if len(valid_kps) == 0:
                part_masks.append(part_mask)
                continue
            
            # Create large, continuous body part regions
            if part_name == 'head':
                if len(valid_kps) >= 1:
                    center_x = int(sum(kp[0] for kp in valid_kps) / len(valid_kps))
                    center_y = int(sum(kp[1] for kp in valid_kps) / len(valid_kps))
                    radius = min(h, w) // 6
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    part_mask = np.exp(-dist_sq / (2 * radius ** 2))
                    
            elif part_name in ['upper_torso_arms', 'lower_torso_arms']:
                if len(valid_kps) >= 2:
                    x_coords = [kp[0] for kp in valid_kps]
                    y_coords = [kp[1] for kp in valid_kps]
                    min_x = max(0, int(min(x_coords)) - w // 8)
                    max_x = min(w, int(max(x_coords)) + w // 8)
                    min_y = max(0, int(min(y_coords)) - h // 10)
                    max_y = min(h, int(max(y_coords)) + h // 10)
                    part_mask[min_y:max_y, min_x:max_x] = 0.8
                    
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    gaussian = np.exp(-dist_sq / (2 * (min(h, w) // 4) ** 2))
                    part_mask = np.maximum(part_mask, gaussian * 0.6)
                    
            elif part_name == 'legs':
                if len(valid_kps) >= 1:
                    x_coords = [kp[0] for kp in valid_kps]
                    y_coords = [kp[1] for kp in valid_kps]
                    min_x = max(0, int(min(x_coords)) - w // 12)
                    max_x = min(w, int(max(x_coords)) + w // 12)
                    min_y = max(0, int(min(y_coords)) - h // 15)
                    max_y = min(h, int(max(y_coords)) + h // 15)
                    part_mask[min_y:max_y, min_x:max_x] = 0.7
                    
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    gaussian = np.exp(-dist_sq / (2 * (min(h, w) // 5) ** 2))
                    part_mask = np.maximum(part_mask, gaussian * 0.5)
                    
            elif part_name == 'feet':
                if len(valid_kps) >= 1:
                    center_x = int(sum(kp[0] for kp in valid_kps) / len(valid_kps))
                    center_y = int(sum(kp[1] for kp in valid_kps) / len(valid_kps))
                    radius = min(h, w) // 8
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    part_mask = np.exp(-dist_sq / (2 * radius ** 2))
            
            # Apply Gaussian blur for smoothness
            part_mask = cv2.GaussianBlur(part_mask, (31, 31), 0)
            part_mask = np.maximum(part_mask, 0.1)
            part_masks.append(part_mask)
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.float32)
        for part_mask in part_masks:
            combined_mask = np.maximum(combined_mask, part_mask)
        
        return part_masks, combined_mask
    
    def prepare_bpbreid_input(self, image: np.ndarray, masks: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input for BPBreID model with masks"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        image_tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension
        
        # Prepare masks for BPBreID (5 parts + background)
        if len(masks) == 5:
            # Resize masks to match BPBreID input size
            resized_masks = []
            for mask in masks:
                mask_resized = cv2.resize(mask, (self.config.data.width, self.config.data.height))
                resized_masks.append(mask_resized)
            
            # Create background mask (inverse of all parts)
            background_mask = 1.0 - np.maximum.reduce(resized_masks)
            
            # Stack masks: [background, part1, part2, part3, part4, part5]
            all_masks = np.stack([background_mask] + resized_masks, axis=0)
            masks_tensor = torch.from_numpy(all_masks).float().unsqueeze(0)  # Add batch dimension
        else:
            # Fallback: create simple mask
            simple_mask = cv2.resize(masks[0], (self.config.data.width, self.config.data.height))
            background_mask = 1.0 - simple_mask
            all_masks = np.stack([background_mask, simple_mask], axis=0)
            masks_tensor = torch.from_numpy(all_masks).float().unsqueeze(0)
        
        return image_tensor, masks_tensor
    
    def extract_bpbreid_features(self, image: np.ndarray, masks: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract BPBreID features using the model with masks"""
        with torch.no_grad():
            # Prepare input
            image_tensor, masks_tensor = self.prepare_bpbreid_input(image, masks)
            
            # Move to device
            if self.device.type == 'cuda':
                image_tensor = image_tensor.cuda()
                masks_tensor = masks_tensor.cuda()
            
            # Forward pass with masks
            model_output = self.model(image_tensor, external_parts_masks=masks_tensor)
            
            # Extract features and visibility scores
            embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output
            
            # Get the main embeddings (using 'parts' embedding)
            if 'parts' in embeddings:
                features = embeddings['parts']
            elif 'bn_foreg' in embeddings:
                features = embeddings['bn_foreg']
            else:
                # Fallback to first available embedding
                features = list(embeddings.values())[0]
            
            # Ensure features are 2D: [batch_size, feature_dim]
            if features.dim() == 1:
                features = features.unsqueeze(0)
            elif features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            return features, visibility_scores
    
    def load_gallery_person(self, gallery_path: str):
        """Load gallery person image and extract features"""
        print(f"Loading gallery person from: {gallery_path}")
        
        if not os.path.exists(gallery_path):
            raise FileNotFoundError(f"Gallery image not found: {gallery_path}")
        
        # Load gallery image
        gallery_image = cv2.imread(gallery_path)
        if gallery_image is None:
            raise ValueError(f"Could not load gallery image: {gallery_path}")
        
        # Create simple full-body mask for gallery image
        h, w = gallery_image.shape[:2]
        gallery_mask = np.ones((h, w), dtype=np.float32)
        gallery_masks = [gallery_mask]  # Single mask for gallery
        
        # Extract features
        gallery_features, gallery_visibility = self.extract_bpbreid_features(gallery_image, gallery_masks)
        
        # Store gallery information
        self.gallery_features = gallery_features
        self.gallery_person_id = os.path.basename(gallery_path).split('.')[0]
        self.gallery_image = gallery_image
        
        print(f"Gallery person loaded: {self.gallery_person_id}")
        return gallery_features
    
    def compute_similarity(self, query_features: torch.Tensor, gallery_features: torch.Tensor) -> float:
        """Compute similarity between query and gallery features"""
        # Ensure both tensors have the same shape
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        if gallery_features.dim() == 1:
            gallery_features = gallery_features.unsqueeze(0)
        
        # Normalize features
        query_norm = F.normalize(query_features, p=2, dim=1)
        gallery_norm = F.normalize(gallery_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(query_norm, gallery_norm, dim=1)
        
        # Ensure we get a single scalar value
        if similarity.numel() > 1:
            similarity = similarity.mean()
        
        return similarity.item()
    
    def visualize_reid_result(self, frame: np.ndarray, combined_mask: np.ndarray, part_masks: List[np.ndarray],
                             bbox: Tuple[int, int, int, int, float], similarity: float, is_match: bool) -> np.ndarray:
        """Visualize re-identification results with masks and confidence"""
        vis_frame = frame.copy()
        
        # Define colors for each body part (BGR format)
        part_colors = [
            (255, 0, 0),      # Head - Blue
            (0, 255, 0),      # Upper Torso/Arms - Green
            (0, 0, 255),      # Lower Torso/Arms - Red
            (255, 255, 0),    # Legs - Cyan
            (255, 0, 255),    # Feet - Magenta
        ]
        
        # Apply colored mask overlay
        mask_colored = np.zeros_like(frame, dtype=np.uint8)
        
        if len(part_masks) > 1:
            for i, (part_mask, color) in enumerate(zip(part_masks, part_colors)):
                if part_mask.max() > 0.1:
                    binary_mask = (part_mask > 0.2).astype(np.uint8)
                    mask_colored[binary_mask > 0] = color
        else:
            if combined_mask.max() > 0.1:
                binary_combined = (combined_mask > 0.2).astype(np.uint8)
                mask_colored[binary_combined > 0] = (0, 255, 0)
        
        # Blend mask with frame
        alpha = 0.6
        vis_frame = cv2.addWeighted(vis_frame, 1-alpha, mask_colored, alpha, 0)
        
        # Draw bounding box with color based on match
        x1, y1, x2, y2, conf = bbox
        box_color = (0, 255, 0) if is_match else (0, 0, 255)  # Green if match, Red if no match
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add confidence score in top right of bounding box
        confidence_text = f"ReID: {similarity:.3f}"
        text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x2 - text_size[0] - 5
        text_y = y1 + text_size[1] + 5
        
        # Draw text background
        cv2.rectangle(vis_frame, (text_x - 2, text_y - text_size[1] - 2), 
                     (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(vis_frame, confidence_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add match status
        status_text = "MATCH" if is_match else "NO MATCH"
        status_color = (0, 255, 0) if is_match else (0, 0, 255)
        status_y = y2 + 20
        
        cv2.putText(vis_frame, status_text, (x1, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return vis_frame
    
    def process_video(self, video_path: str, gallery_path: str, output_path: str = None, 
                     show_preview: bool = True, save_video: bool = True):
        """Process video with BPBreID re-identification"""
        
        # Load gallery person
        self.load_gallery_person(gallery_path)
        
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
                output_path = video_path.replace('.MOV', '_bpbreid_reid.mp4').replace('.mp4', '_bpbreid_reid.mp4')
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Detect persons using YOLO
                    detections = self.detect_persons_yolo(frame)
                    
                    # Process each detected person
                    vis_frame = frame.copy()
                    
                    for i, detection in enumerate(detections):
                        bbox = detection[:4]  # (x1, y1, x2, y2)
                        
                        # Create mask using pose-based method
                        if self.pifpaf_available:
                            combined_mask, part_masks = self.create_pifpaf_mask(frame, bbox)
                        else:
                            combined_mask = self.create_yolo_mask(frame, bbox)
                            part_masks = [combined_mask]
                        
                        # Extract BPBreID features
                        try:
                            query_features, query_visibility = self.extract_bpbreid_features(frame, part_masks)
                            
                            # Compute similarity with gallery person
                            similarity = self.compute_similarity(query_features, self.gallery_features)
                            
                            # Determine if it's a match
                            is_match = similarity > self.reid_threshold
                            
                            # Update statistics
                            self.test_results['frames_with_detection'] += 1
                            if is_match:
                                self.test_results['frames_with_match'] += 1
                            else:
                                self.test_results['frames_without_match'] += 1
                            
                            self.test_results['similarity_scores'].append(similarity)
                            
                            # Visualize result
                            vis_frame = self.visualize_reid_result(vis_frame, combined_mask, part_masks, 
                                                                  detection, similarity, is_match)
                            
                        except Exception as e:
                            print(f"BPBreID processing failed for frame {frame_count}: {e}")
                            # Fallback visualization
                            x1, y1, x2, y2, conf = detection
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(vis_frame, "ERROR", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add processing info
                    info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(detections)} | Gallery: {self.gallery_person_id}"
                    cv2.putText(vis_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add threshold info
                    threshold_text = f"Threshold: {self.reid_threshold}"
                    cv2.putText(vis_frame, threshold_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if save_video:
                        out.write(vis_frame)
                    
                    # Show preview if requested
                    if show_preview:
                        cv2.imshow('BPBreID YOLO Masked ReID', vis_frame)
                    
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
                    elif key == ord('p'):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                else:
                    time.sleep(0.001)
        
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
                
                print(f"\nReID Results:")
                print(f"  Total frames: {self.test_results['total_frames']}")
                print(f"  Frames with detection: {self.test_results['frames_with_detection']}")
                print(f"  Frames with match: {self.test_results['frames_with_match']}")
                print(f"  Match rate: {match_rate:.3f}")
                print(f"  Average similarity: {avg_similarity:.3f}")
                print(f"  ReID threshold: {self.reid_threshold}")
            
            if save_video:
                print(f"  Output saved to: {output_path}")
            
            # Save test results
            results_path = output_path.replace('.mp4', '_results.json') if output_path else 'reid_results.json'
            with open(results_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"  Results saved to: {results_path}")


def main():
    """Main function to run BPBreID re-identification"""
    
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bpbreid_dir = os.path.dirname(current_dir)
    
    # File paths
    VIDEO_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-2-vid.MOV")
    GALLERY_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1.jpg")
    REID_MODEL_PATH = os.path.join(bpbreid_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
    HRNET_PATH = os.path.join(bpbreid_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")
    YOLO_MODEL = "yolov8n.pt"
    
    # Check if files exist
    missing_files = []
    for path, name in [(VIDEO_PATH, "Video"), (GALLERY_PATH, "Gallery image"), 
                       (REID_MODEL_PATH, "BPBreID model"), (YOLO_MODEL, "YOLO model")]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    try:
        # Create BPBreID re-identification system
        print("Initializing BPBreID YOLO Masked ReID system...")
        reid_system = BPBreIDYOLOMaskedReID(
            reid_model_path=REID_MODEL_PATH,
            hrnet_path=HRNET_PATH,
            yolo_model_path=YOLO_MODEL
        )
        
        # Get output path
        video_filename = os.path.basename(VIDEO_PATH)
        output_path = video_filename.replace('.MOV', '_bpbreid_reid.mp4').replace('.mp4', '_bpbreid_reid.mp4')
        
        # Process video
        reid_system.process_video(
            video_path=VIDEO_PATH,
            gallery_path=GALLERY_PATH,
            output_path=output_path,
            show_preview=True,
            save_video=True
        )
        
        print("\nBPBreID re-identification completed successfully!")
        print(f"Processed video saved as: {output_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
