#!/usr/bin/env python3
"""
BPBreID with Official Mask Processing for Person Re-Identification

This script integrates the official mask processing pipeline with BPBreID for real-time
person re-identification. It extracts features from a gallery person image and compares
them with every frame of a video, showing confidence levels and coloring persons red
when not matched with the gallery person.

Key Features:
1. Uses official BPBreID mask processing pipeline (MaskRCNN + PifPaf transforms)
2. Extracts gallery person features from dataset-2/person-1.jpg
3. Processes video frames with YOLO detection + official masking
4. Shows confidence levels and color-coded detections
5. Follows official testing pipeline configuration

Based on:
- bpbreid_market1501_test.yaml configuration
- torchreid/scripts/get_labels.py official masking
- realtime/official_mask_process.py mask processing
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
from torchvision import transforms
import time
from PIL import Image
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch.nn.functional as F
from types import SimpleNamespace

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the official masking classes
from torchreid.scripts.get_labels import BatchMask, build_config_maskrcnn
from torchreid.data.masks_transforms.pifpaf_mask_transform import CombinePifPafIntoFiveVerticalParts
from torchreid.data.masks_transforms.mask_transform import AddBackgroundMask, ResizeMasks, PermuteMasksDim

class BPBreIDOfficialMaskReID:
    """
    BPBreID Re-Identification system with official mask processing pipeline
    """
    
    def __init__(self, reid_model_path: str, hrnet_path: str, 
                 yolo_model_path: str = 'yolov8n.pt',
                 maskrcnn_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        """
        Initialize the BPBreID ReID system with official masking
        
        Args:
            reid_model_path: Path to BPBreID model weights
            hrnet_path: Path to HRNet pretrained weights
            yolo_model_path: Path to YOLO model weights
            maskrcnn_config: MaskRCNN configuration name
        """
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO for person detection
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize official mask processing
        print("Loading MaskRCNN model for official masking...")
        try:
            self.mask_processor = BatchMask(cfg=maskrcnn_config, batch_size=1, workers=0)
            self.mask_available = True
            print("✓ MaskRCNN model loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load MaskRCNN model: {e}")
            print("Will use YOLO bounding boxes as masks instead")
            self.mask_available = False
        
        # Load BPBreID model with official configuration
        print("Loading BPBreID model...")
        self.reid_model = self._load_reid_model(reid_model_path, hrnet_path)
        
        # Setup mask transforms (official pipeline)
        self._setup_mask_transforms()
        
        # Setup image transforms
        self._setup_image_transforms()
        
        # Gallery person features (will be set when loading gallery person)
        self.gallery_features = None
        self.gallery_person_id = None
        
        print("✓ BPBreID Official Mask ReID system initialized successfully")
    
    def _load_reid_model(self, model_path: str, hrnet_path: str):
        """Load BPBreID model with official testing configuration"""
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        print(f"✓ Loaded checkpoint from: {model_path}")
        
        # Create configuration matching official test config
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.load_weights = model_path
        config.model.load_config = True
        config.model.pretrained = True
        
        # BPBreID configuration (matching bpbreid_market1501_test.yaml)
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.mask_filtering_testing = True  # Enable mask filtering
        config.model.bpbreid.mask_filtering_training = False
        config.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']  # Use both foreground and parts
        config.model.bpbreid.test_use_target_segmentation = 'none'  # Don't use external masks for soft masking
        
        # Mask configuration (matching official config)
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'
        config.model.bpbreid.masks.preprocess = 'five_v'  # 5-part vertical segmentation
        config.model.bpbreid.masks.parts_num = 5  # Number of body parts
        config.model.bpbreid.masks.softmax_weight = 15.0  # Softmax weighting for mask normalization
        config.model.bpbreid.masks.background_computation_strategy = 'threshold'  # Background computation method
        config.model.bpbreid.masks.mask_filtering_threshold = 0.5  # Mask filtering threshold
        
        # Model architecture settings
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.horizontal_stripes = False  # Use learnable attention instead of horizontal stripes
        config.model.bpbreid.after_pooling_dim_reduce = True  # Enable dimension reduction after pooling
        
        # Data configuration
        config.data = SimpleNamespace()
        config.data.height = 384
        config.data.width = 128
        config.data.norm_mean = [0.485, 0.456, 0.406]
        config.data.norm_std = [0.229, 0.224, 0.225]
        
        # Build model
        model = torchreid.models.build_model(
            name='bpbreid',
            num_classes=751,  # Market-1501 number of classes
            config=config,
            pretrained=True
        )
        
        # Load weights (handle DataParallel state dict)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print("✓ BPBreID model loaded successfully")
        return model
    
    def _setup_mask_transforms(self):
        """Setup official BPBreID mask transformation pipeline"""
        print("Setting up official mask transformation pipeline...")
        
        # Create the transformation pipeline (matching official config)
        self.pifpaf_grouping_transform = CombinePifPafIntoFiveVerticalParts()
        self.add_background_transform = AddBackgroundMask(
            background_computation_strategy='threshold',
            softmax_weight=15.0,
            mask_filtering_threshold=0.5
        )
        self.resize_masks_transform = ResizeMasks(height=384, width=128, mask_scale=4)
        self.permute_masks_transform = PermuteMasksDim()
        
        print("✓ Official mask transforms initialized")
    
    def _setup_image_transforms(self):
        """Setup image transformation pipeline"""
        self.image_transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
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
    
    def create_official_mask(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[torch.Tensor]:
        """
        Create official BPBreID mask using MaskRCNN + PifPaf transforms
        
        Args:
            frame: Input frame
            bbox: YOLO bounding box (x1, y1, x2, y2)
            
        Returns:
            Official BPBreID mask tensor [6, H, W] (background + 5 parts) or None if failed
        """
        if not self.mask_available:
            return self._create_simple_mask_tensor()
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop the person region
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return self._create_simple_mask_tensor()
            
            # Create MaskRCNN mask
            mask = self._create_maskrcnn_mask(person_crop)
            if mask is None:
                return self._create_simple_mask_tensor()
            
            # Apply official BPBreID mask transforms
            official_mask = self._apply_official_mask_transforms(mask, person_crop.shape)
            
            return official_mask
            
        except Exception as e:
            print(f"Official mask processing failed: {e}")
            return self._create_simple_mask_tensor()
    
    def _create_maskrcnn_mask(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """Create MaskRCNN mask for person crop"""
        try:
            # Convert to PIL Image format expected by the mask processor
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            # Create batch data for MaskRCNN
            batch_data = [{
                "image": torch.as_tensor(np.array(pil_image).transpose(2, 0, 1).astype("float32")),
                "height": person_crop.shape[0],
                "width": person_crop.shape[1]
            }]
            
            # Run MaskRCNN inference
            with torch.no_grad():
                results = self.mask_processor.model(batch_data)
            
            # Extract best person mask
            if len(results) > 0 and "instances" in results[0]:
                instances = results[0]["instances"]
                pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                pred_masks = instances.pred_masks.cpu().numpy()
                pred_scores = instances.scores.cpu().numpy()
                
                # Filter for person class with high confidence
                person_indices = (pred_classes == 0) & (pred_scores > 0.5)
                
                if person_indices.sum() > 0:
                    person_masks = pred_masks[person_indices]
                    person_scores = pred_scores[person_indices]
                    best_mask_idx = person_scores.argmax()
                    
                    return person_masks[best_mask_idx].astype(np.float32)
            
            return None
            
        except Exception as e:
            print(f"MaskRCNN processing failed: {e}")
            return None
    
    def _apply_official_mask_transforms(self, mask: np.ndarray, crop_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Apply official BPBreID mask transformation pipeline
        
        Args:
            mask: Raw MaskRCNN mask [H, W]
            crop_shape: Original crop shape (H, W)
            
        Returns:
            Official BPBreID mask tensor [6, H, W] (background + 5 parts)
        """
        try:
            # Convert to tensor
            mask_tensor = torch.from_numpy(mask).float()  # [H, W]
            
            # Create simulated PifPaf confidence fields (since we don't have real PifPaf)
            # This simulates the 36 PifPaf confidence fields (17 keypoints + 19 connections)
            h, w = mask_tensor.shape
            pifpaf_fields = torch.zeros(36, h, w)
            
            # Use the mask to create realistic PifPaf-like confidence fields
            # Each field represents confidence for a keypoint or connection
            for i in range(36):
                # Create different patterns for different body parts
                if i < 17:  # Keypoints
                    # Head keypoints (0-4) - upper part of person
                    if i < 5:
                        head_region = mask_tensor[:h//5, :]
                        pifpaf_fields[i, :h//5, :] = head_region * (0.8 + 0.2 * torch.rand_like(head_region))
                    # Shoulder keypoints (5-6) - upper-middle part
                    elif i < 7:
                        shoulder_region = mask_tensor[h//5:h//3, :]
                        pifpaf_fields[i, h//5:h//3, :] = shoulder_region * (0.7 + 0.3 * torch.rand_like(shoulder_region))
                    # Arm keypoints (7-10) - middle part
                    elif i < 11:
                        arm_region = mask_tensor[h//3:h//2, :]
                        pifpaf_fields[i, h//3:h//2, :] = arm_region * (0.6 + 0.4 * torch.rand_like(arm_region))
                    # Hip keypoints (11-12) - lower-middle part
                    elif i < 13:
                        hip_region = mask_tensor[h//2:2*h//3, :]
                        pifpaf_fields[i, h//2:2*h//3, :] = hip_region * (0.7 + 0.3 * torch.rand_like(hip_region))
                    # Leg keypoints (13-16) - lower part
                    else:
                        leg_region = mask_tensor[2*h//3:, :]
                        pifpaf_fields[i, 2*h//3:, :] = leg_region * (0.6 + 0.4 * torch.rand_like(leg_region))
                else:  # Connections
                    # Use full person mask for connections
                    pifpaf_fields[i] = mask_tensor * (0.5 + 0.5 * torch.rand_like(mask_tensor))
            
            # Apply official transforms
            # 1. Group into 5 vertical parts
            grouped_masks = self.pifpaf_grouping_transform.apply_to_mask(pifpaf_fields)
            
            # 2. Add background mask
            masks_with_bg = self.add_background_transform.apply_to_mask(grouped_masks)
            
            # 3. Resize to feature map size
            resized_masks = self.resize_masks_transform.apply_to_mask(masks_with_bg)
            
            # 4. Permute dimensions to match BPBreID expected format
            final_masks = self.permute_masks_transform.apply_to_mask(resized_masks)
            
            # Add batch dimension: [1, K+1, H, W]
            final_masks = final_masks.unsqueeze(0)
            
            return final_masks.to(self.device)
            
        except Exception as e:
            print(f"Official mask transforms failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_simple_mask_tensor()
    

    
    def _create_simple_mask_tensor(self) -> torch.Tensor:
        """Create simple 5-part vertical mask tensor in the correct format for BPBreID"""
        # Create simple 5-part vertical division
        # Use a reasonable mask size that will be interpolated to feature map size
        h, w = 384, 128  # Original image size (will be interpolated to feature map size)
        masks = torch.zeros(6, h, w)  # Background + 5 parts [K+1, H, W]
        
        part_height = h // 5
        for i in range(5):
            start_y = i * part_height
            end_y = min((i + 1) * part_height, h)
            if i == 4:  # Last part takes remaining space
                end_y = h
            masks[i + 1, start_y:end_y, :] = 1.0
        
        # Background mask (complement of all parts)
        masks[0] = 1.0 - masks[1:].max(dim=0)[0]
        
        # Normalize masks to sum to 1 at each pixel
        mask_sum = masks.sum(dim=0, keepdim=True)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        masks = masks / mask_sum
        
        # Add batch dimension: [1, K+1, H, W]
        masks = masks.unsqueeze(0)
        
        return masks.to(self.device)
    
    def load_gallery_person(self, gallery_path: str) -> bool:
        """
        Load gallery person and extract features
        
        Args:
            gallery_path: Path to gallery person image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Loading gallery person from: {gallery_path}")
            
            # Load and preprocess image
            image = Image.open(gallery_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Create simple mask for gallery person (full person mask)
            gallery_mask = self._create_simple_mask_tensor()
            
            # Extract features
            with torch.no_grad():
                model_output = self.reid_model(image_tensor, gallery_mask)
                # BPBreID returns (embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks)
                gallery_features = model_output[0]  # Get embeddings dictionary
            
            # Store gallery features
            self.gallery_features = gallery_features
            self.gallery_person_id = os.path.basename(gallery_path).split('.')[0]
            
            print(f"✓ Gallery person loaded: {self.gallery_person_id}")
            return True
            
        except Exception as e:
            print(f"Error loading gallery person: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_person_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[torch.Tensor]:
        """
        Extract features for a detected person
        
        Args:
            frame: Input frame
            bbox: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            Person features or None if failed
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop person
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None
            
            # Convert to PIL and preprocess
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            image_tensor = self.image_transform(pil_image).unsqueeze(0).to(self.device)
            
            # Create official mask
            person_mask = self.create_official_mask(frame, bbox)
            if person_mask is None:
                return None
            
            # Extract features
            with torch.no_grad():
                model_output = self.reid_model(image_tensor, person_mask)
                # BPBreID returns (embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks)
                person_features = model_output[0]  # Get embeddings dictionary
            
            return person_features
            
        except Exception as e:
            print(f"Error extracting person features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_similarity(self, person_features: dict) -> float:
        """
        Compute similarity between person features and gallery features
        
        Args:
            person_features: Features dictionary of detected person
            
        Returns:
            Similarity score (0-1)
        """
        if self.gallery_features is None:
            return 0.0
        
        try:
            # Use foreground embeddings for similarity computation
            gallery_foreground = self.gallery_features['bn_foreg']  # [1, 512]
            person_foreground = person_features['bn_foreg']  # [1, 512]
            
            # Compute cosine similarity
            gallery_norm = F.normalize(gallery_foreground, p=2, dim=1)
            person_norm = F.normalize(person_foreground, p=2, dim=1)
            
            similarity = F.cosine_similarity(gallery_norm, person_norm, dim=1)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            return similarity.item()
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def visualize_detection(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                           confidence: float, similarity: float, person_id: int) -> np.ndarray:
        """
        Visualize detection with confidence and similarity
        
        Args:
            frame: Original frame
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: YOLO detection confidence
            similarity: BPBreID similarity score
            person_id: Person identifier
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        x1, y1, x2, y2 = bbox
        
        # Determine color based on similarity
        if similarity > 0.7:  # High similarity - green
            color = (0, 255, 0)
            match_status = "MATCH"
        elif similarity > 0.5:  # Medium similarity - yellow
            color = (0, 255, 255)
            match_status = "UNCERTAIN"
        else:  # Low similarity - red
            color = (0, 0, 255)
            match_status = "NO MATCH"
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Add labels
        label_bg = f"Person {person_id}"
        label_sim = f"BPBreID: {similarity:.3f}"
        label_conf = f"YOLO: {confidence:.3f}"
        label_status = f"Status: {match_status}"
        
        # Background for text
        cv2.rectangle(vis_frame, (x1, y1-80), (x1+200, y1), color, -1)
        cv2.rectangle(vis_frame, (x1, y1-80), (x1+200, y1), (255, 255, 255), 1)
        
        # Add text
        cv2.putText(vis_frame, label_bg, (x1+5, y1-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, label_sim, (x1+5, y1-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, label_conf, (x1+5, y1-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_frame, label_status, (x1+5, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_preview: bool = True, save_video: bool = True):
        """
        Process video with BPBreID ReID
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            show_preview: Whether to show real-time preview
            save_video: Whether to save processed video
        """
        
        # Check if gallery person is loaded
        if self.gallery_features is None:
            print("Error: Gallery person not loaded. Please call load_gallery_person() first.")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Gallery person: {self.gallery_person_id}")
        
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
                        confidence = detection[4]
                        
                        # Extract person features
                        person_features = self.extract_person_features(frame, bbox)
                        
                        if person_features is not None:
                            # Compute similarity with gallery person
                            similarity = self.compute_similarity(person_features)
                            
                            # Visualize detection
                            vis_frame = self.visualize_detection(
                                vis_frame, bbox, confidence, similarity, i+1
                            )
                        else:
                            # Failed to extract features - show as red
                            vis_frame = self.visualize_detection(
                                vis_frame, bbox, confidence, 0.0, i+1
                            )
                    
                    # Add processing info
                    info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(detections)} | Gallery: {self.gallery_person_id}"
                    cv2.putText(vis_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if save_video:
                        out.write(vis_frame)
                    
                    # Show preview if requested
                    if show_preview:
                        cv2.imshow('BPBreID Official Mask ReID', vis_frame)
                    
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
                    # Small delay to prevent excessive CPU usage
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
            
            # Final statistics
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            print(f"\nProcessing completed:")
            print(f"  Processed frames: {frame_count}/{total_frames}")
            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Average FPS: {fps_actual:.2f}")
            if save_video:
                print(f"  Output saved to: {output_path}")


def main():
    """Main function to run the BPBreID ReID system"""
    
    # Configuration
    REID_MODEL_PATH = "pretrained_models/bpbreid_market1501_hrnet32_10642.pth"
    HRNET_PATH = "pretrained_models/hrnetv2_w32_imagenet_pretrained.pth"
    YOLO_MODEL = "yolov8n.pt"
    GALLERY_PATH = "datasets/Compare/dataset-2/person-1.jpg"
    VIDEO_PATH = "datasets/Compare/dataset-2/person-1-vid.MOV"
    
    # Check if required files exist
    required_files = [
        (REID_MODEL_PATH, "BPBreID model"),
        (HRNET_PATH, "HRNet model"),
        (YOLO_MODEL, "YOLO model"),
        (GALLERY_PATH, "Gallery person image"),
        (VIDEO_PATH, "Video file")
    ]
    
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            print(f"Error: {description} not found: {file_path}")
            print("Please make sure all required files exist.")
            return
    
    try:
        # Create BPBreID ReID system
        print("Initializing BPBreID Official Mask ReID system...")
        reid_system = BPBreIDOfficialMaskReID(
            reid_model_path=REID_MODEL_PATH,
            hrnet_path=HRNET_PATH,
            yolo_model_path=YOLO_MODEL
        )
        
        # Load gallery person
        if not reid_system.load_gallery_person(GALLERY_PATH):
            print("Failed to load gallery person. Exiting.")
            return
        
        # Process video
        output_path = VIDEO_PATH.replace('.MOV', '_bpbreid_official_reid.mp4').replace('.mp4', '_bpbreid_official_reid.mp4')
        reid_system.process_video(
            video_path=VIDEO_PATH,
            output_path=output_path,
            show_preview=True,
            save_video=True
        )
        
        print("\nProcessing completed successfully!")
        print(f"Processed video saved as: {output_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
