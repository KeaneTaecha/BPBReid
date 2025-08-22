#!/usr/bin/env python3
"""
BPBreID with Official Mask Process for Feature Extraction and Comparison

This script implements the official BPBreID testing pipeline with proper mask usage:
1. Uses the official mask creation process from get_labels.py
2. Applies masks to BPBreID following the official testing configuration
3. Extracts features from gallery image and compares with video frames
4. Ensures proper mask format and preprocessing matching the official pipeline

Key Features:
- Official mask filtering during testing (mask_filtering_testing = True)
- Proper external mask usage (test_use_target_segmentation = 'soft')
- Official mask transforms using 'five_v' preprocessing
- Correct feature extraction and similarity computation
- Real-time video processing with YOLO detection
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
import json
from datetime import datetime
import torch.nn.functional as F
from types import SimpleNamespace
from pathlib import Path
import sys

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the official masking classes
from torchreid.scripts.get_labels import BatchMask, build_config_maskrcnn
from torchreid.data.masks_transforms import masks_preprocess_all
from torchreid.data.masks_transforms.pifpaf_mask_transform import CombinePifPafIntoFiveVerticalParts
from torchreid.data.masks_transforms.mask_transform import AddBackgroundMask
from torchreid.utils.tools import read_image, read_masks
from detectron2.config import CfgNode

class BPBreIDOfficialMaskProcessor:
    """
    BPBreID processor that uses the official mask process for feature extraction
    and comparison between gallery images and video frames
    """
    
    def __init__(self, reid_model_path, hrnet_path, yolo_model_path='yolov8n.pt'):
        """
        Initialize the BPBreID processor with official mask support
        
        Args:
            reid_model_path: Path to BPBreID model weights
            hrnet_path: Path to HRNet weights
            yolo_model_path: Path to YOLO model for person detection
        """
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO for person detection
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize the official BatchMask from get_labels.py
        print("Loading MaskRCNN model for official mask processing...")
        try:
            self.mask_processor = BatchMask(cfg="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
                                           batch_size=1, workers=0)
            self.mask_available = True
            print("MaskRCNN model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load MaskRCNN model: {e}")
            print("Will use YOLO bounding boxes as masks instead")
            self.mask_available = False
        
        # Load BPBreID model with official configuration
        print("Loading BPBreID model...")
        self.model = self._load_bpbreid_model(reid_model_path, hrnet_path)
        
        # Setup transforms
        self.setup_transforms()
        
        # Initialize mask transforms
        self.setup_mask_transforms()
        
        # Gallery storage
        self.gallery_features = None
        self.gallery_image_path = None
        
        print("BPBreID Official Mask Processor initialized successfully")
    
    def _load_bpbreid_model(self, model_path, hrnet_path):
        """Load BPBreID model with official testing configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create config that matches official testing configuration
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.load_weights = model_path
        config.model.load_config = True
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.mask_filtering_testing = True  # Enable mask filtering for testing
        config.model.bpbreid.mask_filtering_training = False
        config.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']  # Use both foreground and parts
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'
        config.model.bpbreid.masks.preprocess = 'five_v'  # 5-part vertical segmentation
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.test_use_target_segmentation = 'soft'  # Use soft masking like official
        config.model.bpbreid.testing_binary_visibility_score = False  # Use continuous scores
        config.model.bpbreid.training_binary_visibility_score = False
        
        # Build model with config
        model = torchreid.models.build_model(
            name='bpbreid',
            num_classes=751,  # Market-1501 has 751 identities
            config=config,
            pretrained=True
        )
        
        # Handle DataParallel state dict (remove 'module.' prefix)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def setup_transforms(self):
        """Setup image transforms matching official pipeline"""
        self.transform = transforms.Compose([
            transforms.Resize((384, 128)),  # Official BPBreID input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def setup_mask_transforms(self):
        """Setup mask transforms for official processing"""
        try:
            # Initialize the official mask transforms
            self.five_v_transform = CombinePifPafIntoFiveVerticalParts()
            self.background_mask_transform = AddBackgroundMask(
                background_computation_strategy='threshold',
                mask_filtering_threshold=0.5
            )
            self.mask_transforms_available = True
            print("Mask transforms initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize mask transforms: {e}")
            self.mask_transforms_available = False
    
    def detect_persons_yolo(self, frame: np.ndarray):
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
    
    def create_official_mask(self, frame: np.ndarray, bbox):
        """Create mask using the official MaskRCNN method"""
        if not self.mask_available:
            return self.create_simple_mask(frame, bbox)
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop the person region
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return self.create_simple_mask(frame, bbox)
            
            # Prepare data for BatchMask processing
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            batch_data = [{
                "image": torch.as_tensor(np.array(pil_image).transpose(2, 0, 1).astype("float32")),
                "height": person_crop.shape[0],
                "width": person_crop.shape[1]
            }]
            
            # Run MaskRCNN inference using the official method
            with torch.no_grad():
                results = self.mask_processor.model(batch_data)
            
            # Extract person masks using the official filter_masks method
            if len(results) > 0 and "instances" in results[0]:
                instances = results[0]["instances"]
                pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                pred_masks = instances.pred_masks.cpu().numpy()
                pred_scores = instances.scores.cpu().numpy()
                
                # Filter for person class (class 0 in COCO) with high confidence
                person_indices = (pred_classes == 0) & (pred_scores > 0.5)
                
                if person_indices.sum() > 0:
                    # Get the best person mask
                    person_masks = pred_masks[person_indices]
                    person_scores = pred_scores[person_indices]
                    best_mask_idx = person_scores.argmax()
                    
                    # Get the mask for the cropped region
                    crop_mask = person_masks[best_mask_idx].astype(np.float32)
                    
                    # Create full-frame mask
                    full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                    
                    # Resize crop mask to match the cropped region size
                    if crop_mask.shape != person_crop.shape[:2]:
                        crop_mask = cv2.resize(crop_mask, (person_crop.shape[1], person_crop.shape[0]))
                    
                    # Place the mask in the correct position in the full frame
                    full_mask[y1:y2, x1:x2] = crop_mask
                    
                    return full_mask
            
            # Fallback to simple mask if MaskRCNN failed
            return self.create_simple_mask(frame, bbox)
            
        except Exception as e:
            print(f"MaskRCNN processing failed: {e}")
            return self.create_simple_mask(frame, bbox)
    
    def create_simple_mask(self, frame: np.ndarray, bbox):
        """Create a simple mask from bounding box (fallback method)"""
        x1, y1, x2, y2 = bbox
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        return mask
    
    def process_mask_for_bpbreid(self, mask: np.ndarray, image_size=(384, 128)):
        """Process mask to match BPBreID input format"""
        # Resize mask to match image size
        mask_resized = cv2.resize(mask, (image_size[1], image_size[0]))
        
        # Create 5-part vertical masks (matching 'five_v' preprocessing)
        h, w = mask_resized.shape
        part_height = h // 5
        
        # Create 5 vertical parts + background
        masks = np.zeros((6, h, w), dtype=np.float32)  # 6 = 5 parts + background
        
        # Background mask (inverse of person mask)
        masks[0] = 1.0 - mask_resized
        
        # 5 vertical parts
        for i in range(5):
            start_y = i * part_height
            end_y = (i + 1) * part_height if i < 4 else h
            masks[i + 1, start_y:end_y, :] = mask_resized[start_y:end_y, :]
        
        # Convert to tensor and add batch dimension
        masks_tensor = torch.from_numpy(masks).unsqueeze(0).to(self.device)
        
        return masks_tensor
    
    def extract_features(self, image, mask=None):
        """Extract features from image using BPBreID with optional mask"""
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Process mask if provided
        external_masks = None
        if mask is not None:
            external_masks = self.process_mask_for_bpbreid(mask)
        
        # Extract features
        with torch.no_grad():
            embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks = self.model(
                image_tensor, external_parts_masks=external_masks
            )
        
        # Return both foreground and parts features
        foreground_features = embeddings['bn_foreg']  # [1, 512]
        parts_features = embeddings['bn_parts']  # [1, 5, 512]
        
        return foreground_features, parts_features
    
    def compute_similarity(self, features1, features2):
        """Compute cosine similarity between feature vectors"""
        # Normalize features
        f1_norm = F.normalize(features1, p=2, dim=1)
        f2_norm = F.normalize(features2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(f1_norm, f2_norm, dim=1)
        
        return similarity.item()
    
    def load_gallery_image(self, gallery_path):
        """Load gallery image and extract features"""
        print(f"Loading gallery image: {gallery_path}")
        
        # Load image
        image = cv2.imread(gallery_path)
        if image is None:
            raise ValueError(f"Could not load image: {gallery_path}")
        
        # Create a simple full mask for gallery image (assuming it's a person image)
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        # Extract features
        foreground_features, parts_features = self.extract_features(image, mask)
        
        # Store gallery features
        self.gallery_features = {
            'foreground': foreground_features,
            'parts': parts_features
        }
        self.gallery_image_path = gallery_path
        
        print(f"Gallery features extracted successfully")
        return self.gallery_features
    
    def process_video_frame(self, frame, detections):
        """Process a single video frame and compare with gallery"""
        results = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf = detection
            
            # Crop person region
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            
            # Create mask using official method
            mask = self.create_official_mask(frame, (x1, y1, x2, y2))
            
            # Extract features
            try:
                foreground_features, parts_features = self.extract_features(person_crop, mask)
                
                # Compute similarities
                fg_similarity = self.compute_similarity(
                    self.gallery_features['foreground'], 
                    foreground_features
                )
                
                # Average parts similarity
                parts_similarity = 0
                for j in range(5):  # 5 parts
                    part_sim = self.compute_similarity(
                        self.gallery_features['parts'][0, j:j+1], 
                        parts_features[0, j:j+1]
                    )
                    parts_similarity += part_sim
                parts_similarity /= 5
                
                # Combined similarity (weighted average)
                combined_similarity = 0.6 * fg_similarity + 0.4 * parts_similarity
                
                results.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'foreground_similarity': fg_similarity,
                    'parts_similarity': parts_similarity,
                    'combined_similarity': combined_similarity,
                    'mask': mask
                })
                
            except Exception as e:
                print(f"Feature extraction failed for detection {i}: {e}")
                continue
        
        return results
    
    def visualize_results(self, frame, results):
        """Visualize detection and similarity results"""
        vis_frame = frame.copy()
        
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            conf = result['confidence']
            combined_sim = result['combined_similarity']
            
            # Color based on similarity (green = high similarity, red = low)
            if combined_sim > 0.7:
                color = (0, 255, 0)  # Green
            elif combined_sim > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add labels
            label = f"Person {i+1}: {combined_sim:.3f}"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add detection confidence
            conf_label = f"Conf: {conf:.2f}"
            cv2.putText(vis_frame, conf_label, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_frame
    
    def process_video(self, video_path, gallery_path, output_path=None, 
                     show_preview=True, save_video=True):
        """Process video and compare with gallery image"""
        
        # Load gallery image first
        self.load_gallery_image(gallery_path)
        
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
        
        # Setup video writer if saving
        if save_video:
            if output_path is None:
                output_path = video_path.replace('.MOV', '_bpbreid_results.mp4').replace('.mp4', '_bpbreid_results.mp4')
            
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
                    
                    # Process frame and compare with gallery
                    results = self.process_video_frame(frame, detections)
                    
                    # Visualize results
                    vis_frame = self.visualize_results(frame, results)
                    
                    # Add processing info
                    info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(detections)} | Gallery: {os.path.basename(gallery_path)}"
                    cv2.putText(vis_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if save_video:
                        out.write(vis_frame)
                    
                    # Show preview if requested
                    if show_preview:
                        cv2.imshow('BPBreID Official Mask Results', vis_frame)
                    
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
                    # Small delay to prevent excessive CPU usage when not showing preview
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
    """Main function to run the BPBreID official mask processor"""
    
    # Configuration
    REID_MODEL_PATH = "pretrained_models/bpbreid_market1501_hrnet32_10642.pth"
    HRNET_PATH = "pretrained_models/hrnet32_imagenet.pth"
    YOLO_MODEL = "yolov8n.pt"
    GALLERY_IMAGE = "datasets/Compare/dataset-2/person-1.jpg"
    VIDEO_PATH = "datasets/Compare/dataset-2/person-1-vid.MOV"
    
    # Check if files exist
    if not os.path.exists(REID_MODEL_PATH):
        print(f"Error: ReID model not found: {REID_MODEL_PATH}")
        return
    
    if not os.path.exists(HRNET_PATH):
        print(f"Error: HRNet model not found: {HRNET_PATH}")
        return
    
    if not os.path.exists(YOLO_MODEL):
        print(f"Error: YOLO model not found: {YOLO_MODEL}")
        return
    
    if not os.path.exists(GALLERY_IMAGE):
        print(f"Error: Gallery image not found: {GALLERY_IMAGE}")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        return
    
    try:
        # Create processor
        print("Initializing BPBreID Official Mask Processor...")
        processor = BPBreIDOfficialMaskProcessor(
            reid_model_path=REID_MODEL_PATH,
            hrnet_path=HRNET_PATH,
            yolo_model_path=YOLO_MODEL
        )
        
        # Process video
        output_path = VIDEO_PATH.replace('.MOV', '_bpbreid_official_results.mp4').replace('.mp4', '_bpbreid_official_results.mp4')
        processor.process_video(
            video_path=VIDEO_PATH,
            gallery_path=GALLERY_IMAGE,
            output_path=output_path,
            show_preview=True,
            save_video=True
        )
        
        print("\nProcessing completed successfully!")
        print(f"Results saved as: {output_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
