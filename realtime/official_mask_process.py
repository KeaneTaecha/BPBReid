#!/usr/bin/env python3
"""
YOLO-based Video Processor with Official BPBreID Masking

This script processes video to:
1. Detect persons using YOLO
2. Generate masks using the official masking method from get_labels.py
3. Show the masking visualization

Based on the official masking implementation in torchreid/scripts/get_labels.py
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
from typing import List, Tuple

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the official masking classes
from torchreid.scripts.get_labels import BatchMask, build_config_maskrcnn
from detectron2.config import CfgNode

class YOLOPersonMaskProcessor:
    """
    Video processor that uses YOLO for person detection and applies
    the official BPBreID masking method from get_labels.py
    """
    
    def __init__(self, yolo_model_path='yolov8n.pt', 
                 maskrcnn_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        """
        Initialize the processor with YOLO and MaskRCNN models
        
        Args:
            yolo_model_path: Path to YOLO model weights
            maskrcnn_config: MaskRCNN configuration name for Detectron2
        """
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO for person detection
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize the official BatchMask from get_labels.py
        print("Loading MaskRCNN model (this may take a while)...")
        try:
            self.mask_processor = BatchMask(cfg=maskrcnn_config, batch_size=1, workers=0)
            self.mask_available = True
            print("MaskRCNN model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load MaskRCNN model: {e}")
            print("Will use YOLO bounding boxes as masks instead")
            self.mask_available = False
    
    def detect_persons_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in frame using YOLO
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of (x1, y1, x2, y2, confidence) for detected persons
        """
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
        """
        Create a simple mask from YOLO bounding box (fallback method)
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Binary mask as numpy array
        """
        x1, y1, x2, y2 = bbox
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        return mask
    
    def create_maskrcnn_mask(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create detailed person mask using the official MaskRCNN method from get_labels.py
        
        Args:
            frame: Input frame
            bbox: YOLO bounding box for cropping
            
        Returns:
            Detailed person mask as numpy array
        """
        if not self.mask_available:
            return self.create_yolo_mask(frame, bbox)
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop the person region
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return self.create_yolo_mask(frame, bbox)
            
            # Prepare data for BatchMask processing
            # Convert to PIL Image format expected by the mask processor
            pil_image = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            
            # Create a temporary batch with the cropped image
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
            
            # Fallback to YOLO mask if MaskRCNN failed
            return self.create_yolo_mask(frame, bbox)
            
        except Exception as e:
            print(f"MaskRCNN processing failed: {e}")
            return self.create_yolo_mask(frame, bbox)
    
    def visualize_mask(self, frame: np.ndarray, mask: np.ndarray, 
                      bbox: Tuple[int, int, int, int], person_id: int) -> np.ndarray:
        """
        Visualize the mask on the frame
        
        Args:
            frame: Original frame
            mask: Person mask
            bbox: Bounding box
            person_id: Person identifier
            
        Returns:
            Annotated frame with mask visualization
        """
        # Create visualization
        vis_frame = frame.copy()
        
        # Apply mask overlay
        mask_colored = np.zeros_like(frame, dtype=np.uint8)
        mask_colored[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel
        
        # Blend with original frame
        alpha = 0.4
        vis_frame = cv2.addWeighted(vis_frame, 1-alpha, mask_colored, alpha, 0)
        
        # Draw bounding box
        x1, y1, x2, y2, conf = bbox if len(bbox) == 5 else (*bbox, 0.0)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Person {person_id} ({conf:.2f})" if conf > 0 else f"Person {person_id}"
        cv2.putText(vis_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_preview: bool = True, save_video: bool = True):
        """
        Process video with YOLO detection and official masking
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            show_preview: Whether to show real-time preview
            save_video: Whether to save processed video
        """
        
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
                output_path = video_path.replace('.MOV', '_masked.mp4').replace('.mp4', '_masked.mp4')
            
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
                        
                        # Create mask using official method
                        if self.mask_available:
                            mask = self.create_maskrcnn_mask(frame, bbox)
                            mask_type = "MaskRCNN"
                        else:
                            mask = self.create_yolo_mask(frame, bbox)
                            mask_type = "YOLO Box"
                        
                        # Visualize mask
                        vis_frame = self.visualize_mask(vis_frame, mask, detection, i+1)
                    
                    # Add processing info
                    info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(detections)} | Mask: {mask_type if 'mask_type' in locals() else 'None'}"
                    cv2.putText(vis_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if save_video:
                        out.write(vis_frame)
                    
                    # Show preview if requested
                    if show_preview:
                        cv2.imshow('YOLO Person Masking', vis_frame)
                    
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
    """Main function to run the video processor"""
    
    # Configuration
    VIDEO_PATH = "datasets/Compare/dataset-2/person-1-vid.MOV"
    YOLO_MODEL = "yolov8n.pt"  # Use the existing YOLO model
    
    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        print("Please make sure the video file exists in the specified path.")
        return
    
    # Check if YOLO model exists
    if not os.path.exists(YOLO_MODEL):
        print(f"Error: YOLO model not found: {YOLO_MODEL}")
        print("Please make sure the YOLO model file exists.")
        return
    
    try:
        # Create processor
        print("Initializing YOLO Person Mask Processor...")
        processor = YOLOPersonMaskProcessor(yolo_model_path=YOLO_MODEL)
        
        # Process video
        output_path = VIDEO_PATH.replace('.MOV', '_yolo_masked.mp4')
        processor.process_video(
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