#!/usr/bin/env python3
"""
YOLO-based Video Processor with OpenPifPaf Masking

This script processes video to:
1. Detect persons using YOLO
2. Generate pose-based masks using OpenPifPaf (similar to official get_labels.py approach)
3. Show the masking visualization

Uses available dependencies: OpenCV, Ultralytics YOLO, OpenPifPaf, PyTorch
Based on the official masking concepts from torchreid/scripts/get_labels.py
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
import openpifpaf

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

class YOLOPifPafMaskProcessor:
    """
    Video processor that uses YOLO for person detection and OpenPifPaf 
    for pose-based masking (similar to the official BPBreID approach)
    """
    
    def __init__(self, yolo_model_path='yolov8n.pt', pifpaf_model='shufflenetv2k16'):
        """
        Initialize the processor with YOLO and OpenPifPaf models
        
        Args:
            yolo_model_path: Path to YOLO model weights
            pifpaf_model: OpenPifPaf model name
        """
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO for person detection
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize OpenPifPaf for pose estimation (similar to get_labels.py approach)
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
    
    def create_pifpaf_mask(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Create pose-based confidence fields using OpenPifPaf (following get_labels.py approach)
        
        Args:
            frame: Input frame
            bbox: YOLO bounding box for cropping
            
        Returns:
            Tuple of (combined_mask, individual_part_masks) where:
            - combined_mask: Single mask for visualization
            - individual_part_masks: List of 5 body part confidence fields
        """
        if not self.pifpaf_available:
            simple_mask = self.create_yolo_mask(frame, bbox)
            return simple_mask, [simple_mask]
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Crop the person region with some padding
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
                # No pose detected, use YOLO mask
                simple_mask = self.create_yolo_mask(frame, bbox)
                return simple_mask, [simple_mask]
            
            # Use the first (most confident) pose detection
            pose = predictions[0]
            keypoints = pose.data  # Shape: [17, 3] - (x, y, confidence) for each keypoint
            
            # Create 5-part confidence fields following the BPBreID approach
            part_masks, combined_mask = self._create_confidence_fields(person_crop, keypoints)
            
            # Create full-frame masks
            full_part_masks = []
            full_combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            
            for part_mask in part_masks:
                full_part_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                if part_mask.shape == person_crop.shape[:2]:
                    full_part_mask[crop_y1:crop_y2, crop_x1:crop_x2] = part_mask
                else:
                    # Resize mask to match crop size
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
        """
        Create 5-part confidence fields from keypoints (following BPBreID official approach)
        
        Args:
            image: Cropped person image
            keypoints: OpenPifPaf keypoints [17, 3] (x, y, confidence)
            
        Returns:
            Tuple of (part_masks_list, combined_mask) where:
            - part_masks_list: List of 5 confidence field arrays for each body part
            - combined_mask: Combined mask for general visualization
        """
        h, w = image.shape[:2]
        
        # Define 5-part body segmentation following BPBreID's five_v approach
        part_definitions = {
            'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
            'upper_torso_arms': [5, 6, 7, 8],  # shoulders, elbows
            'lower_torso_arms': [9, 10, 11, 12],  # wrists, hips
            'legs': [13, 14],  # knees
            'feet': [15, 16]  # ankles
        }
        
        part_names = list(part_definitions.keys())
        part_masks = []
        
        # Create confidence field for each body part
        for part_name, keypoint_indices in part_definitions.items():
            part_mask = np.zeros((h, w), dtype=np.float32)
            
            # Get valid keypoints for this part
            valid_kps = []
            for idx in keypoint_indices:
                if idx < len(keypoints) and keypoints[idx, 2] > 0.2:  # Lower confidence threshold
                    valid_kps.append(keypoints[idx])
            
            if len(valid_kps) == 0:
                part_masks.append(part_mask)
                continue
            
            # Create large, continuous body part regions instead of keypoint-based masks
            if part_name == 'head':
                # Create large circular head region
                if len(valid_kps) >= 1:
                    center_x = int(sum(kp[0] for kp in valid_kps) / len(valid_kps))
                    center_y = int(sum(kp[1] for kp in valid_kps) / len(valid_kps))
                    
                    # Large radius for head coverage
                    radius = min(h, w) // 6  # Much larger radius
                    
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    part_mask = np.exp(-dist_sq / (2 * radius ** 2))
                    
            elif part_name in ['upper_torso_arms', 'lower_torso_arms']:
                # Create large rectangular torso/arms regions
                if len(valid_kps) >= 2:
                    x_coords = [kp[0] for kp in valid_kps]
                    y_coords = [kp[1] for kp in valid_kps]
                    
                    # Create large bounding box with padding
                    min_x = max(0, int(min(x_coords)) - w // 8)
                    max_x = min(w, int(max(x_coords)) + w // 8)
                    min_y = max(0, int(min(y_coords)) - h // 10)
                    max_y = min(h, int(max(y_coords)) + h // 10)
                    
                    # Fill the entire region with high confidence
                    part_mask[min_y:max_y, min_x:max_x] = 0.8
                    
                    # Add smooth falloff from center
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    gaussian = np.exp(-dist_sq / (2 * (min(h, w) // 4) ** 2))
                    part_mask = np.maximum(part_mask, gaussian * 0.6)
                    
            elif part_name == 'legs':
                # Create large vertical leg regions
                if len(valid_kps) >= 1:
                    x_coords = [kp[0] for kp in valid_kps]
                    y_coords = [kp[1] for kp in valid_kps]
                    
                    # Create wide vertical regions for legs
                    min_x = max(0, int(min(x_coords)) - w // 12)
                    max_x = min(w, int(max(x_coords)) + w // 12)
                    min_y = max(0, int(min(y_coords)) - h // 15)
                    max_y = min(h, int(max(y_coords)) + h // 15)
                    
                    # Fill the entire region
                    part_mask[min_y:max_y, min_x:max_x] = 0.7
                    
                    # Add smooth falloff
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    gaussian = np.exp(-dist_sq / (2 * (min(h, w) // 5) ** 2))
                    part_mask = np.maximum(part_mask, gaussian * 0.5)
                    
            elif part_name == 'feet':
                # Create medium circular foot regions
                if len(valid_kps) >= 1:
                    center_x = int(sum(kp[0] for kp in valid_kps) / len(valid_kps))
                    center_y = int(sum(kp[1] for kp in valid_kps) / len(valid_kps))
                    
                    # Medium radius for feet
                    radius = min(h, w) // 8
                    
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    part_mask = np.exp(-dist_sq / (2 * radius ** 2))
            
            # Apply strong Gaussian blur to create smooth, large areas
            part_mask = cv2.GaussianBlur(part_mask, (31, 31), 0)
            
            # Ensure minimum confidence for visibility
            part_mask = np.maximum(part_mask, 0.1)
            
            part_masks.append(part_mask)
        
        # Create combined mask by taking maximum across all parts
        combined_mask = np.zeros((h, w), dtype=np.float32)
        for part_mask in part_masks:
            combined_mask = np.maximum(combined_mask, part_mask)
        
        return part_masks, combined_mask
    
    def _connect_keypoints_in_part(self, part_mask: np.ndarray, keypoints_in_part: List, part_name: str, h: int, w: int) -> np.ndarray:
        """
        Connect keypoints within a body part to create continuous confidence regions
        
        Args:
            part_mask: Current part mask
            keypoints_in_part: List of keypoints for this part
            part_name: Name of the body part
            h, w: Image dimensions
            
        Returns:
            Updated part mask with connections
        """
        # Define connections within each part
        part_connections = {
            'head': [(0, 1), (0, 2), (1, 3), (2, 4)],  # Connect facial features
            'upper_torso_arms': [(0, 1), (0, 2), (1, 3)],  # Connect shoulders to elbows
            'lower_torso_arms': [(0, 1), (2, 3)],  # Connect wrists, connect hips
            'legs': [(0, 1)],  # Connect knees
            'feet': [(0, 1)]  # Connect ankles
        }
        
        if part_name not in part_connections:
            return part_mask
        
        connections = part_connections[part_name]
        
        # Draw connections between keypoints
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints_in_part) and end_idx < len(keypoints_in_part):
                start_kp = keypoints_in_part[start_idx]
                end_kp = keypoints_in_part[end_idx]
                
                start_pt = (int(start_kp[0]), int(start_kp[1]))
                end_pt = (int(end_kp[0]), int(end_kp[1]))
                
                # Calculate line thickness based on part type and image size
                if part_name in ['upper_torso_arms', 'lower_torso_arms']:
                    thickness = max(15, int(min(h, w) * 0.08))  # Thick for torso
                else:
                    thickness = max(10, int(min(h, w) * 0.05))  # Thinner for other parts
                
                # Use average confidence of connected keypoints
                avg_conf = (start_kp[2] + end_kp[2]) / 2
                
                # Create a temporary mask for the line
                line_mask = np.zeros((h, w), dtype=np.float32)
                cv2.line(line_mask, start_pt, end_pt, avg_conf, thickness)
                
                # Add to part mask
                part_mask = np.maximum(part_mask, line_mask)
        
        return part_mask
    
    def visualize_mask(self, frame: np.ndarray, combined_mask: np.ndarray, part_masks: List[np.ndarray],
                      bbox: Tuple[int, int, int, int, float], person_id: int) -> np.ndarray:
        """
        Visualize the confidence fields on the frame with different colors for each body part
        
        Args:
            frame: Original frame
            combined_mask: Combined person mask
            part_masks: List of individual body part masks
            bbox: Bounding box with confidence
            person_id: Person identifier
            
        Returns:
            Annotated frame with multi-colored mask visualization
        """
        # Create visualization
        vis_frame = frame.copy()
        
        # Define colors for each body part (BGR format) - More vibrant colors
        part_colors = [
            (255, 0, 0),      # Head - Pure Blue
            (0, 255, 0),      # Upper Torso/Arms - Pure Green
            (0, 0, 255),      # Lower Torso/Arms - Pure Red
            (255, 255, 0),    # Legs - Cyan
            (255, 0, 255),    # Feet - Magenta
        ]
        
        part_names = ['Head', 'Upper Torso/Arms', 'Lower Torso/Arms', 'Legs', 'Feet']
        
        # Create colored overlay for each body part
        mask_colored = np.zeros_like(frame, dtype=np.uint8)
        
        # Check if we have any visible masks
        has_visible_masks = any(part_mask.max() > 0.01 for part_mask in part_masks)
        
        # Simple, highly visible mask application
        if has_visible_masks and len(part_masks) > 1:
            # Apply each part mask with solid bright colors
            for i, (part_mask, color, name) in enumerate(zip(part_masks, part_colors, part_names)):
                if part_mask.max() > 0.1:  # Process masks with reasonable confidence
                    # Create binary mask with lower threshold for better coverage
                    binary_mask = (part_mask > 0.2).astype(np.uint8)
                    # Apply solid color where mask is present
                    mask_colored[binary_mask > 0] = color
        else:
            # Fallback: use combined mask with bright green
            if combined_mask.max() > 0.1:
                binary_combined = (combined_mask > 0.2).astype(np.uint8)
                mask_colored[binary_combined > 0] = (0, 255, 0)  # Bright green
            else:
                # Last resort: rectangular mask for visibility testing
                x1, y1, x2, y2, conf = bbox
                cv2.rectangle(mask_colored, (x1, y1), (x2, y2), (0, 255, 0), -1)
        
        # Strong blend for maximum visibility
        alpha = 0.7  # High alpha for strong mask visibility
        vis_frame = cv2.addWeighted(vis_frame, 1-alpha, mask_colored, alpha, 0)
        
        # Draw bounding box
        x1, y1, x2, y2, conf = bbox
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with person info
        label = f"Person {person_id} ({conf:.2f})"
        cv2.putText(vis_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add legend for body parts (if we have multiple parts)
        if len(part_masks) > 1:
            legend_y_start = 60
            for i, (color, name) in enumerate(zip(part_colors, part_names)):
                y_pos = legend_y_start + i * 25
                # Draw color box
                cv2.rectangle(vis_frame, (10, y_pos - 15), (30, y_pos - 5), color, -1)
                # Draw text
                cv2.putText(vis_frame, name, (35, y_pos - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_preview: bool = True, save_video: bool = True):
        """
        Process video with YOLO detection and pose-based masking
        
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
                output_path = video_path.replace('.MOV', '_pose_masked.mp4').replace('.mp4', '_pose_masked.mp4')
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame")
        
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
                    mask_type = "None"
                    
                    for i, detection in enumerate(detections):
                        bbox = detection[:4]  # (x1, y1, x2, y2)
                        
                        # Create mask using pose-based method
                        if self.pifpaf_available:
                            combined_mask, part_masks = self.create_pifpaf_mask(frame, bbox)
                            mask_type = "PifPaf Confidence Fields"
                        else:
                            combined_mask = self.create_yolo_mask(frame, bbox)
                            part_masks = [combined_mask]
                            mask_type = "YOLO Box"
                        
                        # Visualize mask
                        vis_frame = self.visualize_mask(vis_frame, combined_mask, part_masks, detection, i+1)
                    
                    # Add processing info
                    info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(detections)} | Mask: {mask_type}"
                    cv2.putText(vis_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if save_video:
                        out.write(vis_frame)
                    
                    # Show preview if requested
                    if show_preview:
                        cv2.imshow('YOLO + PifPaf Person Masking', vis_frame)
                    
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
                    elif key == ord('s'):
                        # Save current frame
                        save_path = f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(save_path, vis_frame)
                        print(f"Saved frame to {save_path}")
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

    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))              
    bpbreid_dir = os.path.dirname(current_dir)
    
    # VIDEO_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1-vid.MOV")
    VIDEO_PATH = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "correct.MOV")

    # VIDEO_PATH = "datasets/Compare/dataset-2/person-1-vid.MOV"
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
        print("Initializing YOLO + PifPaf Person Mask Processor...")
        processor = YOLOPifPafMaskProcessor(yolo_model_path=YOLO_MODEL)
        
        # Get just the filename without path
        video_filename = os.path.basename(VIDEO_PATH)
        # Create output path in current directory
        output_path = video_filename.replace('.MOV', '_yolo_pifpaf_masked.mp4').replace('.mp4', '_yolo_pifpaf_masked.mp4')

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
