#!/usr/bin/env python3
"""
YOLO-based Video Processor with OpenPifPaf Masking - Improved Version

This script processes video to:
1. Detect persons using YOLO
2. Generate pose-based masks using OpenPifPaf with smaller keypoints and rotated leg/foot rectangles
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
import math

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

class YOLOPifPafMaskProcessor:
    """
    Video processor that uses YOLO for person detection and OpenPifPaf 
    for pose-based masking with improved leg/foot rotation
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
        Create pose-based confidence fields using OpenPifPaf with improved leg/foot rotation
        
        Args:
            frame: Input frame
            bbox: YOLO bounding box for cropping
            
        Returns:
            Tuple of (combined_mask, individual_part_masks)
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
            
            # Create confidence fields with improved leg/foot rotation
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
    
    def _draw_rotated_rectangle(self, mask: np.ndarray, center: Tuple[int, int], 
                               width: int, height: int, angle: float, confidence: float) -> np.ndarray:
        """
        Draw a rotated rectangle on the mask
        
        Args:
            mask: Mask to draw on
            center: Center point (x, y)
            width: Rectangle width
            height: Rectangle height
            angle: Rotation angle in radians
            confidence: Confidence value to fill with
            
        Returns:
            Updated mask
        """
        h, w = mask.shape
        center_x, center_y = center
        
        # Create rotation matrix
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        # Calculate rotated rectangle corners
        half_w = width / 2
        half_h = height / 2
        
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for corner_x, corner_y in corners:
            rot_x = corner_x * cos_angle - corner_y * sin_angle + center_x
            rot_y = corner_x * sin_angle + corner_y * cos_angle + center_y
            rotated_corners.append((int(rot_x), int(rot_y)))
        
        # Draw filled polygon
        pts = np.array(rotated_corners, np.int32)
        cv2.fillPoly(mask, [pts], confidence)
        
        return mask
    
    def _create_confidence_fields(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Create 7-part confidence fields with smaller keypoints and rotated leg/foot rectangles
        
        Args:
            image: Cropped person image
            keypoints: OpenPifPaf keypoints [17, 3] (x, y, confidence)
            
        Returns:
            Tuple of (part_masks_list, combined_mask)
        """
        h, w = image.shape[:2]
        
        # Define 7-part body segmentation
        part_definitions = {
            'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
            'upper_torso_arms': [5, 6, 7, 8],  # shoulders, elbows
            'lower_torso_arms': [9, 10, 11, 12],  # wrists, hips
            'left_leg': [11, 13],  # left hip, left knee
            'right_leg': [12, 14],  # right hip, right knee
            'left_foot': [13, 15],  # left knee, left ankle
            'right_foot': [14, 16]  # right knee, right ankle
        }
        
        part_names = list(part_definitions.keys())
        part_masks = []
        
        # Create confidence field for each body part
        for part_name, keypoint_indices in part_definitions.items():
            part_mask = np.zeros((h, w), dtype=np.float32)
            
            # Get valid keypoints for this part
            valid_kps = []
            for idx in keypoint_indices:
                if idx < len(keypoints) and keypoints[idx, 2] > 0.15:  # Lower confidence threshold
                    valid_kps.append(keypoints[idx])
            
            if len(valid_kps) == 0:
                part_masks.append(part_mask)
                continue
            
            # Create body part regions with smaller sizes
            if part_name == 'head':
                # Create smaller circular head region
                if len(valid_kps) >= 1:
                    center_x = int(sum(kp[0] for kp in valid_kps) / len(valid_kps))
                    center_y = int(sum(kp[1] for kp in valid_kps) / len(valid_kps))
                    
                    # Smaller radius for head coverage
                    radius = min(h, w) // 10  # Reduced from //6 to //10
                    
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    part_mask = np.exp(-dist_sq / (2 * radius ** 2))
                    
            elif part_name in ['upper_torso_arms', 'lower_torso_arms']:
                # Create smaller rectangular torso/arms regions
                if len(valid_kps) >= 2:
                    x_coords = [kp[0] for kp in valid_kps]
                    y_coords = [kp[1] for kp in valid_kps]
                    
                    # Create smaller bounding box with less padding
                    min_x = max(0, int(min(x_coords)) - w // 12)  # Reduced from //8 to //12
                    max_x = min(w, int(max(x_coords)) + w // 12)
                    min_y = max(0, int(min(y_coords)) - h // 15)  # Reduced from //10 to //15
                    max_y = min(h, int(max(y_coords)) + h // 15)
                    
                    # Fill the region with confidence
                    part_mask[min_y:max_y, min_x:max_x] = 0.7  # Reduced from 0.8 to 0.7
                    
                    # Add smaller smooth falloff from center
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
                    gaussian = np.exp(-dist_sq / (2 * (min(h, w) // 6) ** 2))  # Reduced from //4 to //6
                    part_mask = np.maximum(part_mask, gaussian * 0.5)  # Reduced from 0.6 to 0.5
                    
            elif part_name in ['left_leg', 'right_leg']:
                # Create rotated leg rectangles aligned with leg direction
                if len(valid_kps) >= 2:
                    # Get hip and knee positions
                    hip_x, hip_y = int(valid_kps[0][0]), int(valid_kps[0][1])
                    knee_x, knee_y = int(valid_kps[1][0]), int(valid_kps[1][1])
                    
                    # Calculate leg angle (fixed: remove the extra π/2 rotation)
                    dx = knee_x - hip_x
                    dy = knee_y - hip_y
                    angle = math.atan2(dy, dx)  # This gives the correct leg direction
                    
                    # Calculate leg center and dimensions
                    center_x = (hip_x + knee_x) // 2
                    center_y = (hip_y + knee_y) // 2
                    leg_length = int(math.sqrt(dx*dx + dy*dy)) + h // 8  # Add some padding
                    leg_width = w // 6  # Increased from //10 to //6 for much wider legs
                    
                    # Draw rotated rectangle for the leg
                    part_mask = self._draw_rotated_rectangle(
                        part_mask, (center_x, center_y), 
                        leg_width, leg_length, angle, 0.6
                    )
                    
                elif len(valid_kps) == 1:
                    # Only one keypoint (hip or knee), create wider vertical region
                    kp_x, kp_y = int(valid_kps[0][0]), int(valid_kps[0][1])
                    leg_width = w // 6  # Increased width to match rotated legs
                    leg_height = h // 4  # Height
                    
                    min_x = max(0, kp_x - leg_width)
                    max_x = min(w, kp_x + leg_width)
                    min_y = max(0, kp_y - leg_height // 4)
                    max_y = min(h, kp_y + leg_height)
                    
                    part_mask[min_y:max_y, min_x:max_x] = 0.5
                        
            elif part_name in ['left_foot', 'right_foot']:
                # Create rotated foot rectangles aligned with lower leg direction
                if len(valid_kps) >= 2:
                    # Get knee and ankle positions
                    knee_x, knee_y = int(valid_kps[0][0]), int(valid_kps[0][1])
                    ankle_x, ankle_y = int(valid_kps[1][0]), int(valid_kps[1][1])
                    
                    # Calculate lower leg angle (fixed: remove the extra π/2 rotation)
                    dx = ankle_x - knee_x
                    dy = ankle_y - knee_y
                    angle = math.atan2(dy, dx)  # This gives the correct leg direction
                    
                    # Calculate foot center and dimensions
                    # Position foot slightly beyond ankle in the direction of leg
                    foot_offset = h // 20  # Small offset beyond ankle
                    foot_center_x = ankle_x + int(foot_offset * math.cos(angle))
                    foot_center_y = ankle_y + int(foot_offset * math.sin(angle))
                    
                    foot_width = w // 12  # Increased from //15 to //12 for wider feet
                    foot_height = h // 10  # Increased from //12 to //10 for longer feet
                    
                    # Draw rotated rectangle for the foot
                    part_mask = self._draw_rotated_rectangle(
                        part_mask, (foot_center_x, foot_center_y), 
                        foot_width, foot_height, angle, 0.5
                    )
                    
                elif len(valid_kps) == 1:
                    # Only ankle available, create wider horizontal foot region
                    ankle_x, ankle_y = int(valid_kps[0][0]), int(valid_kps[0][1])
                    
                    foot_width = w // 12  # Increased width to match rotated feet
                    foot_height = h // 10  # Increased height
                    
                    min_x = max(0, ankle_x - foot_width)
                    max_x = min(w, ankle_x + foot_width)
                    min_y = max(0, ankle_y - foot_height // 3)
                    max_y = min(h, ankle_y + foot_height)
                    
                    part_mask[min_y:max_y, min_x:max_x] = 0.4
            
            # Apply smaller Gaussian blur for tighter regions
            part_mask = cv2.GaussianBlur(part_mask, (15, 15), 0)  # Reduced from (21, 21)
            
            # Ensure minimum confidence for visibility
            part_mask = np.maximum(part_mask, 0.02)  # Reduced from 0.05
            
            part_masks.append(part_mask)
        
        # Create combined mask by taking maximum across all parts
        combined_mask = np.zeros((h, w), dtype=np.float32)
        for part_mask in part_masks:
            combined_mask = np.maximum(combined_mask, part_mask)
        
        return part_masks, combined_mask
    
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
        
        # Define colors for each body part (BGR format)
        part_colors = [
            (255, 0, 0),      # Head - Pure Blue
            (0, 255, 0),      # Upper Torso/Arms - Pure Green
            (0, 0, 255),      # Lower Torso/Arms - Pure Red
            (255, 255, 0),    # Left Leg - Cyan
            (255, 0, 255),    # Right Leg - Magenta
            (0, 255, 255),    # Left Foot - Yellow
            (128, 0, 128),    # Right Foot - Purple
        ]
        
        part_names = ['Head', 'Upper Torso/Arms', 'Lower Torso/Arms', 'Left Leg', 'Right Leg', 'Left Foot', 'Right Foot']
        
        # Create colored overlay for each body part
        mask_colored = np.zeros_like(frame, dtype=np.uint8)
        
        # Check if we have any visible masks
        has_visible_masks = any(part_mask.max() > 0.01 for part_mask in part_masks)
        
        # Apply masks with bright colors
        if has_visible_masks and len(part_masks) > 1:
            # Apply each part mask with solid bright colors
            for i, (part_mask, color, name) in enumerate(zip(part_masks, part_colors, part_names)):
                if part_mask.max() > 0.05:  # Lower threshold for better coverage
                    # Create binary mask with lower threshold for better coverage
                    binary_mask = (part_mask > 0.1).astype(np.uint8)  # Lower threshold
                    # Apply solid color where mask is present
                    mask_colored[binary_mask > 0] = color
        else:
            # Fallback: use combined mask with bright green
            if combined_mask.max() > 0.05:
                binary_combined = (combined_mask > 0.1).astype(np.uint8)
                mask_colored[binary_combined > 0] = (0, 255, 0)  # Bright green
            else:
                # Last resort: rectangular mask for visibility testing
                x1, y1, x2, y2, conf = bbox
                cv2.rectangle(mask_colored, (x1, y1), (x2, y2), (0, 255, 0), -1)
        
        # Strong blend for maximum visibility
        alpha = 0.6  # Slightly reduced alpha for better balance
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
                output_path = video_path.replace('.MOV', '_pose_masked_improved.mp4').replace('.mp4', '_pose_masked_improved.mp4')
            
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
                            mask_type = "PifPaf Improved (Rotated Legs)"
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
                        cv2.imshow('YOLO + PifPaf Person Masking (Improved)', vis_frame)
                    
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
        print("Initializing YOLO + PifPaf Person Mask Processor (Improved)...")
        processor = YOLOPifPafMaskProcessor(yolo_model_path=YOLO_MODEL)
        
        # Get just the filename without path
        video_filename = os.path.basename(VIDEO_PATH)
        # Create output path in current directory
        output_path = video_filename.replace('.MOV', '_yolo_pifpaf_improved.mp4').replace('.mp4', '_yolo_pifpaf_improved.mp4')

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