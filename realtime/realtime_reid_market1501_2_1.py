import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
from torchvision import transforms
from collections import defaultdict
import time
from PIL import Image
import os
import torchvision.transforms.functional as F
from scipy import ndimage
import matplotlib.pyplot as plt

# For pose estimation - you'll need to install these
try:
    import openpifpaf
    PIFPAF_AVAILABLE = True
except ImportError:
    print("Warning: OpenPifPaf not available. Install with: pip install openpifpaf")
    PIFPAF_AVAILABLE = False

class PersonReIDWithMasks:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the ReID system with YOLO detector and BPBReID model with mask support"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BPBReID model
        self.reid_model = self._load_reid_model(reid_model_path, hrnet_path)
        self.reid_model.eval()
        
        # Setup transforms for ReID
        self.transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize pose estimation if available
        self.pose_predictor = None
        if PIFPAF_AVAILABLE:
            self._setup_pose_estimation()
        
        # Gallery to store person features
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.gallery_masks = []  # Store masks for each gallery image
        self.next_person_id = 1
        
        # Tracking confidence threshold
        self.reid_threshold = 0.7
        
    def _setup_pose_estimation(self):
        """Setup OpenPifPaf for pose estimation"""
        try:
            import openpifpaf
            # Configure PifPaf
            self.pose_predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
            print("OpenPifPaf pose estimation loaded successfully")
        except Exception as e:
            print(f"Could not load OpenPifPaf: {e}")
            self.pose_predictor = None
    
    def _load_reid_model(self, model_path, hrnet_path):
        """Load the BPBReID model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        from types import SimpleNamespace
        
        # Create config structure for BPBReID with masks enabled
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.load_weights = model_path
        config.model.load_config = True
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.mask_filtering_testing = True  # Enable mask filtering
        config.model.bpbreid.mask_filtering_training = False
        config.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']  # Use both foreground and parts
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'
        config.model.bpbreid.masks.preprocess = 'five_v'  # 5-part body segmentation
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.test_use_target_segmentation = 'none'
        config.model.bpbreid.testing_binary_visibility_score = True
        config.model.bpbreid.training_binary_visibility_score = True
        
        # Build model
        model = torchreid.models.build_model(
            name='bpbreid',
            num_classes=751,
            config=config,
            pretrained=True
        )
        
        # Load state dict
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
        return model
    
    def generate_body_part_masks(self, image):
        """Generate 5-part body segmentation masks"""
        if self.pose_predictor is None:
            # Fallback: create simple vertical strip masks
            return self._create_simple_part_masks(image)
        
        try:
            # Get pose keypoints using OpenPifPaf
            predictions, gt_anns, image_meta = self.pose_predictor.pil_image(image)
            
            if len(predictions) == 0:
                return self._create_simple_part_masks(image)
            
            # Use the first (most confident) person detection
            pose = predictions[0]
            keypoints = pose.data
            
            # Create 5-part masks based on keypoints
            masks = self._create_part_masks_from_keypoints(image, keypoints)
            return masks
            
        except Exception as e:
            print(f"Pose estimation failed: {e}")
            return self._create_simple_part_masks(image)
    
    def _create_simple_part_masks(self, image):
        """Create simple 5-part vertical strip masks as fallback"""
        if isinstance(image, Image.Image):
            h, w = image.size[1], image.size[0]
        else:
            h, w = image.shape[:2]
        
        # Create 5 horizontal strips
        part_height = h // 5
        masks = []
        
        for i in range(5):
            mask = np.zeros((h, w), dtype=np.float32)
            start_y = i * part_height
            end_y = (i + 1) * part_height if i < 4 else h
            mask[start_y:end_y, :] = 1.0
            masks.append(mask)
        
        return masks
    
    def _create_part_masks_from_keypoints(self, image, keypoints):
        """Create body part masks based on pose keypoints"""
        if isinstance(image, Image.Image):
            h, w = image.size[1], image.size[0]
        else:
            h, w = image.shape[:2]
        
        masks = []
        
        # COCO keypoint indices
        # 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists
        # 11-12: hips, 13-14: knees, 15-16: ankles
        
        # Extract keypoint coordinates (x, y, confidence)
        kpts = keypoints.reshape(-1, 3)
        
        # Define body parts based on keypoints
        # Part 0: Head (nose, eyes, ears)
        # Part 1: Torso upper (shoulders to chest)
        # Part 2: Torso lower (chest to hips)
        # Part 3: Upper limbs (arms)
        # Part 4: Lower limbs (legs)
        
        try:
            # Get key landmarks
            nose = kpts[0][:2] if kpts[0][2] > 0.3 else [w//2, h//8]
            left_shoulder = kpts[5][:2] if kpts[5][2] > 0.3 else [w//4, h//4]
            right_shoulder = kpts[6][:2] if kpts[6][2] > 0.3 else [3*w//4, h//4]
            left_hip = kpts[11][:2] if kpts[11][2] > 0.3 else [w//4, 3*h//5]
            right_hip = kpts[12][:2] if kpts[12][2] > 0.3 else [3*w//4, 3*h//5]
            
            # Create masks for each part
            for part_idx in range(5):
                mask = np.zeros((h, w), dtype=np.float32)
                
                if part_idx == 0:  # Head
                    # Head region: from top to shoulder level
                    shoulder_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
                    mask[:max(shoulder_y, h//5), :] = 1.0
                
                elif part_idx == 1:  # Upper torso
                    shoulder_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
                    chest_y = shoulder_y + (h // 6)
                    mask[shoulder_y:chest_y, :] = 1.0
                
                elif part_idx == 2:  # Lower torso
                    chest_y = int((left_shoulder[1] + right_shoulder[1]) / 2) + (h // 6)
                    hip_y = int((left_hip[1] + right_hip[1]) / 2)
                    mask[chest_y:hip_y, :] = 1.0
                
                elif part_idx == 3:  # Upper limbs (arms)
                    shoulder_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
                    hip_y = int((left_hip[1] + right_hip[1]) / 2)
                    # Create arm regions (left and right sides)
                    mask[shoulder_y:hip_y, :w//4] = 1.0  # Left arm area
                    mask[shoulder_y:hip_y, 3*w//4:] = 1.0  # Right arm area
                
                else:  # Part 4: Lower limbs (legs)
                    hip_y = int((left_hip[1] + right_hip[1]) / 2)
                    mask[hip_y:, :] = 1.0
                
                masks.append(mask)
        
        except Exception as e:
            print(f"Error creating keypoint-based masks: {e}")
            return self._create_simple_part_masks(image)
        
        return masks
    
    def extract_features_with_masks(self, image):
        """Extract ReID features with body part masks"""
        # Generate body part masks
        masks = self.generate_body_part_masks(image)
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        # Apply transforms to image
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Resize masks to match transformed image size (384x128)
        resized_masks = []
        for mask in masks:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((128, 384))  # Note: PIL uses (width, height)
            mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).float()
            resized_masks.append(mask_tensor)
        
        # Stack masks: shape (num_parts, H, W)
        masks_tensor = torch.stack(resized_masks, dim=0).unsqueeze(0).to(self.device)
        
        # Extract features with masks
        with torch.no_grad():
            # Pass both image and masks to the model
            outputs = self.reid_model(image_tensor, masks_tensor)
            
            # BPBReID returns (output_dict, visibility_scores, parts_masks)
            if isinstance(outputs, tuple):
                output_dict = outputs[0]
                
                # Use both foreground and parts features
                features_list = []
                
                # Add foreground features
                if 'bn_foreg' in output_dict:
                    features_list.append(output_dict['bn_foreg'])
                
                # Add parts features
                if 'parts' in output_dict:
                    features_list.append(output_dict['parts'])
                
                # Concatenate features
                if len(features_list) > 1:
                    feature_vector = torch.cat(features_list, dim=1)
                else:
                    feature_vector = features_list[0] if features_list else output_dict[list(output_dict.keys())[0]]
                    
            else:
                feature_vector = outputs
        
        # Ensure proper shape and normalize
        if len(feature_vector.shape) > 2:
            feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
        feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
        return feature_vector, masks
        
    def preprocess_with_padding(self, image):
        """
        Pad image to 1:2 aspect ratio, then resize to 384×128
        """
        h, w = image.shape[:2]
        target_ratio = 1/2
        current_ratio = w/h
        
        if current_ratio > target_ratio:
            # Too wide, pad height
            target_h = int(w / target_ratio)
            pad_h = target_h - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, 
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
        else:
            # Too tall, pad width
            target_w = int(h * target_ratio)
            pad_w = target_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # Resize to final dimensions
        image = cv2.resize(image, (128, 384))
        return image
        
    def extract_features(self, image):
        """Extract ReID features with proper aspect ratio handling"""
        
        # Option 3: Smart padding approach
        processed_image = self.preprocess_with_padding(image)
        
        # Convert to PIL and apply remaining transforms
        if isinstance(processed_image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        # Only normalize and convert to tensor (no resize needed)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features using the model
        with torch.no_grad():
            outputs = self.reid_model(image_tensor)
            
            if isinstance(outputs, tuple):
                output_dict = outputs[0]
                if 'bn_foreg' in output_dict:
                    feature_vector = output_dict['bn_foreg']
                else:
                    feature_vector = list(output_dict.values())[0]
            else:
                feature_vector = outputs
        
        if len(feature_vector.shape) > 2:
            feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
        feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
        return feature_vector
    
    def _extract_features_simple(self, image):
        """Simple feature extraction without masks (fallback)"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.reid_model(image_tensor)
            
            if isinstance(outputs, tuple):
                output_dict = outputs[0]
                if 'bn_foreg' in output_dict:
                    feature_vector = output_dict['bn_foreg']
                else:
                    feature_vector = list(output_dict.values())[0]
            else:
                feature_vector = outputs
        
        if len(feature_vector.shape) > 2:
            feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
        feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
        return feature_vector
    
    def match_person(self, query_features):
        """Match query features against gallery"""
        if len(self.gallery_features) == 0:
            return None, 0.0
        
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        similarities = torch.mm(query_features, gallery_tensor.t())
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        if best_similarity > self.reid_threshold:
            return self.gallery_ids[best_idx], best_similarity
        else:
            return None, best_similarity
    
    def add_to_gallery(self, features, person_id, image, masks=None):
        """Add a person to the gallery"""
        self.gallery_features.append(features)
        self.gallery_ids.append(person_id)
        self.gallery_images.append(image.copy())
        self.gallery_masks.append(masks)
    
    def process_frame(self, frame):
        """Process a single frame with mask-based ReID"""
        results = self.yolo(frame, classes=0, conf=0.5)
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        # Extract features with masks
                        features = self.extract_features(person_img)
                        matched_id, similarity = self.match_person(features)
                        
                        if matched_id is not None:
                            label = f"Person {matched_id} ({similarity:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = f"New Person ({similarity:.2f})"
                            color = (0, 0, 255)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
    
    def visualize_masks(self, image, masks):
        """Visualize body part masks"""
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Show original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show each mask
        part_names = ['Head', 'Upper Torso', 'Lower Torso', 'Arms', 'Legs']
        for i, (mask, name) in enumerate(zip(masks, part_names)):
            axes[i+1].imshow(mask, cmap='gray')
            axes[i+1].set_title(f'Part {i}: {name}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def run_camera(self):
        """Run real-time ReID with mask-based processing"""
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Person Re-Identification with Masks', cv2.WINDOW_NORMAL)
        
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current detections to gallery")
        print("- Press 'c' to clear gallery")
        print("- Press 'g' to show gallery")
        print("- Press 'm' to show masks for current detections")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = self.process_frame(frame)
            
            # Add status
            status_text = f"Gallery: {len(self.gallery_ids)} persons | Masks: {'ON' if PIFPAF_AVAILABLE else 'Simple'}"
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Person Re-Identification with Masks', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_current_detections(frame)
            elif key == ord('c'):
                self.gallery_features = []
                self.gallery_ids = []
                self.gallery_images = []
                self.gallery_masks = []
                print("Gallery cleared!")
            elif key == ord('g'):
                self._show_gallery()
            elif key == ord('m'):
                self._show_current_masks(frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _save_current_detections(self, frame):
        """Save current detections to gallery with masks"""
        results = self.yolo(frame, classes=0, conf=0.5)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        try:
                            features, masks = self.extract_features_with_masks(person_img)
                        except:
                            features = self.extract_features(person_img)
                            masks = None
                        
                        matched_id, similarity = self.match_person(features)
                        
                        if matched_id is None:
                            self.add_to_gallery(features, self.next_person_id, person_img, masks)
                            print(f"Added Person {self.next_person_id} to gallery (with masks)")
                            self.next_person_id += 1
    
    def _show_current_masks(self, frame):
        """Show masks for current frame detections"""
        results = self.yolo(frame, classes=0, conf=0.5)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        masks = self.generate_body_part_masks(person_img)
                        self.visualize_masks(person_img, masks)
                        break  # Show only first detection
    
    def _show_gallery(self):
        """Display gallery with mask information"""
        if len(self.gallery_images) == 0:
            print("Gallery is empty!")
            return
        
        gallery_height = 200
        gallery_width = 150
        cols = min(6, len(self.gallery_images))
        rows = (len(self.gallery_images) + cols - 1) // cols
        
        gallery_viz = np.zeros((rows * gallery_height, cols * gallery_width, 3), dtype=np.uint8)
        
        for idx, (img, person_id) in enumerate(zip(self.gallery_images, self.gallery_ids)):
            row = idx // cols
            col = idx % cols
            
            resized = cv2.resize(img, (gallery_width, gallery_height))
            
            y1 = row * gallery_height
            y2 = (row + 1) * gallery_height
            x1 = col * gallery_width
            x2 = (col + 1) * gallery_width
            gallery_viz[y1:y2, x1:x2] = resized
            
            # Add label with mask info
            mask_info = "w/ masks" if self.gallery_masks[idx] is not None else "no masks"
            cv2.putText(gallery_viz, f"ID {person_id}", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(gallery_viz, mask_info, (x1 + 5, y1 + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        cv2.imshow('Gallery', gallery_viz)
        cv2.waitKey(0)
        cv2.destroyWindow('Gallery')


def main():
    # Check if required packages are available
    if not PIFPAF_AVAILABLE:
        print("Warning: OpenPifPaf not available. Using simple mask fallback.")
        print("For full body part detection, install with:")
        print("pip install openpifpaf")
        print()
    
    # Paths to your models
    reid_model_path = "pretrained_models/bpbreid_market1501_hrnet32_10642.pth"
    hrnet_path = "pretrained_models/hrnetv2_w32_imagenet_pretrained.pth"
    
    # Create enhanced ReID system
    reid_system = PersonReIDWithMasks(reid_model_path, hrnet_path)
    
    # Run camera
    reid_system.run_camera()


if __name__ == "__main__":
    main()