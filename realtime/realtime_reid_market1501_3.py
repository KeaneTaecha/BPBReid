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

# For segmentation - using torchvision's built-in models (works on macOS)
try:
    import torchvision.models.segmentation as segmentation_models
    SEGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: Torchvision segmentation not available.")
    SEGMENTATION_AVAILABLE = False

class PersonReIDWithSegmentation:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the ReID system with YOLO detector and BPBReID model with segmentation support"""
        
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
        
        # Initialize segmentation model if available
        self.seg_model = None
        self.seg_transform = None
        if SEGMENTATION_AVAILABLE:
            self._setup_segmentation()
        
        # Gallery to store person features
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.gallery_masks = []  # Store masks for each gallery image
        self.next_person_id = 1
        
        # Tracking confidence threshold
        self.reid_threshold = 0.7
        
    def _setup_segmentation(self):
        """Setup DeepLabV3 segmentation model (works on macOS)"""
        try:
            # Use torchvision's DeepLabV3 with ResNet50 backbone
            # This model is pre-trained on COCO and includes person segmentation
            self.seg_model = segmentation_models.deeplabv3_resnet50(pretrained=True)
            self.seg_model.eval()
            self.seg_model = self.seg_model.to(self.device)
            
            # Setup preprocessing transforms for segmentation
            self.seg_transform = transforms.Compose([
                transforms.Resize((512, 512)),  # DeepLabV3 works well with 512x512
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("DeepLabV3 segmentation model loaded successfully")
        except Exception as e:
            print(f"Could not load segmentation model: {e}")
            self.seg_model = None
    
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
    
    def generate_person_mask(self, image):
        """Generate person segmentation mask using DeepLabV3"""
        if self.seg_model is None:
            # Fallback: create simple full person mask
            return self._create_simple_person_mask(image)
        
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            original_size = pil_image.size  # (width, height)
            
            # Preprocess image for segmentation
            input_tensor = self.seg_transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run segmentation
            with torch.no_grad():
                output = self.seg_model(input_tensor)['out'][0]
                output_predictions = output.argmax(0)  # Get class predictions
            
            # Extract person mask (class 15 in COCO/Pascal VOC for person)
            person_mask = (output_predictions == 15).float().cpu().numpy()
            
            # Resize mask back to original image size
            person_mask_resized = cv2.resize(person_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Check if we found any person pixels
            if person_mask_resized.sum() < 100:  # Too few pixels, use fallback
                return self._create_simple_person_mask(image)
            
            # Generate 5-part masks from the person mask
            masks = self._create_part_masks_from_segmentation(person_mask_resized)
            return masks
            
        except Exception as e:
            print(f"Segmentation failed: {e}")
            return self._create_simple_person_mask(image)
    
    def _create_simple_person_mask(self, image):
        """Create simple 5-part vertical strip masks as fallback"""
        if isinstance(image, Image.Image):
            h, w = image.size[1], image.size[0]
        else:
            h, w = image.shape[:2]
        
        # Create 5 horizontal strips (full person)
        part_height = h // 5
        masks = []
        
        for i in range(5):
            mask = np.zeros((h, w), dtype=np.float32)
            start_y = i * part_height
            end_y = (i + 1) * part_height if i < 4 else h
            mask[start_y:end_y, :] = 1.0
            masks.append(mask)
        
        return masks
    
    def _create_part_masks_from_segmentation(self, person_mask):
        """Create 5-part body masks from person segmentation mask"""
        h, w = person_mask.shape
        masks = []
        
        # Find person boundaries
        person_pixels = np.where(person_mask > 0.5)
        if len(person_pixels[0]) == 0:
            return self._create_simple_person_mask(person_mask)
        
        min_y, max_y = person_pixels[0].min(), person_pixels[0].max()
        min_x, max_x = person_pixels[1].min(), person_pixels[1].max()
        
        person_height = max_y - min_y + 1
        person_width = max_x - min_x + 1
        
        # Define part boundaries based on typical human proportions
        # Head: top 15% of person height
        # Upper torso: 15% - 40%
        # Lower torso: 40% - 60%
        # Upper limbs: 15% - 60% (side regions + center for torso)
        # Lower limbs: 60% - 100%
        
        part_boundaries = [
            (0.0, 0.15),    # Head
            (0.15, 0.40),   # Upper torso
            (0.40, 0.60),   # Lower torso
            (0.15, 0.60),   # Upper limbs (arms + torso area)
            (0.60, 1.0)     # Lower limbs (legs)
        ]
        
        for i, (start_ratio, end_ratio) in enumerate(part_boundaries):
            mask = np.zeros((h, w), dtype=np.float32)
            
            start_y = int(min_y + start_ratio * person_height)
            end_y = int(min_y + end_ratio * person_height)
            start_y = max(0, min(start_y, h-1))
            end_y = max(start_y+1, min(end_y, h))
            
            if i == 3:  # Upper limbs (arms) - focus on side regions
                # Create a mask that includes arm areas and some torso
                region_mask = np.zeros((h, w), dtype=np.float32)
                region_mask[start_y:end_y, :] = person_mask[start_y:end_y, :]
                
                # Enhance arm regions by expanding horizontally
                arm_expansion = max(1, person_width // 6)
                left_boundary = max(0, min_x - arm_expansion)
                right_boundary = min(w, max_x + arm_expansion)
                
                # Create expanded mask for arms
                expanded_mask = np.zeros((h, w), dtype=np.float32)
                expanded_mask[start_y:end_y, left_boundary:right_boundary] = 1.0
                
                # Combine with person mask
                mask = region_mask * expanded_mask
            else:
                # Apply person mask to the vertical region
                mask[start_y:end_y, :] = person_mask[start_y:end_y, :]
            
            # Ensure mask has some content
            if mask.sum() < 10:  # Very few pixels, use simple strip
                mask = np.zeros((h, w), dtype=np.float32)
                mask[start_y:end_y, :] = 1.0
            
            masks.append(mask)
        
        return masks
    
    def extract_features_with_masks(self, image):
        """Extract ReID features with segmentation-generated masks"""
        # Generate person segmentation and part masks
        masks = self.generate_person_mask(image)
        
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
    
    def extract_features(self, image):
        """Extract ReID features (with masks if available, fallback without masks)"""
        try:
            features, masks = self.extract_features_with_masks(image)
            return features
        except Exception as e:
            print(f"Mask-based feature extraction failed: {e}")
            # Fallback to simple feature extraction
            return self._extract_features_simple(image)
    
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
        """Process a single frame with segmentation-based ReID"""
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
                        # Extract features with segmentation masks
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
        """Run real-time ReID with segmentation-based processing"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Set width to 640
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cv2.namedWindow('Person Re-Identification with Segmentation', cv2.WINDOW_NORMAL)
        
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
            mask_type = 'DeepLabV3' if SEGMENTATION_AVAILABLE and self.seg_model else 'Simple'
            status_text = f"Gallery: {len(self.gallery_ids)} persons | Masks: {mask_type}"
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Person Re-Identification with Segmentation', annotated_frame)
            
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
        """Save current detections to gallery with segmentation masks"""
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
                            mask_type = 'DeepLabV3' if SEGMENTATION_AVAILABLE and self.seg_model else 'simple'
                            print(f"Added Person {self.next_person_id} to gallery (with {mask_type} masks)")
                            self.next_person_id += 1
    
    def _show_current_masks(self, frame):
        """Show segmentation masks for current frame detections"""
        results = self.yolo(frame, classes=0, conf=0.5)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        masks = self.generate_person_mask(person_img)
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
            mask_type = 'DeepLabV3' if SEGMENTATION_AVAILABLE and self.seg_model else 'simple'
            mask_info = f"w/ {mask_type}" if self.gallery_masks[idx] is not None else "no masks"
            cv2.putText(gallery_viz, f"ID {person_id}", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(gallery_viz, mask_info, (x1 + 5, y1 + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        cv2.imshow('Gallery', gallery_viz)
        cv2.waitKey(0)
        cv2.destroyWindow('Gallery')


def main():
    # Check if required packages are available
    if not SEGMENTATION_AVAILABLE:
        print("Warning: Torchvision segmentation not available. Using simple mask fallback.")
        print()
    else:
        print("Using DeepLabV3 segmentation model (works on macOS)")
    
    # Paths to your models
    reid_model_path = "pretrained_models/bpbreid_market1501_hrnet32_10642.pth"
    hrnet_path = "pretrained_models/hrnetv2_w32_imagenet_pretrained.pth"
    
    # Create enhanced ReID system with segmentation
    reid_system = PersonReIDWithSegmentation(reid_model_path, hrnet_path)
    
    # Run camera
    reid_system.run_camera()


if __name__ == "__main__":
    main()