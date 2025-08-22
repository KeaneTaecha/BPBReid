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

class PersonReID:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the ReID system with YOLO detector and BPBReID model"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')  # You can use yolov8s.pt for better accuracy
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BPBReID model
        self.reid_model = self._load_reid_model(reid_model_path, hrnet_path)
        self.reid_model.eval()
        
        # Setup transforms for ReID (same as in the config)
        self.transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Gallery to store person features
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Tracking confidence threshold
        self.reid_threshold = 0.7  # Adjust based on your needs
        
    def _load_reid_model(self, model_path, hrnet_path):
        """Load the BPBReID model"""
        # Load the checkpoint first to get the config
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # The model uses a config object, let's create a simple one
        from types import SimpleNamespace
        
        # Create config structure that matches what the model expects
        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.load_weights = model_path
        config.model.load_config = True
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.mask_filtering_testing = True
        config.model.bpbreid.mask_filtering_training = False
        config.model.bpbreid.test_embeddings = ['bn_foreg', 'parts']
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'
        config.model.bpbreid.masks.preprocess = 'eight_v'  # Changed from 'five_v' to 'eight_v'
        config.model.bpbreid.masks.parts_num = 8  # Changed from 5 to 8 parts
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.test_use_target_segmentation = 'none'
        config.model.bpbreid.testing_binary_visibility_score = True
        config.model.bpbreid.training_binary_visibility_score = True
        
        # Build model with config
        model = torchreid.models.build_model(
            name='bpbreid',
            num_classes=702,  # OccludedDuke has 702 training identities
            config=config,
            pretrained=True
        )
        
        # Handle DataParallel state dict (remove 'module.' prefix)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
                
        # Load the cleaned state dict
        model.load_state_dict(new_state_dict, strict=False)
            
        model = model.to(self.device)
        return model
    
    def extract_features(self, image):
        """Extract ReID features from a person image"""
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            # The model returns multiple outputs
            outputs = self.reid_model(image_tensor)
            
            # BPBReID returns a tuple: (output_dict, visibility_scores, parts_masks)
            if isinstance(outputs, tuple):
                output_dict = outputs[0]
                
                # Extract the foreground features (bn_foreg) which are the main features
                if 'bn_foreg' in output_dict:
                    feature_vector = output_dict['bn_foreg']
                elif 'foreground' in output_dict:
                    feature_vector = output_dict['foreground']
                else:
                    # Fallback to first available feature
                    feature_vector = list(output_dict.values())[0]
                    
            elif isinstance(outputs, dict):
                # If it's already a dict
                if 'bn_foreg' in outputs:
                    feature_vector = outputs['bn_foreg']
                elif 'foreground' in outputs:
                    feature_vector = outputs['foreground']
                else:
                    feature_vector = list(outputs.values())[0]
            else:
                # If it's a tensor
                feature_vector = outputs
        
        # Ensure it's 2D (batch_size, feature_dim)
        if len(feature_vector.shape) > 2:
            feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
        # Normalize features
        feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
        return feature_vector
    
    def match_person(self, query_features):
        """Match query features against gallery"""
        if len(self.gallery_features) == 0:
            return None, 0.0
        
        # Stack gallery features
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        
        # Compute cosine similarity
        similarities = torch.mm(query_features, gallery_tensor.t())
        
        # Get best match
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        if best_similarity > self.reid_threshold:
            return self.gallery_ids[best_idx], best_similarity
        else:
            return None, best_similarity
    
    def add_to_gallery(self, features, person_id, image):
        """Add a person to the gallery"""
        self.gallery_features.append(features)
        self.gallery_ids.append(person_id)
        self.gallery_images.append(image.copy())
    
    def process_frame(self, frame):
        """Process a single frame: detect persons and perform ReID"""
        # Detect persons using YOLO
        results = self.yolo(frame, classes=0, conf=0.5)  # class 0 is person
        
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    # Crop person image
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        # Extract features
                        features = self.extract_features(person_img)
                        
                        # Try to match with gallery
                        matched_id, similarity = self.match_person(features)
                        
                        if matched_id is not None:
                            # Known person
                            label = f"Person {matched_id} ({similarity:.2f})"
                            color = (0, 255, 0)  # Green for known persons
                        else:
                            # New person
                            label = f"New Person ({similarity:.2f})"
                            color = (0, 0, 255)  # Red for new persons
                            
                            # Optionally add to gallery automatically
                            # self.add_to_gallery(features, self.next_person_id, person_img)
                            # self.next_person_id += 1
                        
                        # Draw bounding box and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame
    
    def run_camera(self):
        """Run real-time ReID on camera feed"""
        cap = cv2.VideoCapture(0)  # Use 0 for default camera
        
        # Create window
        cv2.namedWindow('Person Re-Identification', cv2.WINDOW_NORMAL)
        
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current detections to gallery")
        print("- Press 'c' to clear gallery")
        print("- Press 'g' to show gallery")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Add status information
            status_text = f"Gallery size: {len(self.gallery_ids)} persons"
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Person Re-Identification', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current detections to gallery
                self._save_current_detections(frame)
            elif key == ord('c'):
                # Clear gallery
                self.gallery_features = []
                self.gallery_ids = []
                self.gallery_images = []
                print("Gallery cleared!")
            elif key == ord('g'):
                # Show gallery
                self._show_gallery()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _save_current_detections(self, frame):
        """Save current detections to gallery"""
        results = self.yolo(frame, classes=0, conf=0.5)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        features = self.extract_features(person_img)
                        matched_id, similarity = self.match_person(features)
                        
                        if matched_id is None:
                            # New person, add to gallery
                            self.add_to_gallery(features, self.next_person_id, person_img)
                            print(f"Added Person {self.next_person_id} to gallery")
                            self.next_person_id += 1
    
    def _show_gallery(self):
        """Display gallery in a separate window"""
        if len(self.gallery_images) == 0:
            print("Gallery is empty!")
            return
        
        # Create gallery visualization
        gallery_height = 200
        gallery_width = 150
        cols = min(6, len(self.gallery_images))
        rows = (len(self.gallery_images) + cols - 1) // cols
        
        gallery_viz = np.zeros((rows * gallery_height, cols * gallery_width, 3), dtype=np.uint8)
        
        for idx, (img, person_id) in enumerate(zip(self.gallery_images, self.gallery_ids)):
            row = idx // cols
            col = idx % cols
            
            # Resize image
            resized = cv2.resize(img, (gallery_width, gallery_height))
            
            # Place in gallery
            y1 = row * gallery_height
            y2 = (row + 1) * gallery_height
            x1 = col * gallery_width
            x2 = (col + 1) * gallery_width
            gallery_viz[y1:y2, x1:x2] = resized
            
            # Add label
            cv2.putText(gallery_viz, f"Person {person_id}", (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Gallery', gallery_viz)
        cv2.waitKey(0)
        cv2.destroyWindow('Gallery')


def main():
    # Paths to your models
    reid_model_path = "pretrained_models/bpbreid_occluded_duke_hrnet32_10670.pth"
    hrnet_path = "pretrained_models/hrnetv2_w32_imagenet_pretrained.pth"
    
    # Create ReID system
    reid_system = PersonReID(reid_model_path, hrnet_path)
    
    # Run camera
    reid_system.run_camera()


if __name__ == "__main__":
    main()