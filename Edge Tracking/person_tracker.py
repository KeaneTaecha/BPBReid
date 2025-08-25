import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
import json
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchreid
from types import SimpleNamespace
from pathlib import Path
import sys

# Add the parent directory to sys.path for torchreid modules
sys.path.append(str(Path(__file__).parent.parent))

class PersonTracker:
    def __init__(self, max_disappeared=15, max_distance=100):
        self.persons = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0

        # BPBReid configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize YOLO for person detection and pose estimation
        self.initialize_model()
        
        # Initialize BPBReid components
        self.initialize_bpbreid()
        
        # Gallery storage for BPBReid
        self.gallery_features = None
        self.gallery_loaded = False
        
        # Reidentification parameters
        self.reid_threshold = 0.45
        self.keypoint_confidence_threshold = 0.5
        self.person_detection_threshold = 0.7
        
        # Frame buffer for new person verification
        self.new_person_buffer = {}
        self.frames_to_check = 4
        self.frames_to_match = 2
        
        self.history_file = "person_history.pkl"
        
        print("Person tracker with BPBReid initialized successfully")

    def initialize_bpbreid(self):
        """Initialize BPBReid model and configuration"""
        try:
            # Get paths - adjust these to your actual model paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            
            reid_model_path = os.path.join(parent_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
            hrnet_path = os.path.join(parent_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")
            
            if not os.path.exists(reid_model_path) or not os.path.exists(hrnet_path):
                print(f"Warning: BPBReid models not found. ReID features will be disabled.")
                print(f"Expected paths: {reid_model_path}, {hrnet_path}")
                self.bpbreid_enabled = False
                return
            
            # Create BPBReid configuration
            self.config = self._create_bpbreid_config(reid_model_path, hrnet_path)
            
            # Load BPBReid model
            self.reid_model = self._load_bpbreid_model()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize((self.config.data.height, self.config.data.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.data.norm_mean, std=self.config.data.norm_std)
            ])
            
            # Initialize YOLO for pose estimation
            self.yolo_pose = YOLO('yolov8n-pose.pt')
            
            self.bpbreid_enabled = True
            print("BPBReid system initialized successfully")
            
        except Exception as e:
            print(f"Error initializing BPBReid: {e}")
            self.bpbreid_enabled = False

    def _create_bpbreid_config(self, model_path, hrnet_path):
        """Create BPBReid configuration"""
        config = SimpleNamespace()
        
        config.model = SimpleNamespace()
        config.model.name = 'bpbreid'
        config.model.load_weights = model_path
        config.model.pretrained = True
        
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'
        config.model.bpbreid.dim_reduce = 'after_pooling'
        config.model.bpbreid.dim_reduce_output = 512
        config.model.bpbreid.last_stride = 1
        config.model.bpbreid.shared_parts_id_classifier = False
        config.model.bpbreid.learnable_attention_enabled = True
        config.model.bpbreid.test_use_target_segmentation = 'soft'
        config.model.bpbreid.testing_binary_visibility_score = True
        config.model.bpbreid.training_binary_visibility_score = True
        config.model.bpbreid.mask_filtering_testing = True
        config.model.bpbreid.mask_filtering_training = True
        
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.masks.preprocess = 'five_v'
        config.model.bpbreid.masks.softmax_weight = 1.0
        config.model.bpbreid.masks.background_computation_strategy = 'threshold'
        config.model.bpbreid.masks.mask_filtering_threshold = 0.3
        
        config.data = SimpleNamespace()
        config.data.height = 384
        config.data.width = 128
        config.data.norm_mean = [0.485, 0.456, 0.406]
        config.data.norm_std = [0.229, 0.224, 0.225]
        
        config.loss = SimpleNamespace()
        config.loss.name = 'part_based'
        
        return config

    def _load_bpbreid_model(self):
        """Load BPBReid model"""
        try:
            model = torchreid.models.build_model(
                name='bpbreid',
                num_classes=751,
                config=self.config,
                pretrained=True
            )
            
            checkpoint = torch.load(self.config.model.load_weights, map_location=self.device)
            
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
            
        except Exception as e:
            print(f"Error loading BPBReid model: {e}")
            raise e

    def load_gallery_person(self, gallery_path):
        """Load gallery person image and extract BPBReid features"""
        if not self.bpbreid_enabled:
            print("BPBReid not enabled, skipping gallery loading")
            return False
            
        try:
            print(f"Loading gallery person from: {gallery_path}")
            
            # Load and detect person in gallery image
            image = cv2.imread(gallery_path)
            if image is None:
                print(f"Could not load gallery image: {gallery_path}")
                return False
            
            # Detect person using YOLO
            results = self.model(image, conf=self.person_detection_threshold, classes=[0], verbose=False)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = box.astype(int)
                        person_img = image[y1:y2, x1:x2]
                        
                        if person_img.size > 0:
                            # Extract BPBReid features
                            self.gallery_features = self.extract_bpbreid_features(person_img)
                            self.gallery_loaded = True
                            print(f"Gallery person loaded successfully. Feature shape: {self.gallery_features.shape}")
                            return True
            
            print("No person detected in gallery image")
            return False
            
        except Exception as e:
            print(f"Error loading gallery person: {e}")
            return False

    def generate_yolo_pose_masks(self, person_img):
        """Generate pose-based masks using YOLO Pose (simplified from BPBReid)"""
        try:
            results = self.yolo_pose(person_img, task='pose')
            
            if len(results) == 0 or not hasattr(results[0], 'keypoints'):
                return None
            
            if results[0].keypoints is None or len(results[0].keypoints.data) == 0:
                return None
            
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            if keypoints.shape[0] != 17:
                return None
            
            h, w = person_img.shape[:2]
            feat_h, feat_w = self.config.data.height // 8, self.config.data.width // 8
            
            # Create simplified 5-part masks
            masks = torch.zeros(1, 6, feat_h, feat_w)
            
            # Simplified mask generation - just create basic regions
            # This is a placeholder - you can copy the full implementation from bpbreid_yolo_masked_reid_fin2.py
            masks[0, 1:] = 0.2  # Basic uniform distribution for now
            masks[0, 0] = 0.0   # Background
            
            # Normalize
            mask_sum = masks.sum(dim=1, keepdim=True)
            mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
            masks = masks / mask_sum
            
            return masks.to(self.device)
            
        except Exception as e:
            print(f"Mask generation error: {e}")
            return None

    def extract_bpbreid_features(self, person_img):
        """Extract BPBReid features from person image"""
        try:
            # Convert BGR to RGB
            if len(person_img.shape) == 3 and person_img.shape[2] == 3:
                person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            else:
                person_img_rgb = person_img
                
            # Convert to PIL Image
            image = Image.fromarray(person_img_rgb)
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate masks
            masks = self.generate_yolo_pose_masks(person_img)
            
            # Extract features
            with torch.no_grad():
                outputs = self.reid_model(image_tensor, external_parts_masks=masks)
                features = self._process_reid_output(outputs)
                
                if features is not None and features.numel() > 0:
                    features = self._apply_feature_normalization(features)
                    return features
                else:
                    return torch.randn(1, 512).to(self.device)
                    
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return torch.randn(1, 512).to(self.device)

    def _process_reid_output(self, outputs):
        """Process BPBReid model output"""
        try:
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                embeddings_dict = outputs[0]
                
                if isinstance(embeddings_dict, dict):
                    priority_keys = ['bn_foreg', 'foreground', 'bn_global', 'global', 'parts']
                    
                    for key in priority_keys:
                        if key in embeddings_dict:
                            features = embeddings_dict[key]
                            if isinstance(features, torch.Tensor) and features.numel() > 0:
                                if len(features.shape) > 2:
                                    features = features.view(features.size(0), -1)
                                return features
                    
                    for key, value in embeddings_dict.items():
                        if isinstance(value, torch.Tensor) and value.numel() > 0:
                            if len(value.shape) > 2:
                                value = value.view(value.size(0), -1)
                            return value
                
                elif isinstance(embeddings_dict, torch.Tensor):
                    if len(embeddings_dict.shape) > 2:
                        return embeddings_dict.view(embeddings_dict.size(0), -1)
                    return embeddings_dict
            
            return None
            
        except Exception as e:
            print(f"Error processing output: {e}")
            return None

    def _apply_feature_normalization(self, features):
        """Apply feature normalization"""
        features = F.normalize(features, p=2, dim=1)
        features = features - features.mean(dim=1, keepdim=True)
        features = F.normalize(features, p=2, dim=1)
        return features

    def compute_similarity(self, query_features, gallery_features):
        """Compute similarity between query and gallery features"""
        if query_features.dim() == 1:
            query_features = query_features.unsqueeze(0)
        if gallery_features.dim() == 1:
            gallery_features = gallery_features.unsqueeze(0)
        
        # Cosine similarity
        cosine_sim = torch.mm(query_features, gallery_features.t())
        
        # Euclidean distance similarity
        query_expanded = query_features.unsqueeze(1)
        gallery_expanded = gallery_features.unsqueeze(0)
        euclidean_dist = torch.norm(query_expanded - gallery_expanded, p=2, dim=2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combined similarity
        combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        return combined_sim.item()

    def check_gallery_match(self, person_img):
        """Check if person matches gallery using BPBReid"""
        if not self.bpbreid_enabled or not self.gallery_loaded:
            return False, 0.0
        
        try:
            # Extract features from current person
            query_features = self.extract_bpbreid_features(person_img)
            
            # Compute similarity
            similarity = self.compute_similarity(query_features, self.gallery_features)
            
            # Check if match
            is_match = similarity > self.reid_threshold
            
            return is_match, similarity
            
        except Exception as e:
            print(f"Error checking gallery match: {e}")
            return False, 0.0

    def initialize_model(self):
        """Initialize YOLOv11 model using Ultralytics"""
        try:
            engine_path = "yolo11n.engine"

            if os.path.exists(engine_path):
                print(f"Loading existing TensorRT engine from {engine_path}")
                self.model = YOLO(engine_path)
            else:
                print("TensorRT engine not found. Creating one")
                self.model = YOLO("yolo11n.pt")
                self.model.export(format="engine", half=True)
                self.model = YOLO("yolo11n.engine")

            print("YOLOv11 model initialized successfully")
            self.conf_threshold = 0.7
            self.person_class = 0

        except Exception as e:
            print(f"Error initializing YOLOv11 model: {e}")
            raise

    def detect_persons(self, frame):
        """Detect persons in the frame using YOLOv11"""
        start_time = time.time()
        try:
            results = self.model(frame,
                                conf=self.conf_threshold,
                                classes=[self.person_class],
                                verbose=False)

            boxes = []
            features_list = []

            for result in results:
                for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    boxes.append([x, y, w, h])
                    
                    # For new detections, we'll check against gallery
                    features_list.append(None)  # Placeholder

            elapsed_time = time.time() - start_time
            print(f"detect_persons time: {elapsed_time:.4f} seconds")

            return boxes, features_list

        except Exception as e:
            print(f"Error in person detection: {e}")
            return [], []

    def update(self, frame, boxes, extracted_features=None):
        """Update person tracking with new detections and BPBReid verification"""
        start_time = time.time()

        if len(boxes) == 0:
            print("No Detection")
            for person_id in list(self.persons.keys()):
                self.persons[person_id]["disappeared"] += 1

                if self.persons[person_id]["disappeared"] > self.max_disappeared:
                    history = self.load_history()
                    history[person_id] = {
                        "features": self.persons[person_id].get("features"),
                        "last_seen": time.time(),
                        "bbox": self.persons[person_id]["bbox"]
                    }
                    self.save_history(history)
                    print(f"Person ID {person_id} moved to history")
                    del self.persons[person_id]

            elapsed_time = time.time() - start_time
            print(f"update (No detection) time: {elapsed_time:.4f} seconds")
            return self.persons

        print(f"Processing {len(boxes)} detections")

        centroids = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            centroid = (int(x + w/2), int(y + h/2))
            centroids.append(centroid)

        used_detections = set()

        # Match with existing persons based on spatial proximity
        if len(self.persons) > 0:
            person_ids = list(self.persons.keys())
            
            for person_id in person_ids:
                person = self.persons[person_id]
                prev_centroid = person["centroid"]
                
                # Find closest detection
                min_dist = float('inf')
                closest_idx = -1
                
                for j, centroid in enumerate(centroids):
                    if j not in used_detections:
                        dist = np.sqrt((prev_centroid[0] - centroid[0])**2 + 
                                     (prev_centroid[1] - centroid[1])**2)
                        if dist < min_dist and dist < self.max_distance:
                            min_dist = dist
                            closest_idx = j
                
                if closest_idx >= 0:
                    # Update existing person
                    self.persons[person_id]["centroid"] = centroids[closest_idx]
                    self.persons[person_id]["bbox"] = boxes[closest_idx]
                    self.persons[person_id]["disappeared"] = 0
                    used_detections.add(closest_idx)
                    print(f"Updated person ID {person_id}")
                else:
                    # Mark as disappeared
                    self.persons[person_id]["disappeared"] += 1
                    if self.persons[person_id]["disappeared"] > self.max_disappeared:
                        del self.persons[person_id]
                        print(f"Removed person ID {person_id}")

        # Process new detections with BPBReid verification
        for j in range(len(boxes)):
            if j in used_detections:
                continue
            
            # Extract person image for BPBReid
            x, y, w, h = boxes[j]
            person_img = frame[y:y+h, x:x+w]
            
            if person_img.size == 0:
                continue
            
            # Initialize buffer for this detection if needed
            detection_key = f"{x}_{y}_{w}_{h}"
            
            if self.bpbreid_enabled and self.gallery_loaded:
                # Check if we're already tracking this detection
                if detection_key not in self.new_person_buffer:
                    self.new_person_buffer[detection_key] = {
                        'frames_checked': 0,
                        'frames_matched': 0,
                        'centroid': centroids[j],
                        'bbox': boxes[j]
                    }
                
                buffer = self.new_person_buffer[detection_key]
                
                # Check against gallery for first 4 frames
                if buffer['frames_checked'] < self.frames_to_check:
                    is_match, similarity = self.check_gallery_match(person_img)
                    buffer['frames_checked'] += 1
                    
                    if is_match:
                        buffer['frames_matched'] += 1
                        print(f"Frame {buffer['frames_checked']}/{self.frames_to_check}: "
                              f"Match found (similarity: {similarity:.3f})")
                    else:
                        print(f"Frame {buffer['frames_checked']}/{self.frames_to_check}: "
                              f"No match (similarity: {similarity:.3f})")
                    
                    # Check if we've evaluated enough frames
                    if buffer['frames_checked'] >= self.frames_to_check:
                        if buffer['frames_matched'] >= self.frames_to_match:
                            # This is the gallery person - assign special ID 0
                            self.persons[0] = {
                                "centroid": centroids[j],
                                "bbox": boxes[j],
                                "disappeared": 0,
                                "is_gallery": True
                            }
                            print(f"Gallery person detected! Assigned ID 0 "
                                  f"({buffer['frames_matched']}/{self.frames_to_check} frames matched)")
                        else:
                            # Not the gallery person - assign regular ID
                            self.register(centroids[j], boxes[j], None)
                            print(f"New person registered with ID {self.next_id-1} "
                                  f"({buffer['frames_matched']}/{self.frames_to_check} frames matched)")
                        
                        # Clean up buffer
                        del self.new_person_buffer[detection_key]
                
            else:
                # BPBReid not enabled - use regular registration
                self.register(centroids[j], boxes[j], None)
                print(f"Registered new person with ID {self.next_id-1} (BPBReid disabled)")

        # Clean up old buffers
        current_detections = {f"{box[0]}_{box[1]}_{box[2]}_{box[3]}" for box in boxes}
        buffer_keys_to_remove = []
        for key in self.new_person_buffer:
            if key not in current_detections:
                buffer_keys_to_remove.append(key)
        for key in buffer_keys_to_remove:
            del self.new_person_buffer[key]

        elapsed_time = time.time() - start_time
        print(f"update time: {elapsed_time:.4f} seconds")
        return self.persons

    def register(self, centroid, bbox, features):
        """Register a new person"""
        existing_ids = set(self.persons.keys())
        history = self.load_history()
        historical_ids = set(history.keys())
        all_ids = existing_ids.union(historical_ids)
        
        if all_ids:
            max_existing_id = max(int(id) for id in all_ids if id != 0)  # Exclude gallery ID
            if self.next_id <= max_existing_id:
                self.next_id = max_existing_id + 1
        
        # Make sure we don't use ID 0 (reserved for gallery)
        if self.next_id == 0:
            self.next_id = 1
        
        self.persons[self.next_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "disappeared": 0,
            "features": features,
            "is_gallery": False
        }
        print(f"Registered new person with ID {self.next_id}")
        self.next_id += 1

    def save_history(self, history):
        """Save history to file"""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(history, f)
            print(f"History saved to {self.history_file}")
        except Exception as e:
            print(f"Error saving history: {e}")

    def load_history(self):
        """Load history from file and return it"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    history = pickle.load(f)
                print(f"History loaded from {self.history_file}, {len(history)} entries")
                return history
            else:
                print("No history file found, starting with empty history")
                return {}
        except Exception as e:
            print(f"Error loading history: {e}")
            return {}