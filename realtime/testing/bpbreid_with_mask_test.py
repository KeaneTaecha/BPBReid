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
import json
from datetime import datetime

# For pose estimation - you'll need to install these
try:
    import openpifpaf
    PIFPAF_AVAILABLE = True
except ImportError:
    print("Warning: OpenPifPaf not available. Install with: pip install openpifpaf")
    PIFPAF_AVAILABLE = False

class PersonReIDVideoTester:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the ReID system for video testing"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load BPBReID model
        self.reid_model = self._load_reid_model(reid_model_path, hrnet_path)
        self.reid_model.eval()
        
        # Setup transforms for ReID - NO RESIZE HERE (we'll do smart padding first)
        self.transform = transforms.Compose([
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
        self.gallery_masks = []
        self.next_person_id = 1
        
        # Tracking confidence threshold
        self.reid_threshold = 0.7
        
        # Test results storage
        self.test_results = {}
        
    def preprocess_with_padding(self, image):
        """
        Smart padding to preserve aspect ratio and match Market-1501 training data
        Steps:
        1. Pad image to 1:2 aspect ratio (width:height = 1:2)
        2. Resize to 128×384 (maintaining the 1:2 ratio)
        3. Final resize to 384×128 (model input size)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        target_ratio = 1/2  # width/height = 0.5 (Market-1501 aspect ratio)
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
        
        # First resize to Market-1501 dimensions (128×384) to match training
        image = cv2.resize(image, (128, 384))  # (width, height)
        
        # Then resize to model input size (384×128)
        image = cv2.resize(image, (384, 128))  # (width, height)
        
        return image
    
    def preprocess_masks_with_padding(self, masks, original_image):
        """
        Apply the same padding and resizing to masks as applied to the image
        """
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        h, w = original_image.shape[:2]
        target_ratio = 1/2
        current_ratio = w/h
        
        processed_masks = []
        
        for mask in masks:
            # Apply same padding as image
            if current_ratio > target_ratio:
                # Pad height
                target_h = int(w / target_ratio)
                pad_h = target_h - h
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                padded_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, 0, 0, 
                                               cv2.BORDER_CONSTANT, value=0)
            else:
                # Pad width
                target_w = int(h * target_ratio)
                pad_w = target_w - w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                padded_mask = cv2.copyMakeBorder(mask, 0, 0, pad_left, pad_right,
                                               cv2.BORDER_CONSTANT, value=0)
            
            # Resize to Market-1501 dimensions first, then to model input
            mask_128_384 = cv2.resize(padded_mask, (128, 384))
            mask_384_128 = cv2.resize(mask_128_384, (384, 128))
            
            processed_masks.append(mask_384_128)
        
        return processed_masks
        
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
        """Generate 5-part body segmentation masks BEFORE padding/resizing"""
        if self.pose_predictor is None:
            # Fallback: create simple vertical strip masks
            return self._create_simple_part_masks(image)
        
        try:
            # Convert to PIL if needed for pose estimation
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Get pose keypoints using OpenPifPaf
            predictions, gt_anns, image_meta = self.pose_predictor.pil_image(pil_image)
            
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
        """Extract ReID features with body part masks and smart padding"""
        # Store original image for mask processing
        original_image = image.copy() if isinstance(image, np.ndarray) else image
        
        # Generate body part masks BEFORE any resizing
        masks = self.generate_body_part_masks(original_image)
        
        # Apply smart padding to image
        processed_image = self.preprocess_with_padding(image)
        
        # Apply smart padding to masks
        processed_masks = self.preprocess_masks_with_padding(masks, original_image)
        
        # Add background mask (like training)
        processed_masks = self.add_background_mask(processed_masks, threshold=0.3)
        
        # Downsample masks to training size (96x32)
        processed_masks = self.resize_masks_to_training_size(processed_masks, scale=4)
        
        # Convert processed image to PIL and apply remaining transforms
        if isinstance(processed_image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = processed_image
        
        # Apply transforms (no resize needed - already done in preprocessing)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Convert processed masks to tensors
        mask_tensors = []
        for mask in processed_masks:
            mask_tensor = torch.from_numpy(mask).float()
            mask_tensors.append(mask_tensor)
        
        # Stack masks: shape (num_parts+1, 96, 32)  # +1 for background
        masks_tensor = torch.stack(mask_tensors, dim=0).unsqueeze(0).to(self.device)

        ## Solution:
        # # Convert processed masks to tensors with consistent shape
        # mask_tensors = []
        # for mask in processed_masks:
        #     mask_tensor = torch.from_numpy(mask).float()
        #     # Ensure 2D shape (H, W)
        #     if len(mask_tensor.shape) == 3:
        #         mask_tensor = mask_tensor.squeeze()
        #     elif len(mask_tensor.shape) == 1:
        #         # Reshape if flattened
        #         mask_tensor = mask_tensor.view(96, 32)  # Training size
        #     mask_tensors.append(mask_tensor)

        # # Stack masks: shape (num_parts+1, 96, 32)  # +1 for background
        # masks_tensor = torch.stack(mask_tensors, dim=0).unsqueeze(0).to(self.device)
        
        # Extract features with masks
        with torch.no_grad():
            # Pass both image and masks to the model
            outputs = self.reid_model(image_tensor, masks_tensor)
            
            # BPBReID returns (output_dict, visibility_scores, parts_masks)
            if isinstance(outputs, tuple):
                output_dict = outputs[0]
                
                # # Use both foreground and parts features
                # features_list = []
                
                # # Add foreground features
                # if 'bn_foreg' in output_dict:
                #     features_list.append(output_dict['bn_foreg'])
                
                # # Add parts features
                # if 'parts' in output_dict:
                #     features_list.append(output_dict['parts'])
                
                # # Concatenate features
                # if len(features_list) > 1:
                #     feature_vector = torch.cat(features_list, dim=1)
                # else:
                #     feature_vector = features_list[0] if features_list else output_dict[list(output_dict.keys())[0]]

                # Use both foreground and parts features
                
                features_list = []

                # Add foreground features
                if 'bn_foreg' in output_dict:
                    foreg_feat = output_dict['bn_foreg']
                    # Ensure 2D shape (batch_size, feature_dim)
                    if len(foreg_feat.shape) > 2:
                        foreg_feat = foreg_feat.view(foreg_feat.size(0), -1)
                    features_list.append(foreg_feat)

                # Add parts features
                if 'parts' in output_dict:
                    parts_feat = output_dict['parts']
                    # Ensure 2D shape (batch_size, feature_dim)
                    if len(parts_feat.shape) > 2:
                        parts_feat = parts_feat.view(parts_feat.size(0), -1)
                    features_list.append(parts_feat)

                # Concatenate features
                if len(features_list) > 1:
                    # Ensure all features have same number of dimensions before concatenation
                    processed_features = []
                    for feat in features_list:
                        if len(feat.shape) == 1:
                            feat = feat.unsqueeze(0)  # Add batch dimension if missing
                        elif len(feat.shape) > 2:
                            feat = feat.view(feat.size(0), -1)  # Flatten if more than 2D
                        processed_features.append(feat)
                    feature_vector = torch.cat(processed_features, dim=1)
                else:
                    feature_vector = features_list[0] if features_list else output_dict[list(output_dict.keys())[0]]
                    
            else:
                feature_vector = outputs
        
        # Ensure proper shape and normalize
        if len(feature_vector.shape) > 2:
            feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
        feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
        return feature_vector, processed_masks
    
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
        # Apply smart padding first
        processed_image = self.preprocess_with_padding(image)
        
        # Convert to PIL
        if isinstance(processed_image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = processed_image
        
        # Apply transforms (no resize needed)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.reid_model(image_tensor)
            
            # if isinstance(outputs, tuple):
            #     output_dict = outputs[0]
            #     if 'bn_foreg' in output_dict:
            #         feature_vector = output_dict['bn_foreg']
            #     else:
            #         feature_vector = list(output_dict.values())[0]
            # else:
            #     feature_vector = outputs

            if isinstance(outputs, tuple):
                output_dict = outputs[0]
                if 'bn_foreg' in output_dict:
                    feature_vector = output_dict['bn_foreg']
                    # Ensure 2D shape
                    if len(feature_vector.shape) > 2:
                        feature_vector = feature_vector.view(feature_vector.size(0), -1)
                else:
                    feature_vector = list(output_dict.values())[0]
                    if len(feature_vector.shape) > 2:
                        feature_vector = feature_vector.view(feature_vector.size(0), -1)
            else:
                feature_vector = outputs
        
        if len(feature_vector.shape) > 2:
            feature_vector = feature_vector.view(feature_vector.size(0), -1)
            
        feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
        return feature_vector
    
    def add_background_mask(self, masks, threshold=0.3):
        """Add background mask as channel 0 using threshold strategy"""
        masks_tensor = torch.stack([torch.from_numpy(mask) for mask in masks])
        background_mask = masks_tensor.max(dim=0)[0] < threshold
        background_mask = background_mask.float()
        return [background_mask.numpy()] + masks
    
    def resize_masks_to_training_size(self, masks, scale=4):
        """Downsample masks to match training (96x32)"""
        target_h, target_w = 384 // scale, 128 // scale  # 96, 32
        resized_masks = []
        for mask in masks:
            resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized)
        return resized_masks
    
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
    
    def add_gallery_image(self, image_path):
        """Add a gallery image from file path"""
        print(f"Loading gallery image: {image_path}")
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Gallery image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect person in image
        results = self.yolo(image, classes=0, conf=0.5)
        
        person_detected = False
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    person_img = image[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        # Extract features
                        try:
                            features, masks = self.extract_features_with_masks(person_img)
                        except:
                            features = self.extract_features(person_img)
                            masks = None
                        
                        # Add to gallery
                        self.add_to_gallery(features, self.next_person_id, person_img, masks)
                        print(f"Added Person {self.next_person_id} to gallery from {image_path}")
                        self.next_person_id += 1
                        person_detected = True
                        break
            
            if person_detected:
                break
        
        if not person_detected:
            raise ValueError(f"No person detected in gallery image: {image_path}")
        
        return True
    
    def test_video(self, video_path, expected_match=True, save_annotated=False, output_path=None):
        """
        Test the ReID model on a video
        
        Args:
            video_path: Path to video file
            expected_match: True if person in video should match gallery, False otherwise
            save_annotated: Whether to save annotated video
            output_path: Path to save annotated video
        
        Returns:
            dict: Test results containing frame counts and accuracy
        """
        print(f"\nTesting video: {video_path}")
        print(f"Expected match: {expected_match}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Setup video writer if saving annotated video
        writer = None
        if save_annotated and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Test results
        results = {
            'video_path': video_path,
            'expected_match': expected_match,
            'total_frames': total_frames,
            'frames_with_detection': 0,
            'frames_with_match': 0,
            'frames_without_match': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'accuracy': 0.0,
            'similarity_scores': [],
            'frame_details': []
        }
        
        frame_idx = 0
        
        print("Processing frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_result = {
                'frame_idx': frame_idx,
                'has_detection': False,
                'has_match': False,
                'similarity': 0.0,
                'correct': False
            }
            
            # Process frame for person detection
            yolo_results = self.yolo(frame, classes=0, conf=0.5)
            annotated_frame = frame.copy()
            
            for r in yolo_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        
                        person_img = frame[y1:y2, x1:x2]
                        
                        if person_img.size > 0:
                            frame_result['has_detection'] = True
                            results['frames_with_detection'] += 1
                            
                            # Extract features and match
                            features = self.extract_features(person_img)
                            matched_id, similarity = self.match_person(features)
                            
                            frame_result['similarity'] = similarity
                            results['similarity_scores'].append(similarity)
                            
                            if matched_id is not None:
                                frame_result['has_match'] = True
                                results['frames_with_match'] += 1
                                label = f"Person {matched_id} ({similarity:.3f})"
                                color = (0, 255, 0)  # Green for match
                            else:
                                results['frames_without_match'] += 1
                                label = f"Unknown ({similarity:.3f})"
                                color = (0, 0, 255)  # Red for no match
                            
                            # Determine if prediction is correct
                            if expected_match:
                                # Should match - correct if we have a match
                                frame_result['correct'] = frame_result['has_match']
                            else:
                                # Should NOT match - correct if we don't have a match
                                frame_result['correct'] = not frame_result['has_match']
                            
                            if frame_result['correct']:
                                results['correct_predictions'] += 1
                            else:
                                results['incorrect_predictions'] += 1
                            
                            # Draw annotation
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Add correctness indicator
                            correct_text = "✓" if frame_result['correct'] else "✗"
                            correct_color = (0, 255, 0) if frame_result['correct'] else (0, 0, 255)
                            cv2.putText(annotated_frame, correct_text, (x1, y2+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, correct_color, 2)
                            
                            break  # Only process first detection per frame
            
            # Add frame info to annotated frame
            info_text = f"Frame {frame_idx+1}/{total_frames} | Expected: {'Match' if expected_match else 'No Match'}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save annotated frame
            if writer:
                writer.write(annotated_frame)
            
            results['frame_details'].append(frame_result)
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Calculate final accuracy
        total_detections = results['frames_with_detection']
        if total_detections > 0:
            results['accuracy'] = results['correct_predictions'] / total_detections
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"Test completed: {results['correct_predictions']}/{total_detections} correct predictions")
        print(f"Accuracy: {results['accuracy']:.3f}")
        
        return results
    
    def run_full_test(self, gallery_image_path, video1_path, video2_path, 
                     save_annotated=True, output_dir="reid_with_mask_test_results"):
        """
        Run complete test suite
        
        Args:
            gallery_image_path: Path to reference image for gallery
            video1_path: Path to video with same person (should match)
            video2_path: Path to video with different person (should not match)
            save_annotated: Whether to save annotated videos
            output_dir: Directory to save results
        """
        print("="*60)
        print("BPBReID Video Test Suite")
        print("="*60)
        
        # Create output directory
        if save_annotated:
            os.makedirs(output_dir, exist_ok=True)
        
        # Clear gallery first
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.gallery_masks = []
        self.next_person_id = 1
        
        # Step 1: Add gallery image
        print("\n1. Loading gallery image...")
        try:
            self.add_gallery_image(gallery_image_path)
        except Exception as e:
            print(f"Error loading gallery image: {e}")
            return None
        
        # Step 2: Test video 1 (should match)
        print("\n2. Testing Video 1 (Same Person - Should Match)...")
        video1_output = os.path.join(output_dir, "video1_annotated.mp4") if save_annotated else None
        
        try:
            results1 = self.test_video(
                video1_path, 
                expected_match=True, 
                save_annotated=save_annotated,
                output_path=video1_output
            )
        except Exception as e:
            print(f"Error testing video 1: {e}")
            return None
        
        # Step 3: Test video 2 (should not match)
        print("\n3. Testing Video 2 (Different Person - Should NOT Match)...")
        video2_output = os.path.join(output_dir, "video2_annotated.mp4") if save_annotated else None
        
        try:
            results2 = self.test_video(
                video2_path, 
                expected_match=False, 
                save_annotated=save_annotated,
                output_path=video2_output
            )
        except Exception as e:
            print(f"Error testing video 2: {e}")
            return None
        
        # Compile final results
        final_results = {
            'test_timestamp': datetime.now().isoformat(),
            'gallery_image': gallery_image_path,
            'reid_threshold': self.reid_threshold,
            'masks_enabled': PIFPAF_AVAILABLE,
            'device': str(self.device),
            'video1_results': results1,
            'video2_results': results2,
            'overall_performance': {
                'total_frames_tested': results1['frames_with_detection'] + results2['frames_with_detection'],
                'total_correct': results1['correct_predictions'] + results2['correct_predictions'],
                'total_incorrect': results1['incorrect_predictions'] + results2['incorrect_predictions'],
                'overall_accuracy': 0.0
            }
        }
        
        # Calculate overall accuracy
        total_tested = final_results['overall_performance']['total_frames_tested']
        total_correct = final_results['overall_performance']['total_correct']
        
        if total_tested > 0:
            final_results['overall_performance']['overall_accuracy'] = total_correct / total_tested
        
        # Save results to JSON
        results_file = os.path.join(output_dir, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print detailed results
        self.print_test_summary(final_results)
        
        print(f"\nResults saved to: {results_file}")
        if save_annotated:
            print(f"Annotated videos saved to: {output_dir}")
        
        return final_results
    
    def print_test_summary(self, results):
        """Print a detailed summary of test results"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Gallery Image: {results['gallery_image']}")
        print(f"ReID Threshold: {results['reid_threshold']}")
        print(f"Masks Enabled: {results['masks_enabled']}")
        print(f"Device: {results['device']}")
        
        # Video 1 Results
        v1 = results['video1_results']
        print(f"\nVIDEO 1 (Same Person - Should Match):")
        print(f"  Path: {v1['video_path']}")
        print(f"  Total Frames: {v1['total_frames']}")
        print(f"  Frames with Person Detection: {v1['frames_with_detection']}")
        print(f"  Frames with Match: {v1['frames_with_match']}")
        print(f"  Frames without Match: {v1['frames_without_match']}")
        print(f"  Correct Predictions: {v1['correct_predictions']}")
        print(f"  Incorrect Predictions: {v1['incorrect_predictions']}")
        print(f"  Accuracy: {v1['accuracy']:.3f} ({v1['accuracy']*100:.1f}%)")
        
        if v1['similarity_scores']:
            avg_sim = np.mean(v1['similarity_scores'])
            max_sim = np.max(v1['similarity_scores'])
            min_sim = np.min(v1['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Video 2 Results
        v2 = results['video2_results']
        print(f"\nVIDEO 2 (Different Person - Should NOT Match):")
        print(f"  Path: {v2['video_path']}")
        print(f"  Total Frames: {v2['total_frames']}")
        print(f"  Frames with Person Detection: {v2['frames_with_detection']}")
        print(f"  Frames with Match: {v2['frames_with_match']}")
        print(f"  Frames without Match: {v2['frames_without_match']}")
        print(f"  Correct Predictions: {v2['correct_predictions']}")
        print(f"  Incorrect Predictions: {v2['incorrect_predictions']}")
        print(f"  Accuracy: {v2['accuracy']:.3f} ({v2['accuracy']*100:.1f}%)")
        
        if v2['similarity_scores']:
            avg_sim = np.mean(v2['similarity_scores'])
            max_sim = np.max(v2['similarity_scores'])
            min_sim = np.min(v2['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Overall Results
        overall = results['overall_performance']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Frames Tested: {overall['total_frames_tested']}")
        print(f"  Total Correct: {overall['total_correct']}")
        print(f"  Total Incorrect: {overall['total_incorrect']}")
        print(f"  Overall Accuracy: {overall['overall_accuracy']:.3f} ({overall['overall_accuracy']*100:.1f}%)")
        
        # Performance Analysis
        print(f"\nPERFORMANCE ANALYSIS:")
        
        # Video 1 analysis (should match)
        v1_match_rate = v1['frames_with_match'] / v1['frames_with_detection'] if v1['frames_with_detection'] > 0 else 0
        print(f"  Video 1 Match Rate: {v1_match_rate:.3f} ({v1_match_rate*100:.1f}%) - Higher is better")
        
        # Video 2 analysis (should not match)
        v2_reject_rate = v2['frames_without_match'] / v2['frames_with_detection'] if v2['frames_with_detection'] > 0 else 0
        print(f"  Video 2 Rejection Rate: {v2_reject_rate:.3f} ({v2_reject_rate*100:.1f}%) - Higher is better")
        
        # False positive/negative analysis
        false_positives = v2['frames_with_match']  # Should not match but did
        false_negatives = v1['frames_without_match']  # Should match but didn't
        
        total_positive_cases = v1['frames_with_detection']
        total_negative_cases = v2['frames_with_detection']
        
        if total_positive_cases > 0:
            false_negative_rate = false_negatives / total_positive_cases
            print(f"  False Negative Rate: {false_negative_rate:.3f} ({false_negative_rate*100:.1f}%) - Lower is better")
        
        if total_negative_cases > 0:
            false_positive_rate = false_positives / total_negative_cases
            print(f"  False Positive Rate: {false_positive_rate:.3f} ({false_positive_rate*100:.1f}%) - Lower is better")
        
        print("="*60)
    
    def generate_comparison_plots(self, results, output_dir="reid_with_mask_test_results"):
        """Generate comparison plots for the test results"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('BPBReID Video Test Results', fontsize=16)
            
            # Plot 1: Accuracy comparison
            videos = ['Video 1\n(Same Person)', 'Video 2\n(Different Person)']
            accuracies = [results['video1_results']['accuracy'], results['video2_results']['accuracy']]
            colors = ['green', 'blue']
            
            axes[0, 0].bar(videos, accuracies, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylim(0, 1)
            
            # Add accuracy values on bars
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 2: Match/No Match distribution
            v1_match = results['video1_results']['frames_with_match']
            v1_no_match = results['video1_results']['frames_without_match']
            v2_match = results['video2_results']['frames_with_match']
            v2_no_match = results['video2_results']['frames_without_match']
            
            x = np.arange(2)
            width = 0.35
            
            axes[0, 1].bar(x - width/2, [v1_match, v2_match], width, label='Matched', color='green', alpha=0.7)
            axes[0, 1].bar(x + width/2, [v1_no_match, v2_no_match], width, label='Not Matched', color='red', alpha=0.7)
            
            axes[0, 1].set_ylabel('Number of Frames')
            axes[0, 1].set_title('Match Distribution')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(videos)
            axes[0, 1].legend()
            
            # Plot 3: Similarity scores distribution for Video 1
            if results['video1_results']['similarity_scores']:
                axes[1, 0].hist(results['video1_results']['similarity_scores'], bins=20, alpha=0.7, color='green')
                axes[1, 0].axvline(x=results['reid_threshold'], color='red', linestyle='--', label=f'Threshold ({results["reid_threshold"]})')
                axes[1, 0].set_xlabel('Similarity Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Video 1 Similarity Scores')
                axes[1, 0].legend()
            
            # Plot 4: Similarity scores distribution for Video 2
            if results['video2_results']['similarity_scores']:
                axes[1, 1].hist(results['video2_results']['similarity_scores'], bins=20, alpha=0.7, color='blue')
                axes[1, 1].axvline(x=results['reid_threshold'], color='red', linestyle='--', label=f'Threshold ({results["reid_threshold"]})')
                axes[1, 1].set_xlabel('Similarity Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Video 2 Similarity Scores')
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, "test_results_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Comparison plots saved to: {plot_path}")
            
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plots: {e}")


def main():
    """Main function to run the video test suite"""
    print("BPBReID Video Test Suite")
    print("=" * 50)
    
    # Get parent directory (one level up from current folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    realtime_dir = os.path.dirname(current_dir)               
    bpbreid_dir = os.path.dirname(realtime_dir) 
    
    # Configuration - paths relative to parent directory
    reid_model_path = os.path.join(bpbreid_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
    hrnet_path = os.path.join(bpbreid_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")
    
    # Test file paths (you need to provide these) - also relative to parent directory
    gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "person.jpg")  # Reference image
    video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "correct.MOV")      # Video with same person
    video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "incorrect.MOV") # Video with different person
    
    # Verify files exist
    required_files = [reid_model_path, hrnet_path, gallery_image_path, video1_path, video2_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all files are available before running the test.")
        return
    
    try:
        # Create test system
        print("Initializing BPBReID test system...")
        tester = PersonReIDVideoTester(reid_model_path, hrnet_path)
        
        # Run full test suite
        results = tester.run_full_test(
            gallery_image_path=gallery_image_path,
            video1_path=video1_path,
            video2_path=video2_path,
            save_annotated=True,
            output_dir="reid_with_mask_test_results"
        )
        
        if results:
            # Generate comparison plots
            tester.generate_comparison_plots(results, "reid_with_mask_test_results")
            
            print("\nTest completed successfully!")
            print(f"Overall accuracy: {results['overall_performance']['overall_accuracy']:.3f}")
            
        else:
            print("Test failed!")
            
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Example usage with custom paths
    print("Example usage:")
    print("python bpbreid_video_test.py")
    print()
    print("Make sure you have the following files:")
    print("1. Gallery image: test_data/gallery_person.jpg")
    print("2. Video 1 (same person): test_data/same_person_video.mp4")
    print("3. Video 2 (different person): test_data/different_person_video.mp4")
    print("4. Model files: pretrained_models/bpbreid_market1501_hrnet32_10642.pth")
    print("5. HRNet weights: pretrained_models/hrnetv2_w32_imagenet_pretrained.pth")
    print()
    
    # Uncomment to run the test
    main()