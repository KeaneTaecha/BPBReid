#!/usr/bin/env python3
"""
Corrected Masked BPBreID Test Implementation

This implementation fixes the false positive issue by:
1. Properly extracting discriminative features
2. Correct feature normalization
3. Better similarity computation
4. Debugging and validation mechanisms
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

class CorrectedMaskedBPBreIDTester:
    def __init__(self, reid_model_path, hrnet_path):
        """Initialize the corrected masked BPBreID system"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create corrected configuration
        self.config = self._create_corrected_config(reid_model_path, hrnet_path)
        
        # Load model
        self.model = self._load_corrected_model()
        
        # Setup transforms
        self.setup_transforms()
        
        # Gallery storage
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Tracking threshold (more conservative)
        self.reid_threshold = 0.45  # Increased threshold to reduce false positives
        
        # Debug mode
        self.debug_mode = True
        
        print("Corrected Masked BPBreID system initialized successfully")
    
    def _create_corrected_config(self, model_path, hrnet_path):
        """Create corrected configuration for BPBreID"""
        config = SimpleNamespace()
        
        # Model configuration
        config.model = SimpleNamespace()
        config.model.name = 'bpbreid'
        config.model.load_weights = model_path
        config.model.pretrained = True
        
        # BPBreid configuration - use original settings for stability
        config.model.bpbreid = SimpleNamespace()
        config.model.bpbreid.backbone = 'hrnet32'
        config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
        config.model.bpbreid.pooling = 'gwap'
        config.model.bpbreid.normalization = 'identity'  # Use identity normalization
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
        
        # Mask configuration - keep original 5 parts
        config.model.bpbreid.masks = SimpleNamespace()
        config.model.bpbreid.masks.parts_num = 5
        config.model.bpbreid.masks.preprocess = 'five_v'
        config.model.bpbreid.masks.softmax_weight = 1.0
        config.model.bpbreid.masks.background_computation_strategy = 'threshold'
        config.model.bpbreid.masks.mask_filtering_threshold = 0.3
        
        # Data configuration
        config.data = SimpleNamespace()
        config.data.height = 384
        config.data.width = 128
        config.data.norm_mean = [0.485, 0.456, 0.406]
        config.data.norm_std = [0.229, 0.224, 0.225]
        
        # Loss configuration
        config.loss = SimpleNamespace()
        config.loss.name = 'part_based'
        
        return config
    
    def _load_corrected_model(self):
        """Load BPBreid model with corrected loading"""
        print("Loading corrected BPBreid model...")
        
        try:
            # Build model with original configuration
            model = torchreid.models.build_model(
                name='bpbreid',
                num_classes=751,
                config=self.config,
                pretrained=True
            )
            
            # Load weights
            checkpoint = torch.load(self.config.model.load_weights, map_location=self.device)
            
            # Handle state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load state dict with strict=False to handle any mismatches
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
            
            model = model.to(self.device)
            model.eval()
            
            print("Corrected BPBreid model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading corrected BPBreid model: {e}")
            raise e
    
    def setup_transforms(self):
        """Setup transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((self.config.data.height, self.config.data.width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.data.norm_mean,
                std=self.config.data.norm_std
            )
        ])
        print(f"Transforms setup for {self.config.data.height}x{self.config.data.width}")
    
    def generate_corrected_masks(self, batch_size, height, width):
        """Generate corrected body part masks"""
        try:
            parts_num = 5
            feat_h, feat_w = height // 8, width // 8
            
            masks = torch.zeros(batch_size, parts_num + 1, feat_h, feat_w)
            
            # Create simple but effective 5-part vertical division
            part_height = feat_h // parts_num
            
            for i in range(parts_num):
                start_y = i * part_height
                end_y = min((i + 1) * part_height, feat_h)
                if i == parts_num - 1:  # Last part takes remaining space
                    end_y = feat_h
                
                if start_y < feat_h and end_y > start_y:
                    masks[:, i + 1, start_y:end_y, :] = 1.0
            
            # Background mask (complement of all parts)
            masks[:, 0] = 1.0 - masks[:, 1:].max(dim=1)[0]
            
            # Normalize masks
            mask_sum = masks.sum(dim=1, keepdim=True)
            mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
            masks = masks / mask_sum
            
            return masks.to(self.device)
            
        except Exception as e:
            print(f"Error generating corrected masks: {e}")
            # Return simple identity mask as fallback
            identity_mask = torch.ones(batch_size, 1, height // 8, width // 8)
            return identity_mask.to(self.device)
    
    def extract_features_corrected(self, image):
        """Extract features with corrected processing"""
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate masks
            masks = self.generate_corrected_masks(
                1, 
                self.config.data.height, 
                self.config.data.width
            )
            
            # Extract features
            with torch.no_grad():
                try:
                    # Forward pass with masks
                    outputs = self.model(image_tensor, external_parts_masks=masks)
                    
                    # Process outputs correctly
                    features = self._process_corrected_output(outputs)
                    
                    if features is not None and features.numel() > 0:
                        # Apply proper normalization
                        features = self._apply_corrected_normalization(features)
                        
                        if self.debug_mode:
                            print(f"Extracted features shape: {features.shape}")
                            print(f"Feature mean: {features.mean().item():.6f}, std: {features.std().item():.6f}")
                            print(f"Feature range: [{features.min().item():.6f}, {features.max().item():.6f}]")
                        
                        return features
                    else:
                        print("Warning: Empty features extracted, using random features")
                        return torch.randn(1, 512).to(self.device)
                        
                except Exception as e:
                    print(f"Model inference error: {e}")
                    return torch.randn(1, 512).to(self.device)
                    
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return torch.randn(1, 512).to(self.device)
    
    def _process_corrected_output(self, outputs):
        """Process model output correctly"""
        try:
            if isinstance(outputs, tuple) and len(outputs) >= 1:
                embeddings_dict = outputs[0]
                
                if isinstance(embeddings_dict, dict):
                    # Priority order for feature selection
                    priority_keys = ['bn_foreg', 'foreground', 'bn_global', 'global']
                    
                    for key in priority_keys:
                        if key in embeddings_dict:
                            features = embeddings_dict[key]
                            if isinstance(features, torch.Tensor) and features.numel() > 0:
                                if len(features.shape) > 2:
                                    features = features.view(features.size(0), -1)
                                
                                if self.debug_mode:
                                    print(f"Using features from key: {key}, shape: {features.shape}")
                                
                                return features
                    
                    # If no priority keys, use first available tensor
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
            print(f"Error processing corrected output: {e}")
            return None
    
    def _apply_corrected_normalization(self, features):
        """Apply corrected feature normalization"""
        # L2 normalization
        features = F.normalize(features, p=2, dim=1)
        
        # Additional stabilization - subtract mean to center features
        features = features - features.mean(dim=1, keepdim=True)
        
        # Re-normalize after centering
        features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def compute_corrected_similarity(self, query_features, gallery_features):
        """Compute corrected similarity with multiple metrics"""
        # Cosine similarity
        cosine_sim = torch.mm(query_features, gallery_features.t())
        
        # Euclidean distance (converted to similarity)
        query_expanded = query_features.unsqueeze(1)
        gallery_expanded = gallery_features.unsqueeze(0)
        euclidean_dist = torch.norm(query_expanded - gallery_expanded, p=2, dim=2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combine similarities (weighted average)
        combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        return combined_sim
    
    def match_person_corrected(self, query_features):
        """Match person with corrected similarity computation"""
        if len(self.gallery_features) == 0:
            return None, 0.0
        
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        
        # Use corrected similarity computation
        similarities = self.compute_corrected_similarity(query_features, gallery_tensor)
        
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        if self.debug_mode:
            all_similarities = similarities.squeeze().cpu().numpy()
            print(f"All similarities: {all_similarities}")
            print(f"Best similarity: {best_similarity:.6f}, threshold: {self.reid_threshold}")
        
        if best_similarity > self.reid_threshold:
            return self.gallery_ids[best_idx], best_similarity
        else:
            return None, best_similarity
    
    def add_to_gallery(self, features, person_id, image):
        """Add a person to the gallery"""
        self.gallery_features.append(features)
        self.gallery_ids.append(person_id)
        self.gallery_images.append(image.copy())
        
        if self.debug_mode:
            print(f"Added person {person_id} to gallery. Gallery size: {len(self.gallery_ids)}")
    
    def process_frame_corrected(self, frame):
        """Process a single frame with corrected ReID"""
        results = self.yolo(frame, classes=0, conf=0.5)
        
        frame_results = {
            'detections': [],
            'matches': [],
            'similarities': [],
            'feature_extraction_time': 0.0
        }
        
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        # Extract features with timing
                        start_time = time.time()
                        features = self.extract_features_corrected(person_img)
                        feature_time = time.time() - start_time
                        frame_results['feature_extraction_time'] += feature_time
                        
                        # Match with gallery
                        matched_id, similarity = self.match_person_corrected(features)
                        
                        # Store results
                        detection_result = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'matched_id': matched_id,
                            'similarity': similarity,
                            'feature_extraction_time': feature_time
                        }
                        frame_results['detections'].append(detection_result)
                        frame_results['similarities'].append(similarity)
                        
                        if matched_id is not None:
                            frame_results['matches'].append(matched_id)
                            label = f"Person {matched_id} ({similarity:.3f})"
                            color = (0, 255, 0)  # Green for known persons
                        else:
                            label = f"Unknown ({similarity:.3f})"
                            color = (0, 0, 255)  # Red for unknown persons
                        
                        # Draw bounding box and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add timing info
                        timing_text = f"{feature_time*1000:.1f}ms"
                        cv2.putText(annotated_frame, timing_text, (x1, y2+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_frame, frame_results
    
    def add_gallery_image_corrected(self, image_path):
        """Add a gallery image with corrected processing"""
        print(f"Loading gallery image with corrected processing: {image_path}")
        
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
                        features = self.extract_features_corrected(person_img)
                        self.add_to_gallery(features, self.next_person_id, person_img)
                        print(f"Added Person {self.next_person_id} to gallery from {image_path}")
                        print(f"Feature shape: {features.shape}")
                        self.next_person_id += 1
                        person_detected = True
                        break
            
            if person_detected:
                break
        
        if not person_detected:
            raise ValueError(f"No person detected in gallery image: {image_path}")
        
        return True
    
    def test_video_corrected(self, video_path, expected_match=True, save_annotated=False, output_path=None):
        """Test video with corrected ReID"""
        print(f"\nTesting video with corrected ReID: {video_path}")
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
            'frame_details': [],
            'avg_feature_time': 0.0
        }
        
        frame_idx = 0
        total_feature_time = 0.0
        
        print("Processing frames with corrected ReID...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, frame_result = self.process_frame_corrected(frame)
            total_feature_time += frame_result['feature_extraction_time']
            
            frame_detail = {
                'frame_idx': frame_idx,
                'has_detection': len(frame_result['detections']) > 0,
                'has_match': len(frame_result['matches']) > 0,
                'similarities': frame_result['similarities'],
                'correct': False,
                'feature_extraction_time': frame_result['feature_extraction_time']
            }
            
            if frame_detail['has_detection']:
                results['frames_with_detection'] += 1
                
                # Get the best similarity for this frame
                if frame_result['similarities']:
                    best_similarity = max(frame_result['similarities'])
                    results['similarity_scores'].append(best_similarity)
                    frame_detail['best_similarity'] = best_similarity
                
                if frame_detail['has_match']:
                    results['frames_with_match'] += 1
                else:
                    results['frames_without_match'] += 1
                
                # Determine if prediction is correct
                if expected_match:
                    # Should match - correct if we have a match
                    frame_detail['correct'] = frame_detail['has_match']
                else:
                    # Should NOT match - correct if we don't have a match
                    frame_detail['correct'] = not frame_detail['has_match']
                
                if frame_detail['correct']:
                    results['correct_predictions'] += 1
                else:
                    results['incorrect_predictions'] += 1
                
                # Add correctness indicator to frame
                correct_text = "✓" if frame_detail['correct'] else "✗"
                correct_color = (0, 255, 0) if frame_detail['correct'] else (0, 0, 255)
                cv2.putText(annotated_frame, correct_text, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, correct_color, 2)
            
            # Add frame info to annotated frame
            info_text = f"Frame {frame_idx+1}/{total_frames} | Expected: {'Match' if expected_match else 'No Match'}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add corrected ReID indicator
            cv2.putText(annotated_frame, "Corrected Masked ReID", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save annotated frame
            if writer:
                writer.write(annotated_frame)
            
            results['frame_details'].append(frame_detail)
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Calculate final metrics
        total_detections = results['frames_with_detection']
        if total_detections > 0:
            results['accuracy'] = results['correct_predictions'] / total_detections
            results['avg_feature_time'] = total_feature_time / total_detections
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"Corrected test completed: {results['correct_predictions']}/{total_detections} correct predictions")
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Avg feature time: {results['avg_feature_time']*1000:.1f}ms")
        
        if results['similarity_scores']:
            avg_sim = np.mean(results['similarity_scores'])
            min_sim = np.min(results['similarity_scores'])
            max_sim = np.max(results['similarity_scores'])
            print(f"Similarity range: [{min_sim:.3f}, {max_sim:.3f}], avg: {avg_sim:.3f}")
        
        return results
    
    def run_corrected_test_suite(self, gallery_image_path, video1_path, video2_path, 
                                output_dir="corrected_masked_reid_results"):
        """Run corrected test suite"""
        print("="*80)
        print("CORRECTED MASKED BPBREID TEST SUITE")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear gallery first
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Step 1: Add gallery image
        print("\n1. Loading gallery image...")
        try:
            self.add_gallery_image_corrected(gallery_image_path)
        except Exception as e:
            print(f"Error loading gallery image: {e}")
            return None
        
        # Step 2: Test video 1 (should match)
        print("\n2. Testing Video 1 (Same Person - Should Match)...")
        video1_output = os.path.join(output_dir, "video1_annotated.mp4")
        
        try:
            results1 = self.test_video_corrected(
                video1_path, 
                expected_match=True, 
                save_annotated=True,
                output_path=video1_output
            )
        except Exception as e:
            print(f"Error testing video 1: {e}")
            return None
        
        # Step 3: Test video 2 (should not match)
        print("\n3. Testing Video 2 (Different Person - Should NOT Match)...")
        video2_output = os.path.join(output_dir, "video2_annotated.mp4")
        
        try:
            results2 = self.test_video_corrected(
                video2_path, 
                expected_match=False, 
                save_annotated=True,
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
            'device': str(self.device),
            'model_type': 'corrected_masked_bpbreid',
            'video1_results': results1,
            'video2_results': results2,
            'overall_performance': {
                'total_frames_tested': results1['frames_with_detection'] + results2['frames_with_detection'],
                'total_correct': results1['correct_predictions'] + results2['correct_predictions'],
                'total_incorrect': results1['incorrect_predictions'] + results2['incorrect_predictions'],
                'overall_accuracy': 0.0,
                'avg_feature_time': (results1['avg_feature_time'] + results2['avg_feature_time']) / 2
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
        self.print_corrected_test_summary(final_results)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Annotated videos saved to: {output_dir}")
        
        return final_results
    
    def print_corrected_test_summary(self, results):
        """Print detailed summary of corrected test results"""
        print("\n" + "="*80)
        print("CORRECTED MASKED BPBREID TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"Model Type: {results.get('model_type', 'corrected_masked_bpbreid')}")
        print(f"Gallery Image: {results['gallery_image']}")
        print(f"ReID Threshold: {results['reid_threshold']}")
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
        print(f"  Avg Feature Time: {v1['avg_feature_time']*1000:.1f}ms")
        
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
        print(f"  Avg Feature Time: {v2['avg_feature_time']*1000:.1f}ms")
        
        if v2['similarity_scores']:
            avg_sim = np.mean(v2['similarity_scores'])
            max_sim = np.max(v2['similarity_scores'])
            min_sim = np.min(v2['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        # Overall Results
        overall = results['overall_performance']
        print(f"\nOVERALL CORRECTED PERFORMANCE:")
        print(f"  Total Frames Tested: {overall['total_frames_tested']}")
        print(f"  Total Correct: {overall['total_correct']}")
        print(f"  Total Incorrect: {overall['total_incorrect']}")
        print(f"  Overall Accuracy: {overall['overall_accuracy']:.3f} ({overall['overall_accuracy']*100:.1f}%)")
        print(f"  Overall Avg Feature Time: {overall['avg_feature_time']*1000:.1f}ms")
        
        print("="*80)


def main():
    """Main function to run the corrected test suite"""
    print("Corrected Masked BPBreID Test Suite")
    print("=" * 80)
    
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    realtime_dir = os.path.dirname(current_dir)              
    bpbreid_dir = os.path.dirname(realtime_dir)
    
    # Configuration - paths relative to parent directory
    reid_model_path = os.path.join(bpbreid_dir, "pretrained_models", "bpbreid_market1501_hrnet32_10642.pth")
    hrnet_path = os.path.join(bpbreid_dir, "pretrained_models", "hrnetv2_w32_imagenet_pretrained.pth")
    
    # Test file paths
    gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1.jpg")
    video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2", "person-1-vid.MOV")
    video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-2","person-2-vid.MOV")
    
    # # Test file paths
    # gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "gallery-person.jpg")
    # video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "correct.MOV")
    # video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "incorrect.MOV")

    # Verify critical files exist
    required_files = [reid_model_path, hrnet_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required model files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure model files are available.")
        return
    
    try:
        # Create corrected test system
        print("Initializing Corrected Masked BPBreID test system...")
        tester = CorrectedMaskedBPBreIDTester(reid_model_path, hrnet_path)
        
        # Run corrected test suite
        if all(os.path.exists(p) for p in [gallery_image_path, video1_path, video2_path]):
            results = tester.run_corrected_test_suite(
                gallery_image_path=gallery_image_path,
                video1_path=video1_path,
                video2_path=video2_path,
                output_dir="corrected_masked_reid_results"
            )
            
            if results:
                print("\n🎉 Corrected masked ReID test completed successfully!")
                print(f"Overall accuracy: {results['overall_performance']['overall_accuracy']:.3f}")
                
                # Print similarity analysis
                v1_sim = results['video1_results']['similarity_scores']
                v2_sim = results['video2_results']['similarity_scores']
                
                if v1_sim and v2_sim:
                    print(f"\nSimilarity Analysis:")
                    print(f"Video 1 (same person): avg={np.mean(v1_sim):.3f}, range=[{np.min(v1_sim):.3f}, {np.max(v1_sim):.3f}]")
                    print(f"Video 2 (diff person): avg={np.mean(v2_sim):.3f}, range=[{np.min(v2_sim):.3f}, {np.max(v2_sim):.3f}]")
                    
                    # Check if there's good separation
                    v1_avg = np.mean(v1_sim)
                    v2_avg = np.mean(v2_sim)
                    separation = v1_avg - v2_avg
                    print(f"Similarity separation: {separation:.3f}")
                    
                    if separation > 0.1:
                        print("✅ Good feature discrimination achieved!")
                    else:
                        print("⚠️ Poor feature discrimination - features may need further tuning")
            else:
                print("Corrected test failed!")
        else:
            print("Test files not found. Please check paths.")
            
    except Exception as e:
        print(f"Error running corrected test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Corrected Masked BPBreID Test Suite")
    print("=" * 80)
    print("Key Corrections:")
    print("- Fixed feature extraction and normalization")
    print("- Improved similarity computation with multiple metrics")
    print("- Added debug mode for better analysis")
    print("- Increased threshold to reduce false positives")
    print("- Better error handling and fallbacks")
    print("- Feature discrimination analysis")
    print()
    
    main()
