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
import json
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# For pose estimation
try:
    import openpifpaf
    PIFPAF_AVAILABLE = True
except ImportError:
    print("Warning: OpenPifPaf not available. Install with: pip install openpifpaf")
    PIFPAF_AVAILABLE = False

class DiagnosticReIDEvaluator:
    def __init__(self, bpbreid_model_path=None, hrnet_path=None, osnet_model_path=None):
        """Initialize evaluation system with enhanced debugging"""
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.bpbreid_model = None
        self.osnet_model = None
        
        # Debug flags
        self.debug_mode = True
        self.save_debug_images = True
        
        # Load models with error handling
        if bpbreid_model_path and hrnet_path:
            print("Loading BPBReID model...")
            try:
                self.bpbreid_model = self._load_bpbreid_model(bpbreid_model_path, hrnet_path)
            except Exception as e:
                print(f"❌ BPBReID loading failed: {e}")
                traceback.print_exc()
                
        print("Loading OSNet model...")
        try:
            self.osnet_model = self._load_osnet_model(osnet_model_path)
        except Exception as e:
            print(f"❌ OSNet loading failed: {e}")
            traceback.print_exc()
        
        # Setup transforms
        self._setup_transforms()
        
        # Initialize pose estimation
        self.pose_predictor = None
        if PIFPAF_AVAILABLE:
            self._setup_pose_estimation()
        
        # Gallery for reference person
        self.gallery_image = None
        self.bpbreid_gallery_features = None
        self.osnet_gallery_features = None
        
        # Evaluation parameters
        self.reid_threshold = 0.5  # Lower threshold for testing
        self.detection_confidence = 0.3  # Lower confidence for more detections
        
        # Results storage with detailed debugging
        self.evaluation_results = {
            'bpbreid': {
                'video1_correct': 0, 'video1_total': 0,
                'video2_correct': 0, 'video2_total': 0,
                'frame_results': {'video1': [], 'video2': []},
                'inference_times': [],
                'errors': [],
                'debug_info': []
            },
            'osnet': {
                'video1_correct': 0, 'video1_total': 0,
                'video2_correct': 0, 'video2_total': 0,
                'frame_results': {'video1': [], 'video2': []},
                'inference_times': [],
                'errors': [],
                'debug_info': []
            }
        }
        
    def _setup_transforms(self):
        """Setup transforms for both models"""
        # BPBReID transforms (with smart padding)
        self.bpbreid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # OSNet transforms (standard resize)
        self.osnet_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _setup_pose_estimation(self):
        """Setup OpenPifPaf for pose estimation"""
        try:
            import openpifpaf
            self.pose_predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
            print("✓ OpenPifPaf pose estimation loaded successfully")
        except Exception as e:
            print(f"❌ Could not load OpenPifPaf: {e}")
            self.pose_predictor = None
    
    def _load_osnet_model(self, model_path=None):
        """Load OSNet model with debugging"""
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading custom OSNet from: {model_path}")
                model = torchreid.models.build_model(
                    name='osnet_x1_0',
                    num_classes=751,
                    pretrained=False
                )
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                model.load_state_dict(state_dict, strict=False)
            else:
                print("Loading pre-trained OSNet...")
                model = torchreid.models.build_model(
                    name='osnet_x1_0',
                    num_classes=751,
                    pretrained=True
                )
            
            model = model.to(self.device)
            model.eval()
            print("✓ OSNet model loaded successfully")
            
            # Test model with dummy input
            dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
            with torch.no_grad():
                dummy_output = model(dummy_input)
                print(f"✓ OSNet test passed - output shape: {dummy_output.shape}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading OSNet: {e}")
            # Try alternative variants
            for variant in ['osnet_x0_75', 'osnet_x0_5']:
                try:
                    print(f"Trying {variant}...")
                    model = torchreid.models.build_model(
                        name=variant,
                        num_classes=751,
                        pretrained=True
                    )
                    model = model.to(self.device)
                    model.eval()
                    print(f"✓ {variant} model loaded successfully")
                    return model
                except:
                    continue
            return None
    
    def _load_bpbreid_model(self, model_path, hrnet_path):
        """Load BPBReID model with enhanced debugging"""
        try:
            print(f"Loading BPBReID from: {model_path}")
            print(f"HRNet path: {hrnet_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"BPBReID model not found: {model_path}")
            if not os.path.exists(hrnet_path):
                raise FileNotFoundError(f"HRNet model not found: {hrnet_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"✓ Checkpoint loaded, keys: {list(checkpoint.keys())}")
            
            from types import SimpleNamespace
            
            # Simplified config for BPBReID
            config = SimpleNamespace()
            config.model = SimpleNamespace()
            config.model.load_weights = model_path
            config.model.load_config = True
            config.model.bpbreid = SimpleNamespace()
            config.model.bpbreid.backbone = 'hrnet32'
            config.model.bpbreid.hrnet_pretrained_path = os.path.dirname(hrnet_path) + '/'
            config.model.bpbreid.learnable_attention_enabled = True
            config.model.bpbreid.mask_filtering_testing = False  # Disable mask filtering for now
            config.model.bpbreid.mask_filtering_training = False
            config.model.bpbreid.test_embeddings = ['bn_foreg']  # Use only foreground features
            config.model.bpbreid.masks = SimpleNamespace()
            config.model.bpbreid.masks.dir = 'pifpaf_maskrcnn_filtering'
            config.model.bpbreid.masks.preprocess = 'five_v'
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
            
            try:
                model = torchreid.models.build_model(
                    name='bpbreid',
                    num_classes=751,
                    config=config,
                    pretrained=True
                )
                print("✓ BPBReID model architecture created")
            except Exception as e:
                print(f"❌ Error creating BPBReID architecture: {e}")
                # Try simplified approach - load as ResNet backbone
                print("Trying simplified ResNet approach...")
                model = torchreid.models.build_model(
                    name='resnet50',
                    num_classes=751,
                    pretrained=True
                )
                print("✓ Using ResNet50 as fallback")
            
            # Load state dict with flexible matching
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {}
            
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_key = k[7:]  # Remove 'module.' prefix
                else:
                    new_key = k
                new_state_dict[new_key] = v
            
            # Load with strict=False to ignore missing keys
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
            model = model.to(self.device)
            model.eval()
            print("✓ BPBReID model loaded successfully")
            
            # Test model with dummy input
            try:
                dummy_input = torch.randn(1, 3, 384, 128).to(self.device)
                with torch.no_grad():
                    dummy_output = model(dummy_input)
                    if isinstance(dummy_output, tuple):
                        dummy_output = dummy_output[0]
                        if isinstance(dummy_output, dict):
                            dummy_output = list(dummy_output.values())[0]
                    print(f"✓ BPBReID test passed - output shape: {dummy_output.shape}")
            except Exception as e:
                print(f"⚠️ BPBReID test failed: {e}")
                print("Model loaded but may have issues during inference")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading BPBReID: {e}")
            traceback.print_exc()
            return None
    
    def preprocess_with_padding(self, image):
        """Smart padding for BPBReID with debugging"""
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            original_shape = image.shape
            h, w = image.shape[:2]
            target_ratio = 1/2
            current_ratio = w/h
            
            if self.debug_mode:
                print(f"🔍 Preprocessing: original {original_shape}, ratio {current_ratio:.2f}")
            
            if current_ratio > target_ratio:
                target_h = int(w / target_ratio)
                pad_h = target_h - h
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, 
                                         cv2.BORDER_CONSTANT, value=[0,0,0])
            else:
                target_w = int(h * target_ratio)
                pad_w = target_w - w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=[0,0,0])
            
            # Resize to BPBReID input size
            image = cv2.resize(image, (128, 384))
            image = cv2.resize(image, (384, 128))
            
            if self.debug_mode:
                print(f"🔍 Preprocessing: final shape {image.shape}")
            
            return image
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            # Fallback: simple resize
            return cv2.resize(image, (384, 128))
    
    def extract_bpbreid_features(self, image, debug_name=""):
        """Extract features using BPBReID with extensive debugging"""
        if self.bpbreid_model is None:
            self.evaluation_results['bpbreid']['errors'].append("Model not loaded")
            return None
            
        start_time = time.time()
        
        try:
            if self.debug_mode:
                print(f"🔍 BPBReID extraction for {debug_name}")
                print(f"🔍 Input image shape: {image.shape}")
            
            # Simplified feature extraction (no masks for now)
            processed_image = self.preprocess_with_padding(image)
            
            # Convert to PIL
            if isinstance(processed_image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = processed_image
            
            # Apply transforms
            image_tensor = self.bpbreid_transform(pil_image).unsqueeze(0).to(self.device)
            
            if self.debug_mode:
                print(f"🔍 Tensor shape: {image_tensor.shape}")
            
            # Extract features
            with torch.no_grad():
                try:
                    outputs = self.bpbreid_model(image_tensor)
                    
                    if self.debug_mode:
                        print(f"🔍 Raw output type: {type(outputs)}")
                        if isinstance(outputs, tuple):
                            print(f"🔍 Tuple length: {len(outputs)}")
                            print(f"🔍 First element type: {type(outputs[0])}")
                        
                    # Handle different output formats
                    if isinstance(outputs, tuple):
                        output_dict = outputs[0]
                        if isinstance(output_dict, dict):
                            # Use the first available feature
                            available_keys = list(output_dict.keys())
                            if self.debug_mode:
                                print(f"🔍 Available keys: {available_keys}")
                            
                            if 'bn_foreg' in output_dict:
                                feature_vector = output_dict['bn_foreg']
                            elif available_keys:
                                feature_vector = output_dict[available_keys[0]]
                            else:
                                raise ValueError("No features in output dict")
                        else:
                            feature_vector = output_dict
                    else:
                        feature_vector = outputs
                    
                    # Ensure proper shape
                    if len(feature_vector.shape) > 2:
                        feature_vector = feature_vector.view(feature_vector.size(0), -1)
                    
                    # Normalize
                    feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
                    
                    end_time = time.time()
                    self.evaluation_results['bpbreid']['inference_times'].append(end_time - start_time)
                    
                    if self.debug_mode:
                        print(f"✓ BPBReID features extracted: {feature_vector.shape}")
                        print(f"✓ Feature norm: {torch.norm(feature_vector).item():.4f}")
                    
                    return feature_vector
                    
                except Exception as e:
                    error_msg = f"Model inference failed: {e}"
                    print(f"❌ {error_msg}")
                    self.evaluation_results['bpbreid']['errors'].append(error_msg)
                    return None
            
        except Exception as e:
            error_msg = f"Feature extraction failed: {e}"
            print(f"❌ BPBReID {error_msg}")
            self.evaluation_results['bpbreid']['errors'].append(error_msg)
            traceback.print_exc()
            return None
    
    def extract_osnet_features(self, image, debug_name=""):
        """Extract features using OSNet with debugging"""
        if self.osnet_model is None:
            self.evaluation_results['osnet']['errors'].append("Model not loaded")
            return None
            
        start_time = time.time()
        
        try:
            if self.debug_mode:
                print(f"🔍 OSNet extraction for {debug_name}")
            
            # Convert to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Apply transforms (resize to 256x128)
            image_tensor = self.osnet_transform(pil_image).unsqueeze(0).to(self.device)
            
            if self.debug_mode:
                print(f"🔍 OSNet tensor shape: {image_tensor.shape}")
            
            # Extract features
            with torch.no_grad():
                feature_vector = self.osnet_model(image_tensor)
            
            # Ensure proper shape and normalize
            if len(feature_vector.shape) > 2:
                feature_vector = feature_vector.view(feature_vector.size(0), -1)
                
            feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)
            
            end_time = time.time()
            self.evaluation_results['osnet']['inference_times'].append(end_time - start_time)
            
            if self.debug_mode:
                print(f"✓ OSNet features extracted: {feature_vector.shape}")
                print(f"✓ Feature norm: {torch.norm(feature_vector).item():.4f}")
            
            return feature_vector
            
        except Exception as e:
            error_msg = f"OSNet extraction failed: {e}"
            print(f"❌ {error_msg}")
            self.evaluation_results['osnet']['errors'].append(error_msg)
            traceback.print_exc()
            return None
    
    def load_gallery_image(self, image_path):
        """Load and process gallery image with debugging"""
        print(f"📸 Loading gallery image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Gallery image not found at {image_path}")
            return False
        
        # Load image
        self.gallery_image = cv2.imread(image_path)
        if self.gallery_image is None:
            print("❌ Could not load gallery image")
            return False
        
        print(f"✓ Gallery image loaded: {self.gallery_image.shape}")
        
        # Save debug image if enabled
        if self.save_debug_images:
            os.makedirs("debug_images", exist_ok=True)
            cv2.imwrite("debug_images/gallery_original.jpg", self.gallery_image)
        
        # Extract features with both models
        success = True
        
        if self.bpbreid_model is not None:
            print("🔄 Extracting BPBReID gallery features...")
            self.bpbreid_gallery_features = self.extract_bpbreid_features(self.gallery_image, "gallery")
            if self.bpbreid_gallery_features is not None:
                print(f"✓ BPBReID gallery features: {self.bpbreid_gallery_features.shape}")
            else:
                print("❌ Failed to extract BPBReID features")
                success = False
        
        if self.osnet_model is not None:
            print("🔄 Extracting OSNet gallery features...")
            self.osnet_gallery_features = self.extract_osnet_features(self.gallery_image, "gallery")
            if self.osnet_gallery_features is not None:
                print(f"✓ OSNet gallery features: {self.osnet_gallery_features.shape}")
            else:
                print("❌ Failed to extract OSNet features")
                success = False
        
        return success
    
    def evaluate_frame(self, frame, video_name, frame_number):
        """Evaluate a single frame with detailed debugging"""
        results = self.yolo(frame, classes=0, conf=self.detection_confidence)
        
        frame_results = {
            'bpbreid': {'detected': False, 'matched': False, 'similarity': 0.0},
            'osnet': {'detected': False, 'matched': False, 'similarity': 0.0}
        }
        
        best_detections = {'bpbreid': None, 'osnet': None}
        detection_count = 0
        
        # Process all detections in frame
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        detection_count += 1
                        debug_name = f"{video_name}_frame{frame_number}_det{detection_count}"
                        
                        if self.debug_mode and frame_number % 50 == 0:  # Debug every 50th frame
                            print(f"🔍 Processing {debug_name}, bbox: ({x1},{y1},{x2},{y2}), conf: {conf:.3f}")
                        
                        # Save debug detection if enabled
                        if self.save_debug_images and frame_number % 100 == 0:
                            cv2.imwrite(f"debug_images/{debug_name}.jpg", person_img)
                        
                        # Test BPBReID
                        if self.bpbreid_model is not None and self.bpbreid_gallery_features is not None:
                            features = self.extract_bpbreid_features(person_img, debug_name)
                            if features is not None:
                                similarity = torch.mm(features, self.bpbreid_gallery_features.t()).item()
                                
                                if self.debug_mode and frame_number % 50 == 0:
                                    print(f"🔍 BPBReID similarity: {similarity:.4f}")
                                
                                if similarity > frame_results['bpbreid']['similarity']:
                                    frame_results['bpbreid'] = {
                                        'detected': True,
                                        'matched': similarity > self.reid_threshold,
                                        'similarity': similarity
                                    }
                                    best_detections['bpbreid'] = (x1, y1, x2, y2, similarity)
                        
                        # Test OSNet
                        if self.osnet_model is not None and self.osnet_gallery_features is not None:
                            features = self.extract_osnet_features(person_img, debug_name)
                            if features is not None:
                                similarity = torch.mm(features, self.osnet_gallery_features.t()).item()
                                
                                if self.debug_mode and frame_number % 50 == 0:
                                    print(f"🔍 OSNet similarity: {similarity:.4f}")
                                
                                if similarity > frame_results['osnet']['similarity']:
                                    frame_results['osnet'] = {
                                        'detected': True,
                                        'matched': similarity > self.reid_threshold,
                                        'similarity': similarity
                                    }
                                    best_detections['osnet'] = (x1, y1, x2, y2, similarity)
        
        # Store results
        for model in ['bpbreid', 'osnet']:
            if frame_results[model]['detected']:
                self.evaluation_results[model][f'{video_name}_total'] += 1
                if frame_results[model]['matched']:
                    self.evaluation_results[model][f'{video_name}_correct'] += 1
            
            # Store frame-level results
            self.evaluation_results[model]['frame_results'][video_name].append({
                'frame': frame_number,
                'detected': frame_results[model]['detected'],
                'matched': frame_results[model]['matched'],
                'similarity': frame_results[model]['similarity']
            })
        
        return frame_results, best_detections
    
    def process_video(self, video_path, video_name, show_progress=True):
        """Process video with enhanced debugging"""
        print(f"\n🎬 Processing {video_name}: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found at {video_path}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS")
        
        frame_number = 0
        last_debug_frame = -1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Evaluate frame
            frame_results, best_detections = self.evaluate_frame(frame, video_name, frame_number)
            
            # Show progress and debug info
            if show_progress and frame_number % 50 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"📈 Progress: {progress:.1f}% - Frame {frame_number}/{total_frames}")
                
                # Show current statistics
                if frame_number > 0:
                    for model in ['bpbreid', 'osnet']:
                        total = self.evaluation_results[model][f'{video_name}_total']
                        correct = self.evaluation_results[model][f'{video_name}_correct']
                        if total > 0:
                            acc = correct / total * 100
                            print(f"   {model}: {correct}/{total} ({acc:.1f}%)")
            
            frame_number += 1
            
            # Limit processing for testing (remove this for full evaluation)
            # if frame_number > 500:  # Process only first 500 frames for debugging
            #     print("⚠️ Limited to 500 frames for debugging")
            #     break
        
        cap.release()
        print(f"✅ Completed processing {video_name}")
        
        # Print final statistics for this video
        print(f"\n📊 {video_name.upper()} FINAL STATS:")
        for model in ['bpbreid', 'osnet']:
            total = self.evaluation_results[model][f'{video_name}_total']
            correct = self.evaluation_results[model][f'{video_name}_correct']
            errors = len(self.evaluation_results[model]['errors'])
            
            if total > 0:
                acc = correct / total * 100
                print(f"   {model}: {correct}/{total} ({acc:.1f}%) - {errors} errors")
            else:
                print(f"   {model}: No detections processed - {errors} errors")
        
        return True
    
    def run_diagnostic_evaluation(self, gallery_image_path, video1_path, video2_path):
        """Run evaluation with comprehensive diagnostics"""
        print("="*60)
        print("🔬 DIAGNOSTIC PERSON REID EVALUATION")
        print("="*60)
        
        print(f"🎯 Evaluation Settings:")
        print(f"   ReID Threshold: {self.reid_threshold}")
        print(f"   Detection Confidence: {self.detection_confidence}")
        print(f"   Debug Mode: {self.debug_mode}")
        print(f"   Save Debug Images: {self.save_debug_images}")
        
        # Load gallery image
        if not self.load_gallery_image(gallery_image_path):
            print("❌ Failed to load gallery image")
            return False
        
        # Process videos
        print(f"\n📹 VIDEO 1 (Same Person): Expected matches")
        success1 = self.process_video(video1_path, 'video1')
        
        print(f"\n📹 VIDEO 2 (Different Person): Expected NO matches")
        success2 = self.process_video(video2_path, 'video2')
        
        if success1 and success2:
            self.generate_diagnostic_report()
            return True
        else:
            print("❌ Evaluation failed!")
            return False
    
    def generate_diagnostic_report(self):
        """Generate detailed diagnostic report"""
        print("\n" + "="*60)
        print("🔬 DIAGNOSTIC EVALUATION RESULTS")
        print("="*60)
        
        for model_name in ['bpbreid', 'osnet']:
            if model_name == 'bpbreid' and self.bpbreid_model is None:
                print(f"\n❌ {model_name.upper()}: Model not loaded")
                continue
            if model_name == 'osnet' and self.osnet_model is None:
                print(f"\n❌ {model_name.upper()}: Model not loaded")
                continue
            
            results = self.evaluation_results[model_name]
            
            print(f"\n🔬 {model_name.upper()} DIAGNOSTIC RESULTS:")
            print("-" * 50)
            
            # Error Analysis
            errors = results['errors']
            if errors:
                print(f"❌ ERRORS ({len(errors)}):")
                for i, error in enumerate(errors[:5]):  # Show first 5 errors
                    print(f"   {i+1}. {error}")
                if len(errors) > 5:
                    print(f"   ... and {len(errors) - 5} more errors")
            else:
                print("✅ NO ERRORS")
            
            # Performance Analysis
            if results['inference_times']:
                avg_time = np.mean(results['inference_times']) * 1000
                fps = 1.0 / np.mean(results['inference_times'])
                print(f"⚡ PERFORMANCE:")
                print(f"   Average inference time: {avg_time:.2f} ms")
                print(f"   Average FPS: {fps:.2f}")
                print(f"   Total inferences: {len(results['inference_times'])}")
            else:
                print("❌ NO PERFORMANCE DATA")
            
            # Video Results
            v1_total = results['video1_total']
            v1_correct = results['video1_correct']
            v2_total = results['video2_total'] 
            v2_correct = results['video2_correct']
            
            print(f"📹 VIDEO RESULTS:")
            if v1_total > 0:
                v1_acc = v1_correct / v1_total * 100
                print(f"   Video 1 (Same Person): {v1_correct}/{v1_total} ({v1_acc:.2f}%)")
            else:
                print(f"   Video 1 (Same Person): No detections processed")
            
            if v2_total > 0:
                v2_acc = (v2_total - v2_correct) / v2_total * 100
                print(f"   Video 2 (Different Person): {v2_total - v2_correct}/{v2_total} ({v2_acc:.2f}%)")
            else:
                print(f"   Video 2 (Different Person): No detections processed")
            
            # Similarity Analysis
            v1_similarities = [r['similarity'] for r in results['frame_results']['video1'] if r['detected']]
            v2_similarities = [r['similarity'] for r in results['frame_results']['video2'] if r['detected']]
            
            if v1_similarities or v2_similarities:
                print(f"📊 SIMILARITY ANALYSIS:")
                if v1_similarities:
                    print(f"   Video 1 similarities: min={min(v1_similarities):.3f}, max={max(v1_similarities):.3f}, avg={np.mean(v1_similarities):.3f}")
                if v2_similarities:
                    print(f"   Video 2 similarities: min={min(v2_similarities):.3f}, max={max(v2_similarities):.3f}, avg={np.mean(v2_similarities):.3f}")
                print(f"   Current threshold: {self.reid_threshold}")
        
        # Recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate recommendations based on diagnostic results"""
        print(f"\n🎯 DIAGNOSTIC RECOMMENDATIONS:")
        print("-" * 50)
        
        # BPBReID Analysis
        if self.bpbreid_model is not None:
            bpb_results = self.evaluation_results['bpbreid']
            if bpb_results['errors']:
                print("🔧 BPBReID Issues:")
                print("   - Model inference errors detected")
                print("   - Try simplifying the model configuration")
                print("   - Check if all required dependencies are installed")
                print("   - Consider using ResNet backbone as fallback")
            
            if bpb_results['video1_total'] == 0:
                print("🔧 BPBReID No Detections:")
                print("   - Feature extraction is completely failing")
                print("   - Check model weights and architecture compatibility")
                print("   - Verify input preprocessing pipeline")
        
        # OSNet Analysis  
        if self.osnet_model is not None:
            osn_results = self.evaluation_results['osnet']
            v1_total = osn_results['video1_total']
            v1_correct = osn_results['video1_correct']
            
            if v1_total > 0 and v1_correct == 0:
                print("🔧 OSNet Similarity Issues:")
                print("   - Model is extracting features but similarities are too low")
                print("   - Consider lowering ReID threshold")
                print("   - Check if gallery image represents the same person properly")
                
                # Analyze similarities
                v1_similarities = [r['similarity'] for r in osn_results['frame_results']['video1'] if r['detected']]
                if v1_similarities:
                    max_sim = max(v1_similarities)
                    avg_sim = np.mean(v1_similarities)
                    print(f"   - Max similarity achieved: {max_sim:.3f}")
                    print(f"   - Average similarity: {avg_sim:.3f}")
                    print(f"   - Current threshold: {self.reid_threshold}")
                    
                    if max_sim < 0.3:
                        print("   - Very low similarities suggest different people or poor feature extraction")
                    elif max_sim < self.reid_threshold:
                        recommended_threshold = max_sim * 0.9
                        print(f"   - Consider threshold: {recommended_threshold:.3f}")
        
        # General Recommendations
        print("🔧 General Recommendations:")
        print("   1. Lower detection confidence if no detections")
        print("   2. Check video quality and person visibility")
        print("   3. Ensure gallery image is clear and well-lit")
        print("   4. Verify that gallery person actually appears in Video 1")
        print("   5. Consider different ReID thresholds for different scenarios")
    
    def run_quick_test(self):
        """Quick test to verify models are working"""
        print("\n🧪 QUICK MODEL TEST")
        print("="*30)
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        
        # Test BPBReID
        if self.bpbreid_model is not None:
            print("Testing BPBReID...")
            features = self.extract_bpbreid_features(dummy_image, "test")
            if features is not None:
                print(f"✅ BPBReID works - output shape: {features.shape}")
            else:
                print("❌ BPBReID failed")
        
        # Test OSNet
        if self.osnet_model is not None:
            print("Testing OSNet...")
            features = self.extract_osnet_features(dummy_image, "test")
            if features is not None:
                print(f"✅ OSNet works - output shape: {features.shape}")
            else:
                print("❌ OSNet failed")


def main():
    print("🔬 Diagnostic Person ReID Evaluation System")
    print("="*50)
    
    # Model paths
    bpbreid_model_path = "pretrained_models/bpbreid_market1501_hrnet32_10642.pth"
    hrnet_path = "pretrained_models/hrnetv2_w32_imagenet_pretrained.pth"
    osnet_model_path = None
    
    # Check if BPBReID models exist
    bpbreid_available = (os.path.exists(bpbreid_model_path) and os.path.exists(hrnet_path))
    
    if not bpbreid_available:
        print("⚠️ BPBReID models not found. Running OSNet-only diagnostic.")
        print("To enable BPBReID comparison, ensure these files exist:")
        print(f"  - {bpbreid_model_path}")
        print(f"  - {hrnet_path}")
        print()
        bpbreid_model_path = None
        hrnet_path = None
    
    try:
        # Create diagnostic evaluator
        evaluator = DiagnosticReIDEvaluator(
            bpbreid_model_path=bpbreid_model_path,
            hrnet_path=hrnet_path,
            osnet_model_path=osnet_model_path
        )
        
        print(f"\n✅ Models Status:")
        print(f"   BPBReID: {'✅ Loaded' if evaluator.bpbreid_model is not None else '❌ Failed'}")
        print(f"   OSNet: {'✅ Loaded' if evaluator.osnet_model is not None else '❌ Failed'}")
        
        if evaluator.osnet_model is None and evaluator.bpbreid_model is None:
            print("❌ No models loaded successfully!")
            return
        
        # Run quick test first
        evaluator.run_quick_test()
        
        # Choose evaluation mode
        print(f"\n🎯 Select diagnostic mode:")
        print("1. Full diagnostic evaluation (with videos)")
        print("2. Quick model test only")
        print("3. Interactive troubleshooting")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Get input files
            while True:
                gallery_path = input("\n📸 Enter path to gallery image: ").strip().strip('"')
                if os.path.exists(gallery_path):
                    break
                print(f"❌ File not found: {gallery_path}")
            
            while True:
                video1_path = input("\n🎬 Enter path to Video 1 (SAME person): ").strip().strip('"')
                if os.path.exists(video1_path):
                    break
                print(f"❌ File not found: {video1_path}")
            
            while True:
                video2_path = input("\n🎬 Enter path to Video 2 (DIFFERENT person): ").strip().strip('"')
                if os.path.exists(video2_path):
                    break
                print(f"❌ File not found: {video2_path}")
            
            # Adjust settings for debugging
            print(f"\n⚙️ Current Settings:")
            print(f"   ReID Threshold: {evaluator.reid_threshold}")
            print(f"   Detection Confidence: {evaluator.detection_confidence}")
            print(f"   Debug Mode: {evaluator.debug_mode}")
            
            adjust = input("Adjust settings? (y/n): ").strip().lower()
            if adjust == 'y':
                try:
                    new_threshold = float(input(f"ReID threshold ({evaluator.reid_threshold}): ") or evaluator.reid_threshold)
                    evaluator.reid_threshold = new_threshold
                    
                    new_conf = float(input(f"Detection confidence ({evaluator.detection_confidence}): ") or evaluator.detection_confidence)
                    evaluator.detection_confidence = new_conf
                    
                    debug_choice = input("Enable debug mode? (y/n): ").strip().lower()
                    evaluator.debug_mode = debug_choice == 'y'
                    
                    save_choice = input("Save debug images? (y/n): ").strip().lower()
                    evaluator.save_debug_images = save_choice == 'y'
                    
                except ValueError:
                    print("Invalid input, using current settings")
            
            # Run diagnostic evaluation
            print(f"\n🚀 Starting diagnostic evaluation...")
            success = evaluator.run_diagnostic_evaluation(gallery_path, video1_path, video2_path)
            
        elif choice == "2":
            print("✅ Quick test completed!")
            
        elif choice == "3":
            print("🔧 Interactive Troubleshooting:")
            print("1. Check if videos have detectable persons")
            print("2. Test gallery image feature extraction")
            print("3. Test similarity computation")
            print("4. Adjust thresholds")
            
            # Add interactive troubleshooting steps here
            sub_choice = input("Select troubleshooting step (1-4): ").strip()
            if sub_choice == "1":
                video_path = input("Enter video path to check: ").strip().strip('"')
                if os.path.exists(video_path):
                    # Quick detection test
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    if ret:
                        results = evaluator.yolo(frame, classes=0, conf=0.3)
                        detection_count = 0
                        for r in results:
                            if r.boxes is not None:
                                detection_count += len(r.boxes)
                        print(f"🔍 Found {detection_count} person detections in first frame")
                    cap.release()
        else:
            print("❌ Invalid choice!")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()