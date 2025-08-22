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
from datetime import datetime
import matplotlib.pyplot as plt

class OSNetPersonReIDTester:
    def __init__(self, osnet_model_name='osnet_x1_0'):
        """Initialize the OSNet ReID system for comprehensive testing
        
        Args:
            osnet_model_name: Name of OSNet model variant
                Options: 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25',
                        'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25',
                        'osnet_ibn_x1_0'
        """
        
        # Load YOLO for person detection
        self.yolo = YOLO('yolov8n.pt')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load OSNet model
        self.model_name = osnet_model_name
        self.reid_model = self._load_osnet_model(osnet_model_name)
        self.reid_model.eval()
        
        # Setup transforms for ReID
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Standard size for OSNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Gallery to store person features
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Tracking confidence threshold
        self.reid_threshold = 0.6  # Adjusted for OSNet
        
        # Test results storage
        self.test_results = {}
        
        print(f"OSNet model '{osnet_model_name}' loaded successfully!")
        
    def _load_osnet_model(self, model_name):
        """Load the OSNet model from torchreid"""
        try:
            # Build OSNet model
            model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,  # This will be ignored for feature extraction
                loss='softmax',
                pretrained=True
            )
            
            model = model.to(self.device)
            
            # Print model info
            print(f"Loaded OSNet model: {model_name}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except Exception as e:
            print(f"Error loading OSNet model: {e}")
            print("Available OSNet models: osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25")
            print("                        osnet_ain_x1_0, osnet_ain_x0_75, osnet_ain_x0_5, osnet_ain_x0_25")
            print("                        osnet_ibn_x1_0")
            raise
    
    def extract_features(self, image):
        """Extract ReID features from a person image using OSNet"""
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            # OSNet returns features directly when in eval mode
            features = self.reid_model(image_tensor)
            
            # Ensure features are 2D
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            # L2 normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
        return features
    
    def match_person(self, query_features):
        """Match query features against gallery using cosine similarity"""
        if len(self.gallery_features) == 0:
            return None, 0.0
        
        # Stack gallery features
        gallery_tensor = torch.cat(self.gallery_features, dim=0)
        
        # Compute cosine similarities
        similarities = torch.mm(query_features, gallery_tensor.t())
        
        # Get best match
        best_similarity, best_idx = similarities.max(dim=1)
        best_similarity = best_similarity.item()
        best_idx = best_idx.item()
        
        # Check threshold
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
        results = self.yolo(frame, classes=0, conf=0.5)
        
        frame_results = {
            'detections': [],
            'matches': [],
            'similarities': []
        }
        
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    # Extract person crop
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        # Extract features
                        features = self.extract_features(person_img)
                        
                        # Try to match with gallery
                        matched_id, similarity = self.match_person(features)
                        
                        # Store results for analysis
                        detection_result = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'matched_id': matched_id,
                            'similarity': similarity
                        }
                        frame_results['detections'].append(detection_result)
                        frame_results['similarities'].append(similarity)
                        
                        if matched_id is not None:
                            frame_results['matches'].append(matched_id)
                            label = f"Person {matched_id} ({similarity:.3f})"
                            color = (0, 255, 0)  # Green for matched persons
                        else:
                            label = f"Unknown ({similarity:.3f})"
                            color = (0, 0, 255)  # Red for unknown persons
                        
                        # Draw bounding box and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1),
                                    color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame, frame_results
    
    def add_gallery_image(self, image_path):
        """Add a gallery image from file path"""
        print(f"Loading gallery image: {image_path}")
        
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
                # Get the largest bounding box (assuming it's the main person)
                best_box = None
                best_area = 0
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_box = [x1, y1, x2, y2]
                
                if best_box:
                    x1, y1, x2, y2 = best_box
                    person_img = image[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        features = self.extract_features(person_img)
                        self.add_to_gallery(features, self.next_person_id, person_img)
                        print(f"Added Person {self.next_person_id} to gallery from {image_path}")
                        print(f"  Feature dimension: {features.shape}")
                        self.next_person_id += 1
                        person_detected = True
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
        print(f"Model: {self.model_name}")
        print(f"Threshold: {self.reid_threshold}")
        
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
            'model_name': self.model_name,
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
        processing_times = []
        
        print("Processing frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process frame
            annotated_frame, frame_result = self.process_frame(frame)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            frame_detail = {
                'frame_idx': frame_idx,
                'has_detection': len(frame_result['detections']) > 0,
                'has_match': len(frame_result['matches']) > 0,
                'similarities': frame_result['similarities'],
                'correct': False
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
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, correct_color, 3)
            
            # Add frame info to annotated frame
            info_text = f"Frame {frame_idx+1}/{total_frames} | Expected: {'Match' if expected_match else 'No Match'}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            model_text = f"Model: {self.model_name} | Threshold: {self.reid_threshold:.2f}"
            cv2.putText(annotated_frame, model_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save annotated frame
            if writer:
                writer.write(annotated_frame)
            
            results['frame_details'].append(frame_detail)
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                avg_time = np.mean(processing_times)
                est_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Processed {frame_idx}/{total_frames} frames | Avg FPS: {est_fps:.1f}")
        
        # Calculate final accuracy
        total_detections = results['frames_with_detection']
        if total_detections > 0:
            results['accuracy'] = results['correct_predictions'] / total_detections
        
        # Add timing information
        if processing_times:
            results['avg_processing_time'] = np.mean(processing_times)
            results['avg_fps'] = 1.0 / results['avg_processing_time']
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        print(f"\nTest completed:")
        print(f"  Correct predictions: {results['correct_predictions']}/{total_detections}")
        print(f"  Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        if results['similarity_scores']:
            print(f"  Avg similarity: {np.mean(results['similarity_scores']):.3f}")
        
        return results
    
    def run_full_test(self, gallery_image_path, video1_path, video2_path, 
                     save_annotated=True, output_dir="osnet_test_results"):
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
        print("OSNet ReID Video Test Suite")
        print("="*60)
        
        # Create output directory
        if save_annotated:
            os.makedirs(output_dir, exist_ok=True)
        
        # Clear gallery first
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
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
            'model_name': self.model_name,
            'gallery_image': gallery_image_path,
            'reid_threshold': self.reid_threshold,
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
        
        # Generate comparison plots
        self.generate_comparison_plots(final_results, output_dir)
        
        print(f"\nResults saved to: {results_file}")
        if save_annotated:
            print(f"Annotated videos saved to: {output_dir}")
        
        return final_results
    
    def print_test_summary(self, results):
        """Print a detailed summary of test results"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Model: OSNet ({results['model_name']})")
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
        
        if v1['similarity_scores']:
            avg_sim = np.mean(v1['similarity_scores'])
            max_sim = np.max(v1['similarity_scores'])
            min_sim = np.min(v1['similarity_scores'])
            print(f"  Similarity Scores - Avg: {avg_sim:.3f}, Max: {max_sim:.3f}, Min: {min_sim:.3f}")
        
        if 'avg_fps' in v1:
            print(f"  Processing Speed: {v1['avg_fps']:.1f} FPS")
        
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
        
        if 'avg_fps' in v2:
            print(f"  Processing Speed: {v2['avg_fps']:.1f} FPS")
        
        # Overall Results
        overall = results['overall_performance']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Frames Tested: {overall['total_frames_tested']}")
        print(f"  Total Correct: {overall['total_correct']}")
        print(f"  Total Incorrect: {overall['total_incorrect']}")
        print(f"  Overall Accuracy: {overall['overall_accuracy']:.3f} ({overall['overall_accuracy']*100:.1f}%)")
        
        print("="*60)
    
    def generate_comparison_plots(self, results, output_dir="test_results"):
        """Generate comparison plots for the test results"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'OSNet ({results["model_name"]}) ReID Video Test Results', fontsize=16)
            
            # Plot 1: Accuracy comparison
            videos = ['Video 1\n(Same Person)', 'Video 2\n(Different Person)']
            accuracies = [results['video1_results']['accuracy'], results['video2_results']['accuracy']]
            colors = ['green', 'blue']
            
            axes[0, 0].bar(videos, accuracies, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add accuracy values on bars
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Match/No Match distribution
            v1_match = results['video1_results']['frames_with_match']
            v1_no_match = results['video1_results']['frames_without_match']
            v2_match = results['video2_results']['frames_with_match']
            v2_no_match = results['video2_results']['frames_without_match']
            
            x = np.arange(2)
            width = 0.35
            
            bars1 = axes[0, 1].bar(x - width/2, [v1_match, v2_match], width, label='Matched', color='green', alpha=0.7)
            bars2 = axes[0, 1].bar(x + width/2, [v1_no_match, v2_no_match], width, label='Not Matched', color='red', alpha=0.7)
            
            axes[0, 1].set_ylabel('Number of Frames')
            axes[0, 1].set_title('Match Distribution')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(videos)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}', ha='center', va='bottom')
            
            # Plot 3: Processing Speed Comparison
            if 'avg_fps' in results['video1_results'] and 'avg_fps' in results['video2_results']:
                fps_values = [results['video1_results']['avg_fps'], results['video2_results']['avg_fps']]
                axes[0, 2].bar(videos, fps_values, color=['green', 'blue'], alpha=0.7)
                axes[0, 2].set_ylabel('FPS')
                axes[0, 2].set_title('Processing Speed')
                axes[0, 2].grid(True, alpha=0.3)
                
                for i, v in enumerate(fps_values):
                    axes[0, 2].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Similarity scores distribution for Video 1
            if results['video1_results']['similarity_scores']:
                axes[1, 0].hist(results['video1_results']['similarity_scores'], bins=30, alpha=0.7, color='green', edgecolor='black')
                axes[1, 0].axvline(x=results['reid_threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold ({results["reid_threshold"]})')
                axes[1, 0].set_xlabel('Similarity Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Video 1 Similarity Scores (Should Match)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add statistics
                mean_sim = np.mean(results['video1_results']['similarity_scores'])
                axes[1, 0].axvline(x=mean_sim, color='blue', linestyle=':', linewidth=2, label=f'Mean ({mean_sim:.3f})')
                axes[1, 0].legend()
            
            # Plot 5: Similarity scores distribution for Video 2
            if results['video2_results']['similarity_scores']:
                axes[1, 1].hist(results['video2_results']['similarity_scores'], bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes[1, 1].axvline(x=results['reid_threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold ({results["reid_threshold"]})')
                axes[1, 1].set_xlabel('Similarity Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Video 2 Similarity Scores (Should NOT Match)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add statistics
                mean_sim = np.mean(results['video2_results']['similarity_scores'])
                axes[1, 1].axvline(x=mean_sim, color='green', linestyle=':', linewidth=2, label=f'Mean ({mean_sim:.3f})')
                axes[1, 1].legend()
            
            # Plot 6: Combined similarity scores comparison
            if results['video1_results']['similarity_scores'] and results['video2_results']['similarity_scores']:
                axes[1, 2].boxplot([results['video1_results']['similarity_scores'], 
                                   results['video2_results']['similarity_scores']],
                                  labels=['Video 1\n(Should Match)', 'Video 2\n(Should NOT Match)'])
                axes[1, 2].axhline(y=results['reid_threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold')
                axes[1, 2].set_ylabel('Similarity Score')
                axes[1, 2].set_title('Similarity Score Comparison')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
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
    
    def test_different_thresholds(self, gallery_image_path, video1_path, video2_path, 
                                 thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        """Test performance with different similarity thresholds"""
        print("\n" + "="*60)
        print("THRESHOLD SENSITIVITY ANALYSIS")
        print("="*60)
        
        # Clear and setup gallery
        self.gallery_features = []
        self.gallery_ids = []
        self.gallery_images = []
        self.next_person_id = 1
        
        # Add gallery image
        print("\nLoading gallery image...")
        try:
            self.add_gallery_image(gallery_image_path)
        except Exception as e:
            print(f"Error loading gallery image: {e}")
            return None
        
        results_by_threshold = []
        
        for threshold in thresholds:
            print(f"\nTesting with threshold: {threshold}")
            self.reid_threshold = threshold
            
            # Test video 1
            results1 = self.test_video(video1_path, expected_match=True, save_annotated=False)
            
            # Test video 2
            results2 = self.test_video(video2_path, expected_match=False, save_annotated=False)
            
            # Calculate combined metrics
            total_correct = results1['correct_predictions'] + results2['correct_predictions']
            total_tested = results1['frames_with_detection'] + results2['frames_with_detection']
            overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
            
            results_by_threshold.append({
                'threshold': threshold,
                'video1_accuracy': results1['accuracy'],
                'video2_accuracy': results2['accuracy'],
                'overall_accuracy': overall_accuracy,
                'video1_match_rate': results1['frames_with_match'] / results1['frames_with_detection'] if results1['frames_with_detection'] > 0 else 0,
                'video2_match_rate': results2['frames_with_match'] / results2['frames_with_detection'] if results2['frames_with_detection'] > 0 else 0
            })
        
        # Print summary table
        print("\n" + "="*80)
        print("THRESHOLD ANALYSIS RESULTS")
        print("="*80)
        print(f"{'Threshold':<10} {'V1 Acc':<10} {'V2 Acc':<10} {'Overall':<10} {'V1 Match%':<12} {'V2 Match%':<12}")
        print("-"*80)
        
        best_threshold = None
        best_accuracy = 0
        
        for r in results_by_threshold:
            print(f"{r['threshold']:<10.2f} {r['video1_accuracy']:<10.3f} {r['video2_accuracy']:<10.3f} "
                  f"{r['overall_accuracy']:<10.3f} {r['video1_match_rate']*100:<12.1f} {r['video2_match_rate']*100:<12.1f}")
            
            if r['overall_accuracy'] > best_accuracy:
                best_accuracy = r['overall_accuracy']
                best_threshold = r['threshold']
        
        print("-"*80)
        print(f"Best threshold: {best_threshold} with overall accuracy: {best_accuracy:.3f}")
        
        # Generate threshold analysis plot
        self._plot_threshold_analysis(results_by_threshold)
        
        return results_by_threshold
    
    def _plot_threshold_analysis(self, results):
        """Plot threshold sensitivity analysis"""
        try:
            thresholds = [r['threshold'] for r in results]
            v1_acc = [r['video1_accuracy'] for r in results]
            v2_acc = [r['video2_accuracy'] for r in results]
            overall_acc = [r['overall_accuracy'] for r in results]
            
            plt.figure(figsize=(12, 6))
            
            # Plot accuracies
            plt.subplot(1, 2, 1)
            plt.plot(thresholds, v1_acc, 'g-o', label='Video 1 (Same Person)', linewidth=2)
            plt.plot(thresholds, v2_acc, 'b-s', label='Video 2 (Different Person)', linewidth=2)
            plt.plot(thresholds, overall_acc, 'r-^', label='Overall', linewidth=2)
            plt.xlabel('Similarity Threshold')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            
            # Plot match rates
            plt.subplot(1, 2, 2)
            v1_match = [r['video1_match_rate'] for r in results]
            v2_match = [r['video2_match_rate'] for r in results]
            plt.plot(thresholds, v1_match, 'g-o', label='Video 1 Match Rate', linewidth=2)
            plt.plot(thresholds, v2_match, 'b-s', label='Video 2 Match Rate', linewidth=2)
            plt.xlabel('Similarity Threshold')
            plt.ylabel('Match Rate')
            plt.title('Match Rate vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            
            plt.tight_layout()
            plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Threshold analysis plot saved to: threshold_analysis.png")
            
        except Exception as e:
            print(f"Error generating threshold analysis plot: {e}")
    
    def compare_osnet_variants(self, gallery_image_path, video1_path, video2_path):
        """Compare different OSNet model variants"""
        print("\n" + "="*60)
        print("OSNET VARIANTS COMPARISON")
        print("="*60)
        
        variants = ['osnet_x0_25', 'osnet_x0_5', 'osnet_x0_75', 'osnet_x1_0']
        results_by_variant = []
        
        for variant in variants:
            print(f"\nTesting variant: {variant}")
            
            try:
                # Create new tester with this variant
                tester = OSNetPersonReIDTester(osnet_model_name=variant)
                
                # Add gallery
                tester.add_gallery_image(gallery_image_path)
                
                # Test videos
                results1 = tester.test_video(video1_path, expected_match=True, save_annotated=False)
                results2 = tester.test_video(video2_path, expected_match=False, save_annotated=False)
                
                # Calculate metrics
                total_correct = results1['correct_predictions'] + results2['correct_predictions']
                total_tested = results1['frames_with_detection'] + results2['frames_with_detection']
                overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
                
                avg_fps = (results1.get('avg_fps', 0) + results2.get('avg_fps', 0)) / 2
                
                results_by_variant.append({
                    'variant': variant,
                    'video1_accuracy': results1['accuracy'],
                    'video2_accuracy': results2['accuracy'],
                    'overall_accuracy': overall_accuracy,
                    'avg_fps': avg_fps
                })
                
            except Exception as e:
                print(f"Error testing variant {variant}: {e}")
                continue
        
        # Print comparison table
        print("\n" + "="*80)
        print("OSNET VARIANTS COMPARISON RESULTS")
        print("="*80)
        print(f"{'Variant':<15} {'V1 Acc':<10} {'V2 Acc':<10} {'Overall':<10} {'Avg FPS':<10}")
        print("-"*80)
        
        for r in results_by_variant:
            print(f"{r['variant']:<15} {r['video1_accuracy']:<10.3f} {r['video2_accuracy']:<10.3f} "
                  f"{r['overall_accuracy']:<10.3f} {r['avg_fps']:<10.1f}")
        
        return results_by_variant


def main():
    """Main function to run the OSNet test suite"""
    print("OSNet ReID Video Test Suite")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    realtime_dir = os.path.dirname(current_dir)               
    bpbreid_dir = os.path.dirname(realtime_dir) 
    
    # Test file paths
    gallery_image_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "gallery-person.jpg")
    video1_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1", "correct.MOV")
    video2_path = os.path.join(bpbreid_dir, "datasets", "Compare", "dataset-1","incorrect.MOV")
    
    # Check if using default paths
    if "path/to" in gallery_image_path:
        print("\nPlease update the file paths in the main() function to match your dataset location.")
        print("\nAlternatively, you can run the script with command line arguments:")
        print("python osnet_reid_test.py <gallery_image> <video1> <video2>")
        
        # Try to get paths from command line arguments
        import sys
        if len(sys.argv) == 4:
            gallery_image_path = sys.argv[1]
            video1_path = sys.argv[2]
            video2_path = sys.argv[3]
        else:
            print("\nExample paths structure:")
            print("  gallery_image_path = 'datasets/Compare/dataset-2/person-1.jpg'")
            print("  video1_path = 'datasets/Compare/dataset-2/person-1-vid.MOV'")
            print("  video2_path = 'datasets/Compare/dataset-2/person-2-vid.MOV'")
            return
    
    # Verify files exist
    required_files = [gallery_image_path, video1_path, video2_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("\nMissing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all test files are available.")
        return
    
    try:
        print("\nChoose test mode:")
        print("1. Full test suite with default OSNet model")
        print("2. Test with specific OSNet variant")
        print("3. Threshold sensitivity analysis")
        print("4. Compare all OSNet variants")
        print("5. Quick test (no video saving)")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            # Full test suite with default model
            print("\nInitializing OSNet test system...")
            tester = OSNetPersonReIDTester(osnet_model_name='osnet_x1_0')
            
            results = tester.run_full_test(
                gallery_image_path=gallery_image_path,
                video1_path=video1_path,
                video2_path=video2_path,
                save_annotated=True,
                output_dir="osnet_test_results"
            )
            
            if results:
                print("\nFull test completed successfully!")
            else:
                print("Test failed!")
                
        elif choice == "2":
            # Test with specific variant
            print("\nAvailable OSNet variants:")
            print("  1. osnet_x0_25 (Fastest, least accurate)")
            print("  2. osnet_x0_5")
            print("  3. osnet_x0_75")
            print("  4. osnet_x1_0 (Standard)")
            print("  5. osnet_ain_x1_0 (With AIN)")
            print("  6. osnet_ibn_x1_0 (With IBN)")
            
            variant_choice = input("Enter variant number (1-6): ").strip()
            
            variant_map = {
                '1': 'osnet_x0_25',
                '2': 'osnet_x0_5',
                '3': 'osnet_x0_75',
                '4': 'osnet_x1_0',
                '5': 'osnet_ain_x1_0',
                '6': 'osnet_ibn_x1_0'
            }
            
            variant = variant_map.get(variant_choice, 'osnet_x1_0')
            
            print(f"\nInitializing OSNet ({variant}) test system...")
            tester = OSNetPersonReIDTester(osnet_model_name=variant)
            
            results = tester.run_full_test(
                gallery_image_path=gallery_image_path,
                video1_path=video1_path,
                video2_path=video2_path,
                save_annotated=True,
                output_dir=f"osnet_{variant}_results"
            )
            
        elif choice == "3":
            # Threshold sensitivity analysis
            print("\nInitializing OSNet for threshold analysis...")
            tester = OSNetPersonReIDTester(osnet_model_name='osnet_x1_0')
            
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            results = tester.test_different_thresholds(
                gallery_image_path=gallery_image_path,
                video1_path=video1_path,
                video2_path=video2_path,
                thresholds=thresholds
            )
            
        elif choice == "4":
            # Compare all variants
            print("\nComparing OSNet variants...")
            tester = OSNetPersonReIDTester()  # Just for the comparison method
            
            results = tester.compare_osnet_variants(
                gallery_image_path=gallery_image_path,
                video1_path=video1_path,
                video2_path=video2_path
            )
            
        elif choice == "5":
            # Quick test without saving videos
            print("\nInitializing OSNet for quick test...")
            tester = OSNetPersonReIDTester(osnet_model_name='osnet_x1_0')
            
            results = tester.run_full_test(
                gallery_image_path=gallery_image_path,
                video1_path=video1_path,
                video2_path=video2_path,
                save_annotated=False,
                output_dir="osnet_quick_results"
            )
            
        else:
            print("Invalid choice. Exiting.")
            
    except Exception as e:
        print(f"Error running OSNet test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("OSNet Person Re-Identification Video Testing Suite")
    print("="*60)
    print("\nFeatures:")
    print("- Multiple OSNet model variants support")
    print("- Frame-by-frame video testing")
    print("- Accuracy metrics and performance analysis")
    print("- Similarity score distribution analysis")
    print("- Threshold sensitivity testing")
    print("- Visual results with annotated videos")
    print("- Comprehensive comparison plots")
    print()
    
    main()