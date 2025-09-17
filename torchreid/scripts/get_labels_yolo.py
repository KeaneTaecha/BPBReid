import argparse
import glob
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset

import sys
from pathlib import Path

# Add the parent directory to sys.path to import torchreid modules
sys.path.append(str(Path(__file__).parent.parent))
from torchreid.utils.yolo_pose_masks import YOLOPoseMaskGenerator


def get_image_paths(source, path_format=False):
    """
    Get the paths of all image files in a directory.

    Args:
        source (str): Directory path.
        path_format (bool, optional): Return paths as Path objects if True, otherwise as strings. Default is False.

    Returns:
        image_paths (List[str or Path]): List of image file paths.
    """
    image_paths = glob.glob(f"{source}/**/*.[jJ][pP][gG]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[pP][nN][gG]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[jJ][pP][eE][gG]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[tT][iI][fF]", recursive=True) + \
                  glob.glob(f"{source}/**/*.[tT][iI][fF][fF]", recursive=True)
    if path_format:
        image_paths = [Path(path_str) for path_str in image_paths]
    return image_paths


def format_path(img_path, dataset_dir):
    """
    Formats the given image path based on the dataset directory.

    Args:
        img_path (str): The path of the image file.
        dataset_dir (str): The directory path of the dataset.

    Returns:
        str: The formatted path of the image file.
    """
    if "occluded_reid" in dataset_dir.lower() or "occluded-reid" in dataset_dir.lower():
        return os.path.join(os.path.basename(os.path.dirname(os.path.dirname(img_path))), os.path.basename(img_path))
    elif "p-dukemtmc_reid" in dataset_dir.lower() or "p-dukemtmc-reid" in dataset_dir.lower():
        return os.path.join(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path)))),
                            os.path.basename(os.path.dirname(os.path.dirname(img_path))), os.path.basename(img_path))
    return os.path.relpath(img_path, dataset_dir)


def get_label_paths(img_paths, dataset_dir):
    """
    Get the paths of label files corresponding to the image paths.

    Args:
        img_paths (List[str]): List of image file paths.
        dataset_dir (str): Directory path of the dataset.

    Returns:
        relative_paths (List[str]): List of relative paths of the image files.
        file_paths (List[str]): List of label file paths.
    """
    relative_paths, file_paths = [], []
    for img_name in img_paths:
        relative_path = format_path(img_name, dataset_dir)
        file_path = os.path.join(dataset_dir, "masks", "yolo_pose", relative_path + ".npy")
        relative_paths.append(relative_path)
        file_paths.append(file_path)
    return relative_paths, file_paths


def skip_existing(imagery, dataset_dir):
    """
    Filter out image paths for which label files already exist.

    Args:
        imagery (List[str]): List of image file paths.
        dataset_dir (str): Directory path of the dataset.

    Returns:
        new_imagery (List[str]): List of image file paths for which label files do not exist.
    """
    relative_paths, file_paths = get_label_paths(img_paths=imagery, dataset_dir=dataset_dir)
    new_imagery = []
    for index, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            new_imagery.append(imagery[index])
    return new_imagery


def save_files(files, files_path, verbose=True):
    """
    Save files to specified paths.

    Args:
        files (List[object]): List of files to be saved.
        files_path (List[str]): List of paths where files will be saved.
        verbose (bool, optional): Print progress if True. Default is True.
    """
    for file, file_path in zip(files, files_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, file)
        if verbose:
            print(f"Processed {os.path.basename(file_path)}")


class ImageDataset(Dataset):
    """
    Custom dataset class for loading images.

    Args:
        imagery (List[Path]): List of image file paths.

    Returns:
        (str, np.ndarray): Tuple containing the image file path and the loaded image.
    """

    def __init__(self, imagery: List[Path]):
        self.imagery = imagery

    def __getitem__(self, index):
        return self.imagery[index], cv2.imread(str(self.imagery[index]))

    def __len__(self):
        return len(self.imagery)


# YOLOPoseMaskGenerator is now imported from torchreid.utils.yolo_pose_masks


class BatchYOLOPose:
    def __init__(self, yolo_model_path: str = "yolov8n-pose.pt", batch_size: int = 1, workers: int = 0):
        """
        Initialize the BatchYOLOPose class for performing batched YOLO pose estimation.

        Args:
            yolo_model_path (str): Path to YOLO model weights.
            batch_size (int, optional): Batch size for processing images. Defaults to 1.
            workers (int, optional): Number of worker processes for data loading. Defaults to 0.
        """
        print(f"* YOLO Pose model -> {yolo_model_path}")
        
        # Load YOLO model
        print(f"Loading YOLO model: {yolo_model_path}")
        yolo_model = YOLO(yolo_model_path)
        
        # Initialize YOLO pose mask generator with correct parameters
        self.mask_generator = YOLOPoseMaskGenerator(
            yolo_model=yolo_model,
            keypoint_confidence_threshold=0.5,
            height=384,
            width=128
        )
        
        # Print device information for BatchYOLOPose
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"* BatchYOLOPose Device: {str(device).upper()}")
        
        # Set the batch size for processing images
        self.batch_size = batch_size
        
        # Set the number of worker processes for data loading
        self.workers = workers
        
        # Timing statistics
        self.timing_stats = {
            'total_time': 0.0,
            'yolo_inference_time': 0.0,
            'mask_generation_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': [],
            'per_frame_times': []  # New: track time per individual frame
        }

    def __call__(self, imagery: List[Path] or List[str], dataset_dir: List[Path] or List[str],
                 is_overwrite: bool = False, verbose: bool = False):
        """
        Perform the batch processing of imagery to generate and save YOLO pose mask files.

        Args:
            imagery (List[Path] or List[str]): A list of image paths or image filenames.
            dataset_dir (List[Path] or List[str]): A list of dataset directories.
            is_overwrite (bool, optional): Whether to overwrite existing mask files. Defaults to False.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        """
        assert len(imagery) > 0, "No images found in imagery."

        if not is_overwrite:
            # Skip existing images if overwrite is disabled
            imagery = skip_existing(imagery, dataset_dir)

        # Create an instance of the ImageDataset class
        dataset = ImageDataset(imagery)

        # Create a data loader for batch processing
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True
        )

        total_batches = len(loader)
        progress_bar = tqdm.tqdm(total=total_batches, desc="Processing YOLO Pose", unit="batch")
        
        # Reset timing stats
        self.timing_stats = {
            'total_time': 0.0,
            'yolo_inference_time': 0.0,
            'mask_generation_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': [],
            'per_frame_times': []  # Add this missing key
        }

        with torch.no_grad():
            for batch_idx, (paths, images) in enumerate(loader):
                batch_start_time = time.time()

                # Get the file paths for saving the mask files
                relative_paths, mask_file_paths = get_label_paths(img_paths=paths, dataset_dir=dataset_dir)
                
                # Process each image in the batch
                batch_masks = []
                for img_path, image in zip(paths, images):
                    if image is None:
                        print(f"Warning: Could not load image {img_path}")
                        # Create empty mask for failed images
                        empty_mask = np.zeros((1, 6, 48, 16))  # Standard BPBreID mask size
                        batch_masks.append(empty_mask)
                        continue
                    
                    # Time YOLO pose inference and mask generation per frame
                    frame_start = time.time()
                    masks = self.mask_generator.generate_masks(image, device=None)
                    frame_end = time.time()
                    frame_time = frame_end - frame_start
                    
                    # Record per-frame timing
                    self.timing_stats['per_frame_times'].append(frame_time)
                    
                    if masks is None:
                        print(f"Warning: No pose detected in {img_path} (took {frame_time:.3f}s)")
                        # Create empty mask for images without pose
                        empty_mask = np.zeros((1, 6, 48, 16))  # Standard BPBreID mask size
                        batch_masks.append(empty_mask)
                    else:
                        print(f"Successfully processed {os.path.basename(img_path)} (took {frame_time:.3f}s)")
                        batch_masks.append(masks)
                    
                    # Update timing stats
                    self.timing_stats['yolo_inference_time'] += frame_time

                # Time saving
                save_start = time.time()
                save_files(batch_masks, mask_file_paths, verbose)
                save_end = time.time()
                save_time = save_end - save_start
                
                batch_end_time = time.time()
                batch_total_time = batch_end_time - batch_start_time
                
                # Update timing stats
                self.timing_stats['total_time'] += batch_total_time
                self.timing_stats['saving_time'] += save_time
                self.timing_stats['num_images'] += len(paths)
                
                # Record per-batch timing
                per_image_time = batch_total_time / len(paths)
                self.timing_stats['per_image_times'].extend([per_image_time] * len(paths))
                
                if verbose:
                    print(f"Batch {batch_idx + 1}/{total_batches}: "
                          f"Save: {save_time:.3f}s, "
                          f"Total: {batch_total_time:.3f}s "
                          f"({per_image_time:.3f}s per image)")

                progress_bar.update(1)

            progress_bar.close()
            
        # Print final timing statistics
        self._print_timing_stats("YOLO Pose Processing")
    
    def _print_timing_stats(self, stage_name):
        """Print detailed timing statistics"""
        stats = self.timing_stats
        print(f"\n=== {stage_name} Timing Statistics ===")
        print(f"Total images processed: {stats['num_images']}")
        print(f"Total time: {stats['total_time']:.3f}s")
        print(f"Average time per image: {stats['total_time']/stats['num_images']:.3f}s")
        print(f"YOLO inference time: {stats['yolo_inference_time']:.3f}s ({stats['yolo_inference_time']/stats['total_time']*100:.1f}%)")
        print(f"Saving time: {stats['saving_time']:.3f}s ({stats['saving_time']/stats['total_time']*100:.1f}%)")
        
        # Per-frame timing statistics
        if stats['per_frame_times']:
            print(f"\nPer-frame timing statistics:")
            print(f"  Average time per frame: {np.mean(stats['per_frame_times']):.3f}s")
            print(f"  Min time per frame: {min(stats['per_frame_times']):.3f}s")
            print(f"  Max time per frame: {max(stats['per_frame_times']):.3f}s")
            print(f"  Std dev per frame: {np.std(stats['per_frame_times']):.3f}s")
            print(f"  Frames per second: {1.0/np.mean(stats['per_frame_times']):.2f}")
        
        if stats['per_image_times']:
            print(f"\nPer-batch timing statistics:")
            print(f"  Min time per image: {min(stats['per_image_times']):.3f}s")
            print(f"  Max time per image: {max(stats['per_image_times']):.3f}s")
            print(f"  Std dev per image: {np.std(stats['per_image_times']):.3f}s")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Source dataset containing image files')
    parser.add_argument('--yolo-model-path', type=str, default="yolov8n-pose.pt",
                        help='Path to YOLO model weights')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size for processing images')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    parser.add_argument('--single-image', type=str,
                        help='Process only a single image for timing measurement')
    args = parser.parse_args()

    # Print device information
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"* Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"* CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"* CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Get image paths
    if args.single_image:
        # Process only a single image for timing measurement
        if os.path.exists(args.single_image):
            img_paths = [args.single_image]
            print(f"Processing single image: {args.single_image}")
        else:
            print(f"Error: Image file {args.single_image} not found!")
            return
    else:
        img_paths = get_image_paths(args.source)
        print(f"Found {len(img_paths)} images to process")

    # Overall timing
    total_start_time = time.time()
    
    # Perform YOLO Pose processing
    print("\n" + "="*50)
    print("YOLO POSE MASK GENERATION")
    print("="*50)
    print("Generating 5-part body masks:")
    print("1. Head")
    print("2. Upper body (upper half of torso + upper arms)")
    print("3. Lower body (lower half of torso + lower arms)")
    print("4. Upper legs (thighs and upper calf - stops at 75% to ankle)")
    print("5. Foot (lower calf from 75% + ankle + foot area)")
    print("="*50)
    
    yolo_model = BatchYOLOPose(yolo_model_path=args.yolo_model_path,
                               batch_size=args.batch_size,
                               workers=args.num_workers)
    yolo_model(imagery=img_paths, dataset_dir=args.source, is_overwrite=False, verbose=True)
    
    # Overall timing summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "="*50)
    print("OVERALL TIMING SUMMARY")
    print("="*50)
    print(f"Total processing time: {total_time:.3f}s")
    if args.single_image:
        print(f"Time for single image: {total_time:.3f}s")
    else:
        print(f"Average time per image: {total_time/len(img_paths):.3f}s")
        print(f"Images per second: {len(img_paths)/total_time:.2f}")
    
    print(f"\nMasks saved to: {os.path.join(args.source, 'masks', 'yolo_pose')}")


if __name__ == '__main__':
    main()
