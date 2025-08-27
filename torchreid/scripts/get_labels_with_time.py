import argparse
import glob
import os
import time
from pathlib import Path
from typing import List

import cv2
import detectron2.data.transforms as T
import numpy as np
import openpifpaf
import torch
import tqdm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.model_zoo import get_checkpoint_url, get_config_file
from detectron2.modeling import build_model
from detectron2.structures import Instances
from torch.utils.data import DataLoader, Dataset


def build_config_maskrcnn(model_config_name):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_filename=get_config_file(model_config_name))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(model_config_name)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


def compare_arrays(array1, array2):
    """
    Compare two arrays and calculate Mean Absolute Error (MAE) and percentage difference.

    Args:
        array1 (np.ndarray): First array.
        array2 (np.ndarray): Second array.

    Returns:
        mae (float): Mean Absolute Error (MAE) between the arrays.
        mae_percentage (float): Percentage difference between the arrays.
    """

    def calculate_mae(array1, array2):
        # Calculate Mean Absolute Error (MAE)
        mae = np.mean(np.abs(array1 - array2))
        mae_percentage = (mae / np.max(array1)) * 100
        return mae, mae_percentage

    print(f"Average percentage difference: {calculate_mae(array1, array2)[1]}%")


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


def get_label_paths(is_mask, img_paths, dataset_dir):
    """
    Get the paths of label files corresponding to the image paths.

    Args:
        is_mask (bool): Indicates if the label is a mask or not.
        img_paths (List[str]): List of image file paths.
        dataset_dir (str): Directory path of the dataset.

    Returns:
        relative_paths (List[str]): List of relative paths of the image files.
        file_paths (List[str]): List of label file paths.
    """
    relative_paths, file_paths = [], []
    for img_name in img_paths:
        relative_path = format_path(img_name, dataset_dir)
        if not is_mask:
            file_path = os.path.join(dataset_dir, "masks", "pifpaf", relative_path + ".confidence_fields.npy")
        else:
            file_path = os.path.join(dataset_dir, "masks", "pifpaf_maskrcnn_filtering", relative_path + ".npy")
        relative_paths.append(relative_path)
        file_paths.append(file_path)
    return relative_paths, file_paths


def skip_existing(is_mask, imagery, dataset_dir):
    """
    Filter out image paths for which label files already exist.

    Args:
        is_mask (bool): Indicates if the label is a mask or not.
        imagery (List[str]): List of image file paths.
        dataset_dir (str): Directory path of the dataset.

    Returns:
        new_imagery (List[str]): List of image file paths for which label files do not exist.
    """
    relative_paths, file_paths = get_label_paths(is_mask=is_mask, img_paths=imagery, dataset_dir=dataset_dir)
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


class BatchPifPaf:
    def __init__(self, model_name: str = "shufflenetv2k16", batch_size: int = None, workers: int = None):
        """
        Initializes a BatchPifPaf object.

        Args:
            model_name (str): Name of the OpenPifPaf model to use.
            batch_size (int): Batch size for inference.
            workers (int): Number of workers for data loading.
        """
        models = [
            'resnet50',
            'shufflenetv2k16',
            'shufflenetv2k30',
        ]
        assert model_name in models, f"Model name must be one of {models}"

        print(f"* OpenPifPaf model ->  {model_name}")
        # Define the OpenPifPaf model
        self.model = openpifpaf.Predictor(checkpoint=model_name, visualize_image=True, visualize_processed_image=True)
        self.batch_size = batch_size if batch_size else self.model.batch_size
        self.workers = workers if workers else self.model.loader_workers if self.model.loader_workers is not None else 0
        self.__collate = openpifpaf.datasets.collate_images_anns_meta
        
        # Timing statistics
        self.timing_stats = {
            'total_time': 0.0,
            'pifpaf_inference_time': 0.0,
            'confidence_extraction_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': []
        }

    def __call__(self, imagery: List[Path] or List[str], dataset_dir: List[Path] or List[str],
                 is_overwrite: bool = False, verbose: bool = False):
        """
        Perform batch processing on the given imagery using the OpenPifPaf model.

        Args:
            imagery (List[Path] or List[str]): List of image paths or image file names.
            dataset_dir (List[Path] or List[str]): List of dataset directories.
            is_overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        Yields:
            torch.Tensor: Predictions for each image as a NumPy array.
        """

        assert len(imagery) > 0, "No images found in imagery."

        if not is_overwrite:
            imagery = skip_existing(False, imagery, dataset_dir)

        dataset = openpifpaf.datasets.ImageList(
            imagery,
            preprocess=self.model.preprocess,
            with_raw_image=True
        )
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            pin_memory=self.model.device.type != 'cpu',
            num_workers=self.workers,
            collate_fn=self.__collate,
        )

        total_batches = len(loader)
        progress_bar = tqdm.tqdm(total=total_batches, desc="Processing PifPaf", unit="batch")
        
        # Reset timing stats
        self.timing_stats = {
            'total_time': 0.0,
            'pifpaf_inference_time': 0.0,
            'confidence_extraction_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch_start_time = time.time()
                
                if len(batch) == 3:
                    processed_image_batch, gt_anns_batch, meta_batch = batch
                elif len(batch) == 4:
                    image_batch, processed_image_batch, gt_anns_batch, meta_batch = batch

                # Specify the file path where you want to save the .npy file
                relative_paths, file_paths = get_label_paths(False, [d["file_name"] for d in meta_batch], dataset_dir)

                # Time PifPaf inference
                pifpaf_start = time.time()
                pifpaf_conf: torch.Tensor = self.__get_pifpaf_conf(processed_image_batch)
                pifpaf_end = time.time()
                pifpaf_time = pifpaf_end - pifpaf_start
                
                # Time saving
                save_start = time.time()
                save_files(pifpaf_conf.numpy(), file_paths, verbose)
                save_end = time.time()
                save_time = save_end - save_start
                
                batch_end_time = time.time()
                batch_total_time = batch_end_time - batch_start_time
                
                # Update timing stats
                self.timing_stats['total_time'] += batch_total_time
                self.timing_stats['pifpaf_inference_time'] += pifpaf_time
                self.timing_stats['saving_time'] += save_time
                self.timing_stats['num_images'] += len(meta_batch)
                
                # Record per-batch timing
                per_image_time = batch_total_time / len(meta_batch)
                self.timing_stats['per_image_times'].extend([per_image_time] * len(meta_batch))
                
                if verbose:
                    print(f"Batch {batch_idx + 1}/{total_batches}: "
                          f"PifPaf: {pifpaf_time:.3f}s, "
                          f"Save: {save_time:.3f}s, "
                          f"Total: {batch_total_time:.3f}s "
                          f"({per_image_time:.3f}s per image)")

                progress_bar.update(1)

            progress_bar.close()
            
        # Print final timing statistics
        self._print_timing_stats("PifPaf Processing")

    def __get_pifpaf_conf(self, processed_image_batch: Instances):
        """
        Get the confidence scores from the processed image batch.

        Args:
            processed_image_batch (Instances): Processed image batch containing pose estimation fields.

        Returns:
            torch.Tensor: Confidence scores for keypoints and connections.
        """
        # Retrieve the pose estimation fields from the model processor
        fields_batch = self.model.processor.fields_batch(self.model.model, processed_image_batch,
                                                         device=self.model.device)

        # Extract the pif (keypoint) and paf (connection) fields from the batch
        pif, paf = zip(*fields_batch)

        # Extract the confidence scores for keypoints (index 1 in each field)
        pif_confidence_scores = torch.stack(pif)[:, :, 1]
        paf_confidence_scores = torch.stack(paf)[:, :, 1]

        # Concatenate the confidence scores for keypoints and connections
        pifpaf_confidence_scores = torch.cat((pif_confidence_scores, paf_confidence_scores), dim=1)

        # Return the concatenated confidence scores
        return pifpaf_confidence_scores
    
    def _print_timing_stats(self, stage_name):
        """Print detailed timing statistics"""
        stats = self.timing_stats
        print(f"\n=== {stage_name} Timing Statistics ===")
        print(f"Total images processed: {stats['num_images']}")
        print(f"Total time: {stats['total_time']:.3f}s")
        print(f"Average time per image: {stats['total_time']/stats['num_images']:.3f}s")
        print(f"PifPaf inference time: {stats['pifpaf_inference_time']:.3f}s ({stats['pifpaf_inference_time']/stats['total_time']*100:.1f}%)")
        print(f"Saving time: {stats['saving_time']:.3f}s ({stats['saving_time']/stats['total_time']*100:.1f}%)")
        if stats['per_image_times']:
            print(f"Min time per image: {min(stats['per_image_times']):.3f}s")
            print(f"Max time per image: {max(stats['per_image_times']):.3f}s")
            print(f"Std dev per image: {np.std(stats['per_image_times']):.3f}s")


class BatchMask:
    def __init__(self, cfg: CfgNode or str, batch_size: int = None, workers: int = None):
        """
        Initialize the BatchMask class for performing batched instance segmentation using a Mask R-CNN model.

        Args:
            cfg (CfgNode or str): Configuration options for the Mask R-CNN model.
            batch_size (int, optional): Batch size for processing images. Defaults to None.
            workers (int, optional): Number of worker processes for data loading. Defaults to None.
        """
        # Clone the provided configuration or get a default configuration
        self.cfg = build_config_maskrcnn(cfg) if isinstance(cfg, str) else cfg.clone()
        print(f"* MaskRCNN model ->  {cfg if isinstance(cfg, str) else self.cfg.MODEL.WEIGHTS}")

        # Set the batch size for processing images, defaulting to 32 if not provided
        self.batch_size = batch_size if batch_size else 32

        # Set the number of worker processes for data loading, defaulting to the number of CPU cores
        self.workers = workers if workers is not None else 0

        # Build the Mask R-CNN model
        self.model = build_model(self.cfg)

        # Set the model to evaluation mode
        self.model.eval()

        # Load the pre-trained weights for the model
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        # Define the augmentation transform for resizing images
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )

        # Set the input image format to RGB or BGR based on the configuration
        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        
        # Timing statistics
        self.timing_stats = {
            'total_time': 0.0,
            'maskrcnn_inference_time': 0.0,
            'filtering_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': []
        }

    def __collate(self, batch):
        """
        Collates a batch of images and their paths for use in data loading.

        Args:
            batch (list): A list of tuples containing image paths and corresponding images.

        Returns:
            tuple: A tuple containing two lists: the paths of the images and the processed data.

        """
        paths, data = [], []
        for path, image in batch:
            if self.input_format == "RGB":
                # Convert image format from RGB to BGR if required by the model
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            # Apply augmentation and transformation to the image
            image = self.aug.get_transform(image).apply_image(image)
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
            paths.append(path)
        return paths, data

    def __call__(self, imagery: List[Path] or List[str], dataset_dir: List[Path] or List[str],
                 is_overwrite: bool = False, verbose: bool = False):
        """
        Perform the batch processing of imagery to generate and save mask files.

        Args:
            imagery (List[Path] or List[str]): A list of image paths or image filenames.
            dataset_dir (List[Path] or List[str]): A list of dataset directories.
            is_overwrite (bool, optional): Whether to overwrite existing mask files. Defaults to False.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.

        """
        assert len(imagery) > 0, "No images found in imagery."

        if not is_overwrite:
            # Skip existing images if overwrite is disabled
            imagery = skip_existing(True, imagery, dataset_dir)

        # Create an instance of the ImageDataset class
        dataset = ImageDataset(imagery)

        # Create a data loader for batch processing
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True
        )

        total_batches = len(loader)
        progress_bar = tqdm.tqdm(total=total_batches, desc="Processing MaskRCNN", unit="batch")
        
        # Reset timing stats
        self.timing_stats = {
            'total_time': 0.0,
            'maskrcnn_inference_time': 0.0,
            'filtering_time': 0.0,
            'saving_time': 0.0,
            'num_images': 0,
            'per_image_times': []
        }

        with torch.no_grad():
            for batch_idx, (paths, batch) in enumerate(loader):
                batch_start_time = time.time()
                
                # Get the paths and file paths for saving the mask files
                relative_paths, pifpaf_file_paths = get_label_paths(is_mask=False, img_paths=paths,
                                                                    dataset_dir=dataset_dir)

                assert all(os.path.exists(path) for path in
                           pifpaf_file_paths), "Some PiPaf Label File ('.confidence_fields.npy') does not exist!"

                # Time MaskRCNN inference
                maskrcnn_start = time.time()
                maskrcnn_results = self.model(batch)
                maskrcnn_end = time.time()
                maskrcnn_time = maskrcnn_end - maskrcnn_start
                
                # Time filtering
                filter_start = time.time()
                pifpaf_filtered: List[np.ndarray] = self.__filter_pifpaf_with_mask(batch, pifpaf_file_paths)
                filter_end = time.time()
                filter_time = filter_end - filter_start

                # Get the file paths for saving the mask files
                _, mask_file_paths = get_label_paths(is_mask=True, img_paths=paths, dataset_dir=dataset_dir)

                # Time saving
                save_start = time.time()
                save_files(pifpaf_filtered, mask_file_paths, verbose)
                save_end = time.time()
                save_time = save_end - save_start
                
                batch_end_time = time.time()
                batch_total_time = batch_end_time - batch_start_time
                
                # Update timing stats
                self.timing_stats['total_time'] += batch_total_time
                self.timing_stats['maskrcnn_inference_time'] += maskrcnn_time
                self.timing_stats['filtering_time'] += filter_time
                self.timing_stats['saving_time'] += save_time
                self.timing_stats['num_images'] += len(paths)
                
                # Record per-batch timing
                per_image_time = batch_total_time / len(paths)
                self.timing_stats['per_image_times'].extend([per_image_time] * len(paths))
                
                if verbose:
                    print(f"Batch {batch_idx + 1}/{total_batches}: "
                          f"MaskRCNN: {maskrcnn_time:.3f}s, "
                          f"Filter: {filter_time:.3f}s, "
                          f"Save: {save_time:.3f}s, "
                          f"Total: {batch_total_time:.3f}s "
                          f"({per_image_time:.3f}s per image)")

                progress_bar.update(1)

            progress_bar.close()
            
        # Print final timing statistics
        self._print_timing_stats("MaskRCNN Processing")

    def __filter_pifpaf_with_mask(self, batch,
                                  pifpaf_file_paths: List[Path] or List[str]):
        """
        Filter PifPaf predictions using segmentation masks.

        Args:
            paths (List[Path] or List[str]): List of image paths or filenames.
            batch: Batch data containing images.
            pifpaf_file_paths (List[Path] or List[str]): List of PifPaf label file paths.

        Returns:
            List[np.ndarray]: Filtered PifPaf arrays.

        """

        # Order the bounding boxes by distance from the center of the image(default)
        def order_bbox(image_size, bbox_list, only_horizontal=False, only_vertical=False):
            distances = []
            image_height, image_width = image_size
            center_x = image_width // 2
            center_y = image_height // 2

            for i, bbox in enumerate(bbox_list):
                x1, y1, x2, y2 = bbox
                bbox_center_x = (x1 + x2) // 2
                bbox_center_y = (y1 + y2) // 2
                distance = bbox_center_x if only_horizontal else bbox_center_y if only_vertical else np.sqrt(
                    (bbox_center_x - center_x) ** 2 + (bbox_center_y - center_y) ** 2)
                distances.append((i, distance))
            distances = sorted(distances, key=lambda x: x[1])
            return distances

        # Filter segmentations masks based on class and distance from the center of the image
        def filter_masks(results):
            image_size = results[0]["instances"].image_size
            pred_boxes, scores, pred_classes, pred_masks = results[0]["instances"].get_fields().values()
            if len(pred_masks) == 0:
                raise Exception("Error: Pifpaf model did not return any masks!")

            # Filter out all masks that are not person
            filtered_boxes, filtered_masks = zip(
                *[(box.cpu().numpy(), mask.cpu().numpy()) for box, mask, cls in
                  zip(pred_boxes, pred_masks, pred_classes) if cls == 0])

            # Order the masks by bbox distance to the center of the image
            distances = order_bbox(image_size, filtered_boxes)
            filtered_masks = [filtered_masks[i] for i, _ in distances]

            return filtered_masks

        # Filter PifPaf array using segmentation mask
        def filter_pifpaf_with_mask(pifpaf_array, mask, is_resize_pifpaf=False, interpolation=cv2.INTER_CUBIC):
            if is_resize_pifpaf:
                # Resize the PifPaf array to match the size of the mask
                pifpaf_resized = np.transpose(pifpaf_array, (1, 2, 0))
                pifpaf_resized = cv2.resize(pifpaf_resized, dsize=(mask.shape[1], mask.shape[0]),
                                            interpolation=interpolation)
                pifpaf_resized = np.transpose(pifpaf_resized, (2, 0, 1))

                # Filter the PifPaf array using the segmentation mask
                filtered_pifpaf = mask * pifpaf_resized
                filtered_pifpaf = np.array(
                    [cv2.resize(slice, (9, 17), interpolation=cv2.INTER_CUBIC) for slice in filtered_pifpaf])

                return filtered_pifpaf
            # Resize the mask to match the size of the PifPaf array
            mask_resized = cv2.resize(mask.astype(np.uint8), (pifpaf_array.shape[2], pifpaf_array.shape[1]))
            filtered_pifpaf = mask_resized * pifpaf_array
            return filtered_pifpaf

        # Get the masks from the PifPaf predictions
        masks = filter_masks(self.model(batch))

        # Load the PifPaf label arrays
        pifpaf_labels = [np.load(pifpaf_file_path) for pifpaf_file_path in pifpaf_file_paths]

        # Filter the PifPaf arrays using the masks
        pifpaf_filtered = [filter_pifpaf_with_mask(pifpaf_label, mask) for pifpaf_label, mask in
                           zip(pifpaf_labels, masks)]

        return pifpaf_filtered
    
    def _print_timing_stats(self, stage_name):
        """Print detailed timing statistics"""
        stats = self.timing_stats
        print(f"\n=== {stage_name} Timing Statistics ===")
        print(f"Total images processed: {stats['num_images']}")
        print(f"Total time: {stats['total_time']:.3f}s")
        print(f"Average time per image: {stats['total_time']/stats['num_images']:.3f}s")
        print(f"MaskRCNN inference time: {stats['maskrcnn_inference_time']:.3f}s ({stats['maskrcnn_inference_time']/stats['total_time']*100:.1f}%)")
        print(f"Filtering time: {stats['filtering_time']:.3f}s ({stats['filtering_time']/stats['total_time']*100:.1f}%)")
        print(f"Saving time: {stats['saving_time']:.3f}s ({stats['saving_time']/stats['total_time']*100:.1f}%)")
        if stats['per_image_times']:
            print(f"Min time per image: {min(stats['per_image_times']):.3f}s")
            print(f"Max time per image: {max(stats['per_image_times']):.3f}s")
            print(f"Std dev per image: {np.std(stats['per_image_times']):.3f}s")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Source dataset containing image files')
    parser.add_argument('--maskrcnn-cfg-file', type=str, default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                        help='Configuration file for the Mask R-CNN model')
    parser.add_argument('--pifpaf-model-name', type=str, default="shufflenetv2k16",
                        help='Name of the PifPaf model')
    parser.add_argument('-b', '--batch-size', type=int,
                        help='Batch size for processing images')
    parser.add_argument('--num-workers', type=int,
                        help='Number of worker processes for data loading')
    parser.add_argument('--single-image', type=str,
                        help='Process only a single image for timing measurement')
    args = parser.parse_args()

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
    
    # Perform PifPaf processing
    print("\n" + "="*50)
    print("STAGE 1: PifPaf Pose Estimation")
    print("="*50)
    pifpaf_model = BatchPifPaf(model_name=args.pifpaf_model_name,
                               batch_size=args.batch_size,
                               workers=args.num_workers)
    pifpaf_model(imagery=img_paths, dataset_dir=args.source, is_overwrite=False, verbose=True)

    # Perform Mask R-CNN processing
    print("\n" + "="*50)
    print("STAGE 2: MaskRCNN Segmentation Filtering")
    print("="*50)
    mask_model = BatchMask(cfg=args.maskrcnn_cfg_file,
                           batch_size=args.batch_size,
                           workers=args.num_workers)
    mask_model(imagery=img_paths, dataset_dir=args.source, is_overwrite=False, verbose=True)
    
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


if __name__ == '__main__':
    main()
