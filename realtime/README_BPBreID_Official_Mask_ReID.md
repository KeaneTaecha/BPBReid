# BPBreID Official Mask ReID System

This script integrates the official BPBreID mask processing pipeline with person re-identification for real-time video processing. It extracts features from a gallery person image and compares them with every frame of a video, showing confidence levels and color-coded detections.

## Features

- **Official BPBreID Mask Processing**: Uses the same mask processing pipeline as the official BPBreID testing
- **MaskRCNN + PifPaf Transforms**: Applies official mask transformations including `CombinePifPafIntoFiveVerticalParts` and `AddBackgroundMask`
- **Real-time ReID**: Processes video frames with YOLO detection and BPBreID feature extraction
- **Confidence Visualization**: Shows BPBreID confidence levels and YOLO detection confidence
- **Color-coded Detections**: 
  - 🟢 **Green**: High similarity (>0.7) - MATCH
  - 🟡 **Yellow**: Medium similarity (0.5-0.7) - UNCERTAIN  
  - 🔴 **Red**: Low similarity (<0.5) - NO MATCH

## Requirements

### Files Required
- `pretrained_models/bpbreid_market1501_hrnet32_10642.pth` - BPBreID model weights
- `pretrained_models/hrnet32_imagenet.pth` - HRNet pretrained weights
- `yolov8n.pt` - YOLO model weights
- `datasets/Compare/dataset-2/person-1.jpg` - Gallery person image
- `datasets/Compare/dataset-2/person-1-vid.MOV` - Video to process

### Dependencies
- torch
- torchvision
- opencv-python
- ultralytics (YOLO)
- PIL
- numpy
- detectron2 (for MaskRCNN)

## Usage

### Basic Usage
```bash
cd realtime
python bpbreid_official_mask_reid.py
```

### Custom Configuration
You can modify the paths in the `main()` function:

```python
# Configuration
REID_MODEL_PATH = "pretrained_models/bpbreid_market1501_hrnet32_10642.pth"
HRNET_PATH = "pretrained_models/hrnet32_imagenet.pth"
YOLO_MODEL = "yolov8n.pt"
GALLERY_PATH = "datasets/Compare/dataset-2/person-1.jpg"
VIDEO_PATH = "datasets/Compare/dataset-2/person-1-vid.MOV"
```

### Programmatic Usage
```python
from bpbreid_official_mask_reid import BPBreIDOfficialMaskReID

# Initialize system
reid_system = BPBreIDOfficialMaskReID(
    reid_model_path="path/to/bpbreid_model.pth",
    hrnet_path="path/to/hrnet_model.pth",
    yolo_model_path="yolov8n.pt"
)

# Load gallery person
reid_system.load_gallery_person("path/to/gallery_person.jpg")

# Process video
reid_system.process_video(
    video_path="path/to/video.mp4",
    output_path="output_video.mp4",
    show_preview=True,
    save_video=True
)
```

## How It Works

### 1. Mask Processing Pipeline
The script follows the official BPBreID mask processing pipeline:

1. **YOLO Detection**: Detects persons in each frame
2. **MaskRCNN Segmentation**: Creates detailed person masks using MaskRCNN
3. **PifPaf Simulation**: Simulates PifPaf confidence fields (36 fields: 17 keypoints + 19 connections)
4. **Official Transforms**: Applies official BPBreID transforms:
   - `CombinePifPafIntoFiveVerticalParts`: Groups into 5 vertical body parts
   - `AddBackgroundMask`: Adds background mask with threshold strategy
   - `ResizeMasks`: Resizes to feature map size (96x32)
   - `PermuteMasksDim`: Permutes dimensions to match BPBreID format

### 2. BPBreID Configuration
The script uses the official testing configuration from `bpbreid_market1501_test.yaml`:

```yaml
model:
  bpbreid:
    mask_filtering_testing: True
    test_embeddings: ['bn_foreg', 'parts']
    test_use_target_segmentation: 'soft'
    testing_binary_visibility_score: False
    masks:
      preprocess: 'five_v'
      softmax_weight: 15.0
      background_computation_strategy: 'threshold'
```

### 3. Feature Extraction and Matching
1. **Gallery Person**: Extracts features from the gallery person image
2. **Video Processing**: For each detected person in video frames:
   - Creates official BPBreID masks
   - Extracts features using BPBreID model
   - Computes cosine similarity with gallery features
3. **Visualization**: Shows confidence levels and color-coded detections

## Output

The script produces:
- **Real-time Preview**: Shows processed frames with detections and confidence levels
- **Output Video**: Saves processed video with annotations
- **Console Output**: Progress updates and processing statistics

### Visualization Elements
- **Bounding Boxes**: Color-coded based on similarity
- **Confidence Labels**: 
  - BPBreID similarity score (0-1)
  - YOLO detection confidence
  - Match status (MATCH/UNCERTAIN/NO MATCH)
- **Processing Info**: Frame count, person count, gallery person ID

## Performance Considerations

### Jetson Optimization
For Jetson deployment:
- The script is designed to work on Jetson devices
- Uses batch size of 1 for mask processing
- Includes fallback to simple masks if MaskRCNN fails
- Optimized for real-time processing

### Memory Usage
- Loads models once during initialization
- Processes frames sequentially to minimize memory usage
- Uses torch.no_grad() for inference

## Troubleshooting

### Common Issues

1. **MaskRCNN Loading Failed**
   - The script will fall back to simple YOLO-based masks
   - Check detectron2 installation and model availability

2. **Model Loading Errors**
   - Verify all model paths are correct
   - Ensure model files exist and are accessible

3. **Video Processing Issues**
   - Check video file format and codec
   - Ensure sufficient disk space for output video

### Debug Mode
The script includes extensive error handling and logging. Check console output for:
- Model loading status
- Mask processing warnings
- Feature extraction errors
- Processing statistics

## Technical Details

### Mask Format
The BPBreID model expects masks in format `[N, K+1, H, W]` where:
- N: Batch size (1 for real-time)
- K+1: Number of parts + background (6 for 5 parts + background)
- H, W: Feature map height and width (96, 32)

### Similarity Computation
Uses cosine similarity between normalized feature vectors:
```python
similarity = (cosine_similarity + 1) / 2  # Convert from [-1,1] to [0,1]
```

### Thresholds
- **High Similarity**: >0.7 (Green - MATCH)
- **Medium Similarity**: 0.5-0.7 (Yellow - UNCERTAIN)
- **Low Similarity**: <0.5 (Red - NO MATCH)

## Integration with Official Pipeline

This script is designed to be as close as possible to the official BPBreID testing pipeline:

1. **Same Configuration**: Uses identical model and mask settings
2. **Same Transforms**: Applies official mask transformation pipeline
3. **Same Preprocessing**: Uses official image normalization and resizing
4. **Same Feature Extraction**: Uses both foreground and parts embeddings

The main difference is that it processes video frames in real-time rather than static images, and includes YOLO detection for person localization.
