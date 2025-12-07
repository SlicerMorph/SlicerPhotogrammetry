# VideoMasking User Guide

VideoMasking is a 3D Slicer module for extracting and masking frames from video using SAMURAI (Segment Anything Model for Universal and Robust AI tracking). It converts video recordings into masked individual frames that can be used for photogrammetry reconstruction with the ODM module.

## Table of Contents
1. [Overview](#overview)
2. [What is SAMURAI?](#what-is-samurai)
3. [Prerequisites](#prerequisites)
4. [SAMURAI Setup](#samurai-setup)
5. [Video Preparation](#video-preparation)
   - [Video Requirements](#video-requirements)
   - [Video Format Conversion](#video-format-conversion)
   - [Frame Extraction](#frame-extraction)
6. [ROI Selection and Tracking](#roi-selection-and-tracking)
7. [Key-Frame Filtering](#key-frame-filtering)
8. [Saving Output](#saving-output)
9. [Next Steps](#next-steps)

---

## Overview

**VideoMasking** enables you to create masked frames from video recordings of objects. Instead of taking individual photographs, you can record a video of your specimen rotating on a turntable and let VideoMasking:

1. Extract all frames from the video automatically
2. Track the object across all frames using SAMURAI
3. Filter similar frames using a similarity index to reduce redundancy
4. Generate masked frames ready for photogrammetry

### What VideoMasking Allows You To Do
1. **Convert video formats** (MOV to MP4) if needed.
2. **Automatically extract all frames** from video (up to 2000 frames maximum).
3. **Select a Region of Interest (ROI)** on the first frame to identify your object.
4. **Automatically track and mask** the object across all frames using SAMURAI.
5. **Filter similar frames** based on visual similarity to reduce redundant frames.
6. **Output masked frames** ready for reconstruction with the [ODM module](ODM.md).

---

## What is SAMURAI?

**SAMURAI** (Segment Anything Model for Universal and Robust AI tracking) is a state-of-the-art video object segmentation model. It extends the Segment Anything Model (SAM) with motion-aware memory capabilities, allowing it to:

- Track objects across video frames with zero-shot learning
- Handle occlusions and object deformations
- Maintain consistent segmentation throughout the video

SAMURAI is particularly effective for photogrammetry workflows where you need consistent object masking across many frames captured from different angles.

---

## Prerequisites

VideoMasking requires:

- **GPU with CUDA support**: SAMURAI benefits significantly from GPU acceleration
- **PyTorch with CUDA**: Installed automatically via Slicer's PyTorchUtils
- **Sufficient disk space**: For video conversion and frame extraction

> **Note:** If you're using [MorphoCloud On Demand](https://instances.morpho.cloud), all prerequisites are already configured.

---

## SAMURAI Setup

Before using VideoMasking, you need to set up the SAMURAI repository:

1. Open the **VideoMasking** module in 3D Slicer.
2. Expand the **SAMURAI Setup** collapsible section.
3. Click **Clone SAMURAI** to download the SAMURAI repository.
   - This clones the [SlicerMorph SAMURAI fork](https://github.com/SlicerMorph/Samurai.git) into the module's Support directory.
4. Wait for the setup to complete. The module will download model checkpoints automatically when needed.

### Checkpoint Selection
- Select the appropriate **checkpoint** for your use case.
- Choose your **device** (CUDA for GPU acceleration, or CPU as fallback).

---

## Video Preparation

### Output Format Selection

Before loading your video, choose your preferred image format for extracted frames:
- **PNG**: Lossless compression, larger files (if you opt to use compressed PNG, beware that workflow can be significantly slower)
- **JPG**: Smaller file sizes, slight quality loss

This setting applies to extracted frames, masks, and all saved outputs.

### Video Requirements

VideoMasking has specific frame limits to ensure memory-efficient processing:

- **Maximum frames**: 2000 frames total (~33 seconds at 60fps)
- **Automatic chunking**: Videos with more than 600 frames are automatically split into smaller chunks for memory safety
- **All frames extracted**: The module extracts every frame from your video (no frame interval selection)

If your video exceeds 2000 frames, you'll need to trim it before processing.

### Video Format Conversion

If your video is in MOV format (common from iPhone/camera recordings), conversion is handled automatically:

1. Expand the **Video Prep** collapsible section.
2. Select your input video file using the **Video File** selector.
3. The module will automatically convert MOV to MP4 when you click **Load Video**.

### Frame Extraction

Frame extraction happens automatically when you load a video:

1. Select your video file.
2. Set the **Frames Directory** where extracted frames will be saved.
3. Click **Load Video** to begin preparation.
4. The module will:
   - Convert MOV to MP4 if needed
   - Validate the frame count (must be ≤2000 frames)
   - For videos >600 frames: Split into chunks and extract only the first frame initially (for ROI setup)
   - For videos ≤600 frames: Extract all frames immediately
5. Wait for extraction to complete.

> **Note:** All frames are extracted automatically - there is no frame interval or "every Nth frame" setting. The module extracts every single frame from your video.

---

## ROI Selection and Tracking

After video preparation:

1. Expand the **ROI & Tracking** collapsible section.
2. Click **Load Frames** to load the extracted frames into the viewer.
3. The first frame will be displayed in the Red slice viewer.

### Selecting the ROI

1. Click **Select ROI on First Frame**.
2. Draw a bounding box around your object in the first frame:
   - Click and drag to create a rectangle that encompasses the entire object.
   - Make sure the box includes the complete object with a small margin. Try to reduce the amount of background in the ROI. 
3. Review your selection - this ROI will be used to initialize tracking.

### Running Tracking

1. Once satisfied with the ROI, click **Finalize ROI & Run Tracking**.
2. SAMURAI will process all frames:
   - The model uses the ROI to identify the object in frame 1.
   - It then tracks and segments the object through all subsequent frames.
   - For chunked videos, each chunk is processed sequentially to manage memory.
   - Progress is shown in the log panel.
3. When complete, masks will be generated for all frames.

---

## Key-Frame Filtering

After tracking is complete, you can reduce the number of frames using **similarity-based filtering**. This is important because consecutive video frames are often very similar, and having too many similar frames can slow down or degrade photogrammetry reconstruction.

### How Filtering Works

The filtering algorithm:
1. Compares each masked frame to the previously kept frame
2. Calculates visual dissimilarity based on the masked region only
3. Keeps frames that are sufficiently different from the last kept frame
4. Always keeps the first frame as a starting point

### Using the Filter

1. After tracking completes, locate the **Key-Frame Filtering** section.
2. Adjust the **Similarity Threshold** slider:
   - **Higher values** (e.g., 0.90): Keep more frames (frames must be very similar to be removed)
   - **Lower values** (e.g., 0.70): Keep fewer frames (more aggressive filtering)
   - **Default**: Start around 0.80-0.85 
3. Click **Filter Key Frames**.
4. The module will report how many frames were kept (e.g., "Kept 85/300 frames").

> **Tip:** For photogrammetry, you typically want 150-300 final frames with good coverage of all viewing angles. Start with the suggested default threshold and adjust if needed.

---

## Saving Output

### Saving Frames

1. Expand the **Save** section.
2. Select an output folder using **Browse**.
3. Click **Save Outputs**.
4. The module saves:
   - **original/Set1/**: Original (unmasked) frames
   - **masked/Set1/**: Masked frames and binary mask files (`_mask` suffix)

If you ran key-frame filtering, only the filtered frames are saved. Otherwise, all frames are saved.

### EXIF Metadata

The module automatically embeds camera metadata (extracted from the video) into saved images, which helps photogrammetry software estimate camera parameters.

---

## Next Steps

Once VideoMasking has generated your masked frames:

1. Note the output directory containing your masked frames (the `masked/` subfolder).
2. Open the [ODM module](ODM.md).
3. Set the **Masked Images Folder** to the `masked/` folder from VideoMasking output.
4. Configure and run the reconstruction task.

---

## Tips for Best Results

### Video Recording
- **Use a turntable**: Place your object on a rotating platform for consistent coverage.
- **Steady camera**: Use a tripod to minimize camera shake.
- **Good lighting**: Ensure even, diffuse lighting without harsh shadows.
- **Plain background**: A solid, contrasting background helps with segmentation.
- **Slow rotation**: 20-30 seconds for a full rotation at 60fps gives ~1200-1800 frames.
- **Keep within limits**: Videos must be ≤2000 frames (~33 seconds at 60fps).

### ROI Selection
- **Include the entire object**: The initial ROI should fully contain the object.
- **Add margin**: A small margin around the object helps with tracking.
- **Avoid background clutter**: If possible, position the object away from similar-colored backgrounds.

### Filtering
- **Start with defaults**: The 0.80 similarity threshold works well for most videos.
- **Check coverage**: After filtering, ensure you still have frames from all viewing angles.
- **Re-filter if needed**: You can adjust the threshold and re-run filtering.

---

## Troubleshooting

### Video Too Long Error
- Trim your video to ≤2000 frames (~33 seconds at 60fps)
- Use video editing software or ffmpeg to cut the video

### Tracking Loses the Object
- Try a larger initial ROI
- Ensure the object doesn't leave the frame during the video
- Check that the object is clearly visible and contrasts with background

### Out of Memory Errors
- The module automatically chunks long videos to prevent this
- If errors persist, try a shorter video
- Close other GPU-intensive applications

### Slow Processing
- Ensure CUDA is being used (check device selection)
- Processing time depends on video length and resolution
- Consider using MorphoCloud for faster GPU access
