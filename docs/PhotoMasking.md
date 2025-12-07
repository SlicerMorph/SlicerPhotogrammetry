# PhotoMasking User Guide

PhotoMasking is a 3D Slicer module for semi-automatic image masking using the Segment Anything Model (SAM). It removes backgrounds from photographs, preparing them for photogrammetry reconstruction with the ODM module.

## Table of Contents
1. [Overview](#overview)
2. [What is SAM?](#what-is-sam)
3. [Choosing the SAM Model Variant](#choosing-the-sam-model-variant)
4. [Setting Up Input/Output Folders](#setting-up-inputoutput-folders)
5. [Processing Folders and Browsing Image Sets](#processing-folders-and-browsing-image-sets)
6. [Masking Workflows](#masking-workflows)
   - [Batch Masking (All Images)](#batch-masking-all-images)
   - [Single Image Masking](#single-image-masking)
7. [Mask Resolution and Performance](#mask-resolution-and-performance)
8. [Monitoring Masking Progress](#monitoring-masking-progress)

---

## Overview

**PhotoMasking** is designed to help users prepare large sets of photographs for photogrammetry reconstruction by removing backgrounds and isolating the object of interest. The module integrates:

- **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)** for efficient masking of each image (removing background, highlighting your object).

### What PhotoMasking Allows You To Do
1. **Run Segment Anything Model (SAM)** conveniently in 3D Slicer.  
2. **Batch-mask or individually mask** sets of images using bounding boxes and optional inclusion/exclusion points to guide segmentation.
3. **Output masked images** ready for reconstruction with the [ODM module](ODM.md).

---

## What is SAM?

**SAM (Segment Anything Model)** is a state-of-the-art segmentation model by Meta. It can segment objects in images with minimal user input:
- **Bounding boxes** around the object you wish to mask, and
- **Inclusion (green) or Exclusion (red) points** indicating details SAM might miss or include erroneously.

SAM supports several variants (ViT-base, ViT-large, ViT-huge) differing in file size, GPU memory demands, and inference speed.

---

## Choosing the SAM Model Variant

Upon first opening the **PhotoMasking** module, you will see a dropdown labeled **SAM Variant**. The three typical variants are:

1. **ViT-base (~376 MB):**  
   - Fastest inference, least GPU memory usage. Good if you have limited GPU or use CPU.
2. **ViT-large (~1.03 GB):**  
   - Balanced in speed and memory usage.  
3. **ViT-huge (~2.55 GB):**  
   - Highest accuracy potential, but requires significant GPU RAM and more time.

### Loading the Chosen Model
- Select the variant that fits your resources (e.g., **ViT-base** if you're unsure).
- Click **Load Model**.  
- Wait for the download if you haven't already downloaded the weights.  
- Once loaded, you can proceed with folder processing.

> **Tip:** If you have a strong GPU (e.g., 8GB VRAM or more), you can try **ViT-large** or **ViT-huge**. If you are CPU-only or have ~4GB VRAM or less, **ViT-base** is safer.

---

## Setting Up Input/Output Folders

1. **Input Folder:** Should contain **multiple subfolders** (each subfolder is one "image set"). For example:
   - `Beaver_Skull_Images`
     - `Set1` (all photographs of the object taken in a first orientation - e.g., top view)
     - `Set2` (all photographs of the object taken in a second orientation - e.g., bottom view)
     - etc.

   Keeping similar orientations in sets helps with the workflow associated with masking the background (see below).

2. **Output Folder:** A separate folder you create for masked images and where final results will be placed. 

   > **NOTE:** Do NOT place the output folder inside the Input Folder. The Output folder has to reside outside of the Input folder structure, which is recursively processed.

Use the **Directory** pickers in the module UI to select these paths. PhotoMasking stores your selection so you don't have to re-pick them every time. Just remember to change them for each new reconstruction project.

> **Tip:** If you have a large collection of photos taken at different angles, consider using the [ClusterPhotos module](ClusterPhotos.md) first to automatically organize them into similar groups.

---

## Processing Folders and Browsing Image Sets

Once you've loaded a SAM model and chosen valid **Input** and **Output** directories:

- Click **Process Folders**.  
  - The module will scan each subfolder in the Input folder.  
  - Any recognized image (`*.jpg`, `*.png`, etc.) will be listed and ready for masking.
- A **progress bar** indicates the scanning progress.  
- After processing, you can pick any subfolder (image set) in the **Image Set** dropdown to inspect or mask.

### Navigating Sets
- The **Image Set** combobox shows each subfolder name.
- Select a set to see its images.
- You can switch sets at any time — each set's state (masked or not) is preserved.

### Image List
- The **Image List** table shows the filename and image number for each image in the selected image set.
- You can click on each image in this list to load and view the image, and its related mask, if present.
- Unmasked images appear **red** in this list but will change to **green** once masked.

---

## Masking Workflows

Masking removes background and keeps only the foreground object. PhotoMasking provides **two** main masking workflows:

1. **Batch Masking:** For when you want to define a single bounding box (ROI) that applies across **all** images in a set.
2. **Single Image Masking:** For fine-tuning individual images or if batch mode didn't capture the object well enough.

### Batch Masking (All Images)

1. **Select a Set** from the dropdown.
2. Click **Place/Adjust ROI for All Images**.  
   - This removes any previously created masks for that set.
   - Puts you in a special **global bounding box** mode.
3. Slicer will prompt you to **place a bounding box** that should encompass the object in every image.
   - Use the 2D slice viewer (**Red**) to **drag and resize** the ROI, ensuring the bounding region covers the entire object.
   - Switch between images with the **<** and **>** buttons to verify the bounding box is suitable for all angles.
4. When satisfied, click **Finalize ROI and Mask All Images**.  
   - SAM processes each image. If you checked **quarter** or **half** resolution (see [Mask Resolution](#mask-resolution-and-performance)), it speeds up processing at the cost of fine detail.
   - A **progress bar** updates you on how many images have been masked and provides an estimate of how long remains.

Once batch masking completes, **all images** in that set will have a masked `.jpg` color output and `_mask.jpg` binary mask in the Output folder.

---

### Single Image Masking

Use this when you want more precise masks or to correct images from the batch approach:

1. **Navigate** to the problematic image using the **<** and **>** buttons.  
2. Click **Place Bounding Box** to define a bounding ROI around that single image's object.
   - **Note:** If the image was already masked, you'll be prompted to remove the existing mask before placing a new ROI.
3. **Optionally** place **inclusion (green)** or **exclusion (red)** fiducial points:
   - **Add Inclusion Points**: Mark parts of the object that were incorrectly excluded.
   - **Add Exclusion Points**: Mark background areas or clutter incorrectly included.
   - After choosing one of these modes, click anywhere in the 2D viewer to drop points.
   - **Click 'Stop Adding'** once you're done placing those points.
4. Finally, click **Mask Current Image**. SAM uses the bounding box plus your points to refine the mask.

> You can also **clear** all points or remove them individually if you need to restart.

**Result:** A masked image pair (`.jpg` + `_mask.jpg`) is saved to your Output folder. The 2D viewer will show the masked version in the 'Red2' slice viewer as a preview.

---

## Mask Resolution and Performance

Beneath the image set tools, you'll find these **radio buttons**:
- **Full resolution (1.0)**
- **Half resolution (0.5)**
- **Quarter resolution (0.25)**

Choosing **half** or **quarter** can speed up masking, especially on CPU. The trade-off is lower detail near object boundaries.

> **Tip**: If you have a powerful GPU, you can keep **full resolution** for maximum detail. If you notice slow performance or run out of memory, switch to a lower resolution.

---

## Monitoring Masking Progress

- A label like **'Masked: 3/20'** appears, telling you how many images have a finalized mask.
- For batch mode, a **dedicated progress bar** shows the overall time remaining and a per-image estimate.
- For single-image mode, there is no progress bar, but Slicer will show a brief 'processing' message.

---

## Next Steps

Once all your images are masked, proceed to the [ODM module](ODM.md) to reconstruct your 3D model using the masked images.
