# ODM User Guide

ODM (OpenDroneMap) is a 3D Slicer module for photogrammetry reconstruction. It takes masked images from the PhotoMasking or VideoMasking modules and reconstructs them into a 3D textured model using NodeODM.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Managing NodeODM](#managing-nodeodm)
   - [Installing and Launching NodeODM](#installing-and-launching-nodeodm)
   - [Stopping the NodeODM Container](#stopping-the-nodeodm-container)
4. [Selecting the Masked Images Folder](#selecting-the-masked-images-folder)
5. [Find-GCP and Marker Detection](#find-gcp-and-marker-detection)
6. [Configuring and Running ODM Tasks](#configuring-and-running-odm-tasks)
7. [Saving and Restoring Tasks](#saving-and-restoring-tasks)
8. [Importing the Final 3D Model into Slicer](#importing-the-final-3d-model-into-slicer)

---

## Overview

**ODM** is the photogrammetry reconstruction engine module that interfaces with **NodeODM/WebODM** to convert masked photographs into 3D models. This module accepts masked images from either:
- [PhotoMasking module](PhotoMasking.md) - for static photograph masking
- [VideoMasking module](VideoMasking.md) - for video frame extraction and masking

### What ODM Allows You To Do
1. **Launch and manage NodeODM** Docker containers for reconstruction.
2. **Generate GCP data** (if you have marker coordinates) to obtain accurate physical dimensions of the specimens.
3. **Configure reconstruction parameters** with detailed tooltips for each option.
4. **Monitor task progress** in real-time via console logs.
5. **Download and import the final 3D model** back into Slicer for further inspection and visualization.
6. **Save and restore reconstruction tasks** for reproducibility.

---

## Prerequisites

ODM requires Docker to be installed on your system:
- [Install Docker](https://docs.docker.com/engine/install/)
- [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU acceleration)

> **Note:** If you're using [MorphoCloud On Demand](https://instances.morpho.cloud), all prerequisites are already installed.

---

## Managing NodeODM

### Installing and Launching NodeODM

The ODM module includes tools to manage NodeODM:

1. **Launch NodeODM**  
   - Checks for Docker.  
   - Pulls the `opendronemap/nodeodm:gpu` image if not present.  
   - Attempts to start the container on **port 3002** with GPU enabled.  

2. Click the **Launch NodeODM** button to start the service.

> You can ignore these steps if you already have a local NodeODM instance running. In that case, just enter its IP and port in the configuration section.

**Node IP** defaults to `127.0.0.1` with **port** `3002`. Unless you have specialized Docker port forwarding or remote server usage, stick with these defaults.

### Stopping the NodeODM Container

Click **Stop Node** to stop any running NodeODM container on port 3002.

---

## Selecting the Masked Images Folder

Before running a reconstruction task:

1. Use the **Masked Images Folder** selector to point to your output folder containing masked images.
2. This folder should contain:
   - Masked color images (`.jpg`)
   - Binary mask files (`_mask.jpg`)
   - Optionally, a `combined_gcp_list.txt` file for georeferencing

The masked images can come from either the [PhotoMasking](PhotoMasking.md) or [VideoMasking](VideoMasking.md) modules.

---

## Find-GCP and Marker Detection

This module supports generating a single **GCP (Ground Control Points) file** if you have images containing ArUco markers and a known coordinate file:

1. **Clone Find-GCP**: Downloads the [Find-GCP](https://github.com/zsiki/Find-GCP) Python script into your Slicer environment.
2. Provide your **Find-GCP Script** path (auto-filled if you clicked 'Clone') and your **GCP Coord File**.
3. Pick the **ArUco Dictionary ID** you used when printing/generating your markers. Commonly `2` or `4`.
4. Click **Generate GCP File from Images**. The script will produce `combined_gcp_list.txt` in your masked images folder, merging GCP references from all images.

> **What is a GCP Coord File?**  
> It lists real-world coordinates for each marker ID, ensuring the final reconstruction is positioned accurately in 3D space (e.g., for georeferencing) and has accurate physical dimensions.

---

## Configuring and Running ODM Tasks

The **Launch WebODM Task** section provides options for configuring your reconstruction:

### Connection Settings
- **Node IP**: Your local or remote NodeODM service (default `127.0.0.1`)
- **Node Port**: Port number (default `3002`)

### Reconstruction Parameters
A list of advanced parameters is available, each with a **tooltip** for detailed explanation:
- **ignore-gsd**: Ignore Ground Sample Distance calculations
- **matcher-neighbors**: Number of nearby images to match
- **mesh-octree-depth**: Detail level of the mesh
- **mesh-size**: Maximum number of triangles in the mesh
- **min-num-features**: Minimum features to extract per image
- **pc-filter**: Point cloud filtering level
- **depthmap-resolution**: Resolution for depth maps
- **matcher-type**: Algorithm for image matching (bruteforce, bow, flann)
- **feature-type**: Feature detection algorithm (dspsift, akaze, hahog, orb, sift)
- **feature-quality**: Quality level for feature extraction
- **pc-quality**: Point cloud quality
- **optimize-disk-space**: Reduce disk usage during processing
- **no-gpu**: Disable GPU acceleration

### Additional Settings
- **max-concurrency**: Number of parallel processes used by the reconstruction pipeline
- **Dataset name**: A friendly label for your reconstruction

### Running the Task

**Steps to run**:
1. Confirm your masked images folder is set correctly.
2. Adjust parameters or accept defaults (tuned for typical scenarios).  
3. Click **Run WebODM Task With Selected Parameters**.

The module then:
- Uploads your masked images (and optional `combined_gcp_list.txt`) to NodeODM.
- Creates a new WebODM task using these parameters.

### Task Monitoring
- A console log appears in the **Console Log** box showing real-time progress.
- You can click **Stop Monitoring** if you want to disconnect from real-time updates (the task continues on NodeODM in the background).

---

## Saving and Restoring Tasks

You can **Save Task** at any point (even while a task is running). This creates a JSON file containing:
- Your folder paths
- WebODM parameters
- (Optional) Output directory of the WebODM results

Later, **Restore Task** re-loads everything exactly as it was, so you can continue or re-run the reconstruction.

> **Recommendation**: Save the task JSON into the same folder you used for the masked images, so you have everything in one place.

---

## Importing the Final 3D Model into Slicer

When the ODM task finishes:
1. The module automatically downloads the result into a folder named something like `WebODM_<hash>` under your masked images folder.
2. Click **Import WebODM Model** to load the `odm_textured_model_geo.obj` (or equivalent) back into Slicer.
3. Slicer will switch to a **3D layout** showing you the reconstructed mesh.

From there, you can continue analyzing or refining in Slicer, or export to other software.

---

## Tips for Best Results

- **More images = better results**: Use 50-200 images with good overlap (60-80% recommended).
- **Consistent lighting**: Avoid shadows and reflections when photographing.
- **Use a turntable setup**: Rotating the object while keeping the camera fixed produces consistent results.
- **Include ArUco markers**: For accurate physical scale of your 3D model.
- **GPU acceleration**: Significantly speeds up reconstruction, especially for large image sets.
