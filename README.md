# SlicerMorph Photogrammetry Extension

<img src="PhotoMasking/Resources/Icons/PhotoMasking.png" width="300">

The SlicerMorph Photogrammetry extension provides a complete workflow for reconstructing 3D models from photographs or video within 3D Slicer. The extension includes four specialized modules that work together to transform your images into textured 3D models.

## Modules

| Module | Description |
|--------|-------------|
| [**PhotoMasking**](docs/PhotoMasking.md) | Semi-automatic image masking using SAM (Segment Anything Model). Removes backgrounds from photographs to prepare them for 3D reconstruction. |
| [**VideoMasking**](docs/VideoMasking.md) | Video-based masking using SAMURAI. Extracts frames from video and tracks/masks objects automatically across all frames. |
| [**ODM**](docs/ODM.md) | 3D reconstruction engine using OpenDroneMap/NodeODM. Converts masked images into textured 3D models. |
| [**ClusterPhotos**](docs/ClusterPhotos.md) | AI-powered image clustering using Vision Transformers. Organizes large photo collections into similar groups for efficient batch masking. |

## Typical Workflows

### Workflow 1: Photographs → 3D Model
1. **(Optional)** Use **ClusterPhotos** to organize photos by viewing angle
2. Use **PhotoMasking** to remove backgrounds from photographs
3. Use **ODM** to reconstruct the 3D model

### Workflow 2: Video → 3D Model
1. Use **VideoMasking** to extract and mask frames from video
2. Use **ODM** to reconstruct the 3D model

## Citing Photogrammetry

If you use the Photogrammetry extension in a scientific publication (conference abstract, preprint, journal paper, etc.), please cite:

**[Thomas OO, Zhang C, Maga AM. 2025. SlicerMorph photogrammetry: an open-source photogrammetry workflow for reconstructing 3D models. Biol Open; 14 (8): bio062126. doi: https://doi.org/10.1242/bio.062126](https://journals.biologists.com/bio/article/doi/10.1242/bio.062126/368790/SlicerMorph-photogrammetry-An-open-source)**

For details about photographing specimens using a low-cost setup and using ArUco markers for physical scale, see:

**[Zhang and Maga (2023) An Open-Source Photogrammetry Workflow for Reconstructing 3D Models](https://academic.oup.com/iob/article/5/1/obad024/7221338)**

## Prerequisites

### Running on MorphoCloud

There are no prerequisites if [you are using MorphoCloud On Demand](https://instances.morpho.cloud). All necessary libraries are preloaded.

### Running Locally

We recommend using MorphoCloud On Demand for the best experience:
- **GPU Acceleration**: NVIDIA A100 GPUs significantly speed up both masking and reconstruction
- **Typical runtime**: 60-70 minutes for the sample data workflow on MorphoCloud

To run locally, you'll need:
1. [Docker](https://docs.docker.com/engine/install/)
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)
3. Admin access to your computer

> **Note:** Due to Docker installation complexities, the Photogrammetry extension is currently only available in the Slicer Extension Catalogue for Linux. Again, we suggest running the Photogrammetry extension in [MorphoCloud On Demand](https://instances.morpho.cloud) using g3.xl flavor for best performance. 

## Sample Data

### Full Dataset
Unprocessed photographs from [15 mountain beavers used in Zhang and Maga, 2022](https://seattlechildrens1.box.com/v/PhotogrammetrySampleData)

### Tutorial Dataset
Single specimen (UWBM 82409) used in tutorials:
https://app.box.com/shared/static/z8pypqqmel8pv4mp5k01philfrqep8xm.zip

## Video Tutorial

[Watch the Photogrammetry video tutorial on YouTube](https://www.youtube.com/watch?v=YRHlb0dGyNc&t=9s)

## Documentation

- [PhotoMasking User Guide](docs/PhotoMasking.md) - Image masking with SAM
- [VideoMasking User Guide](docs/VideoMasking.md) - Video frame extraction and masking with SAMURAI
- [ODM User Guide](docs/ODM.md) - 3D reconstruction with NodeODM
- [ClusterPhotos User Guide](docs/ClusterPhotos.md) - AI-powered image organization

## Funding

The Photogrammetry extension is supported by grants (DBI/2301405, OAC/2118240) from the National Science Foundation to AMM (Seattle Children's Research Institute).

## Acknowledgements

The Photogrammetry extension uses the following open-source projects:

* [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) - for segmenting foreground objects in photographs
* [pyODM from the OpenDroneMap project](https://github.com/OpenDroneMap/PyODM) - for stereophotogrammetric reconstruction of 3D models
* [SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory](https://github.com/yangchris11/samurai) - for segmenting foreground objects from video
* [Vision Transformer (ViT-large)](https://huggingface.co/google/vit-large-patch16-224) - for image clustering in ClusterPhotos

We thank these groups for making their tools publicly available.
