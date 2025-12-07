# ClusterPhotos User Guide

ClusterPhotos is a 3D Slicer module for automatically organizing large collections of photographs into similar groups using AI-powered image clustering. It helps prepare images for the PhotoMasking module by grouping photos with similar content into subfolders.

## Table of Contents
1. [Overview](#overview)
2. [When to Use ClusterPhotos](#when-to-use-clusterphotos)
3. [How It Works](#how-it-works)
4. [Getting Started](#getting-started)
5. [Parameters](#parameters)
6. [Workflow](#workflow)
7. [Understanding the Results](#understanding-the-results)
8. [Next Steps](#next-steps)

---

## Overview

**ClusterPhotos** uses a Vision Transformer (ViT) model to analyze and group photographs based on visual similarity. When you have a large collection of photographs taken from various angles, ClusterPhotos can automatically organize them into subfolders containing images with similar content or viewpoints.

### What ClusterPhotos Allows You To Do
1. **Load and analyze large image collections** using deep learning embeddings.
2. **Automatically cluster images** into groups based on visual similarity.
3. **Visualize clusters** using UMAP dimensionality reduction in an interactive plot.
4. **Copy clustered images** into organized subfolders ready for PhotoMasking.

---

## When to Use ClusterPhotos

ClusterPhotos is particularly useful when:

- You have **hundreds of photographs** taken from many different angles.
- Photos were taken in a **continuous sequence** without organized grouping.
- You want to **organize images by viewing angle** (e.g., top views, side views, bottom views).
- You need to prepare images for **batch masking** in PhotoMasking, which works best with images showing similar orientations.

### Example Use Case

You photographed a specimen from all angles, resulting in 300 images. Instead of manually sorting them into groups, ClusterPhotos can:
1. Analyze all images using AI
2. Identify natural groupings (e.g., "top view", "left side", "bottom view")
3. Organize them into subfolders for efficient batch masking

---

## How It Works

ClusterPhotos uses a multi-step process:

1. **Feature Extraction**: A Vision Transformer model (google/vit-large-patch16-224) creates a high-dimensional embedding for each image.
2. **Graph Construction**: A k-nearest neighbors graph connects similar images.
3. **Spectral Clustering**: Recursive spectral clustering divides images into groups.
4. **Visualization**: UMAP reduces the high-dimensional embeddings to 2D for visualization.

---

## Getting Started

### Prerequisites

ClusterPhotos will automatically install required dependencies:
- transformers (HuggingFace)
- umap-learn
- scikit-learn
- plotly

### Opening ClusterPhotos

1. Open 3D Slicer.
2. Navigate to **Modules** → **SlicerMorph** → **Photogrammetry** → **ClusterPhotos**.

---

## Parameters

### Image Folder
Select the directory containing your images. Supported formats include `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.tif`.

### Model Name
The HuggingFace model used for image embeddings. Default is `google/vit-large-patch16-224`. This is a large Vision Transformer that provides high-quality image representations.

### k-neighbors
Number of neighbors for the k-NN graph construction. Higher values create denser connections between images.
- **Default**: 10
- **Range**: 2-200
- **Tip**: Increase for larger image sets

### Max Eigenvectors
Maximum number of eigenvectors used in spectral clustering. Controls the granularity of cluster detection.
- **Default**: 15
- **Range**: 2-100

### Max Cluster Size
Maximum number of images allowed in each final cluster. Smaller values create more, finer-grained clusters.
- **Default**: 40
- **Range**: 2-2000
- **Tip**: Set based on your typical image set size for masking

### UMAP n_neighbors
Number of neighbors for UMAP dimensionality reduction. Affects how local vs. global structure is preserved in visualization.
- **Default**: 10
- **Range**: 2-200

### UMAP min_dist
Minimum distance between points in UMAP visualization. Lower values allow tighter clustering of similar points.
- **Default**: 0.1
- **Range**: 0.0-1.0

---

## Workflow

### Step 1: Load the Model

1. Set the **Model name** (or use the default).
2. Click **Load Model**.
3. Wait for the model to download (first time only) and load.

### Step 2: Load and Embed Images

1. Select your **Image folder** containing the photographs.
2. Click **Load Images & Embed**.
3. A progress bar shows the embedding process.
4. This may take several minutes for large image sets.

### Step 3: Cluster and Visualize

1. Adjust clustering parameters if needed.
2. Click **Cluster & Plot**.
3. An interactive UMAP visualization appears showing:
   - Each point represents one image
   - Colors indicate cluster assignments
   - Hover over points to see image filenames

### Step 4: Review Clusters

- The **Cluster Summary** table shows:
  - Cluster ID
  - Number of images in each cluster
  - Representative image names
- Review the visualization to ensure clusters make sense.
- Adjust parameters and re-cluster if needed.

### Step 5: Copy Clustered Images

1. Click **Copy Images to Subfolders**.
2. Choose an output directory.
3. Images will be copied into subfolders named `Cluster_0`, `Cluster_1`, etc.
4. The output directory structure is ready for PhotoMasking.

---

## Understanding the Results

### UMAP Visualization

The 2D scatter plot shows:
- **Nearby points**: Images that are visually similar
- **Distinct clusters**: Groups of images with shared characteristics
- **Colors**: Different clusters are shown in different colors

### Cluster Quality

Good clustering should:
- Group images with similar viewing angles together
- Separate clearly different viewpoints
- Create reasonably sized groups (not too large, not too small)

### Adjusting Results

If clusters don't look right:
- **Too few clusters**: Decrease Max Cluster Size
- **Too many clusters**: Increase Max Cluster Size
- **Clusters too mixed**: Increase k-neighbors or Max Eigenvectors
- **Clusters too fragmented**: Decrease k-neighbors

---

## Next Steps

After clustering your images:

1. The output folder contains subfolders with grouped images.
2. Open the [PhotoMasking module](PhotoMasking.md).
3. Set the **Input Folder** to your ClusterPhotos output directory.
4. Each cluster subfolder becomes an "image set" in PhotoMasking.
5. Use batch masking to efficiently mask each set.

---

## Tips for Best Results

### Image Collection
- **Consistent lighting**: Helps the model identify similar viewpoints.
- **Clear subject**: Images with prominent, centered subjects cluster better.
- **Sufficient variety**: At least 50-100 images work well for clustering.

### Parameter Tuning
- **Start with defaults**: They work well for typical photogrammetry datasets.
- **Iterate**: Run clustering, review, adjust, repeat.
- **Balance cluster sizes**: Aim for 20-50 images per cluster for efficient masking.

### Performance
- **First run is slower**: Model download and initial embedding take time.
- **GPU acceleration**: Embedding is faster with a CUDA-capable GPU.
- **Large datasets**: Consider processing in batches if you have thousands of images.

---

## Example Output Structure

After running ClusterPhotos on a folder with 200 images:

```
Output_Folder/
├── Cluster_0/          (45 images - top views)
│   ├── IMG_001.jpg
│   ├── IMG_015.jpg
│   └── ...
├── Cluster_1/          (38 images - front views)
│   ├── IMG_003.jpg
│   ├── IMG_022.jpg
│   └── ...
├── Cluster_2/          (42 images - side views)
│   └── ...
├── Cluster_3/          (40 images - back views)
│   └── ...
└── Cluster_4/          (35 images - bottom views)
    └── ...
```

This structure is directly usable as input for PhotoMasking.
