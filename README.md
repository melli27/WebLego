# WebLego: Visvu-2025-Schiebel

An interactive, browser-based system for exploring large 3D volumetric datasets using GPU-accelerated supervoxel segmentation, exhaustive graph clustering, and hierarchical meta-clustering.
The project is inspired by the paper “FeatureLego: Volume Exploration Using Exhaustive Clustering of Super-Voxels.”

## Project Goal & Visualization Technique

The goal of this project was to implement the FeatureLego framework (Jadhav et al., 2019) using modern WebGPU and D3.js FeatureLego is a volume exploration approach that partitions a volumetric dataset into semantic regions ("Legos") that users can interactively select and group. This implementation leverages WebGPU for high-performance volume rendering and clustering, and D3.js for the interactive meta-cluster tree visualization.

Since the code and the datasets used in the paper were not available, we re-implemented the core ideas of the FeatureLego pipeline with practical adaptations for browser-based execution and limited development scope. Certain advanced visualization features and optimization steps described in the original paper were simplified or omitted in favor of clarity, robustness, and real-time performance. 

### Visualization Techniques Implemented

-   **WebGU** using WebGPU fragment shaders.
-   **Hierarchical Tree Visualization** (D3) for browsing meta-clusters.

## Algorithm Summary (Re-Implemented)

### 1. GPU 3D SLIC Supervoxel Segmentation

-   Cluster centers initialized on a regular 3D grid.
-   Each voxel computes a distance to nearby centers based on intensity
    and spatial proximity.
-   Iterative assignment and update converge to compact supervoxels.

### 2. Felzenszwalb--Huttenlocher (FH) Graph Clustering

-   Each supervoxel represented by a 64-bin normalized intensity
    histogram.
-   26-neighborhood adjacency graph.
-   Edge weight = χ² histogram distance.
-   Exhaustive sweep over parameter *k* to find stable regions.

### 3. Meta-Clustering

-   Regions compared using voxel-weighted Jaccard overlap.
-   Similarity graph → minimum spanning tree → threshold cut.
-   Connected components form meta-clusters.
-   Containment relationships create a hierarchy.

### 4. Interactive Visualization

-   GPU raymarching for volume rendering.
-   Tree-based browsing and region highlighting.

## Installation & Running Instructions

### Requirements

-   WebGPU-enabled browser (Chrome / Edge).
-   Local web server.

### Run Locally
Open:
    http://127.0.0.1:5500/index.html


## Usage Guide

### Adjustable Parameters

-   **DataSets**
    -   Dataset: Switch between loaded .vol files (Aneurism, Fuel, etc.).
-   **Visualization setting**
    -   Highlight Intensity: Adjusts the intensity of the highlight effect.
    -   Show Only Highlight: Toggle to render only the selected region
-   **Meta-Clustering**
    -   Maximum branching: Controls the breadth of the meta-cluster tree.

### Interaction

  Control      Action
  ------------ ----------------------------
  Mouse Drag   Rotate volume
  Tree Click   Highlight region


## References

1.  Jahav et al. *FeatureLego: Volume Exploration Using Exhaustive
    Clustering of Super-Voxels*.
2.  WebGPU Specification -- https://gpuweb.github.io/gpuweb/
3.  D3.js -- https://d3js.org


## Author

Melanie Schiebel 12023986
