# Performance benchmarking of HRNet for object detection in variable lighting conditions

This project evaluates object detection accuracy under varying illumination—morning, afternoon, evening, and night. It analyzes how lighting affects model performance, especially for autonomous vehicles where safe navigation depends on reliable detection. The aim is to quantify performance, expose challenges in low-light scenes, and identify opportunities for more robust real-world deployment.

We use the [Day-night Dataset](https://www.kaggle.com/datasets/stevemark/daynight-dataset) from Kaggle.

## Project Overview

Standard object detection models often degrade in low-light conditions. This project benchmarks a pre-trained **HRNet (High-Resolution Network)** on semantic segmentation tasks across different times of day. It further investigates whether image enhancement techniques (CLAHE and Denoising) can recover performance in night-time scenes.

**Key Features:**
- **Time-of-Day Categorization:** Automatically classifies images into Morning, Afternoon, Evening, and Night based on timestamps.
- **HRNet Backbone:** Uses `hrnet_w48` pretrained on Cityscapes for robust high-resolution feature extraction.
- **Low-Light Enhancement:** Implements Contrast Limited Adaptive Histogram Equalization (CLAHE) and Fast Non-Local Means Denoising to improve night-time images.
- **Performance Metrics:** Evaluates "Average Confidence" and "Average Number of Objects Detected" to quantify improvement.

## Dataset

- **Source:** Kaggle Day-night Dataset
- **Total Images:** 1,722
- **Distribution:**
  - **Morning (05:00 - 12:00):** 500 images
  - **Afternoon (12:00 - 17:00):** 365 images
  - **Evening (17:00 - 21:00):** 286 images
  - **Night (21:00 - 05:00):** 571 images

## Methodology

1.  **Data Preprocessing:**
    -   Images are paired with timestamps parsed from filename metadata.
    -   Images are categorized into four distinct time periods.
2.  **Model Inference:**
    -   A pre-trained HRNet model (`hrnet_w48_cityscapes_v2.pth`) performs semantic segmentation.
    -   We compute the mean probability (confidence) of detected classes and count unique objects per image.
3.  **Night Enhancement:**
    -   **CLAHE:** Applied to the L-channel of LAB converted images (Clip Limit: 2.0, Tile Grid: 8x8).
    -   **Denoising:** `cv2.fastNlMeansDenoisingColored` (h=10, hColor=10, templateWindowSize=7, searchWindowSize=21).
4.  **Comparison:**
    -   Metrics are calculated for original night images vs. enhanced night images.

## Results

Our experiments show that night-time conditions significantly reduce model confidence and object recall. However, applying enhancement techniques yields measurable improvements.

| Metric | Night (Baseline) | Night (Enhanced) | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg. Confidence** | 0.8210 | **0.8308** | +1.2% |
| **Avg. Objects Detected** | 6.0 | **8.0** | +33% |

*Note: Morning, Afternoon, and Evening baseline metrics generally exceed Night performance, highlighting the impact of illumination.*

## Usage

### Requirements
```txt
torch
torchvision
opencv-python
pillow
matplotlib
numpy
