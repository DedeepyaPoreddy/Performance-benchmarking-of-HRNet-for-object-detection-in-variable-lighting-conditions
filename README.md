# Image Classification & Segmentation Sensitivity to Lighting (HRNet)

This project studies how outdoor-scene **semantic segmentation** behaves under different lighting conditions by grouping images by time-of-day (using timestamp metadata), running a pre-trained **HRNet** segmentation model, and reporting proxy performance metrics (confidence + number of detected classes/objects).

## Dataset

- **Source (Kaggle)**: Day-night Dataset: https://www.kaggle.com/datasets/stevemark/daynight-dataset
- In this project, images are further categorized into **morning, afternoon, evening, and night** based on timestamp hour ranges.

## Goals

- Categorize images into **morning/afternoon/evening/night** using timestamps.
- Run semantic segmentation on sampled images per category using a pre-trained HRNet model (no fine-tuning; no ground-truth masks).
- Compare segmentation behavior across time-of-day categories using:
  - **Average confidence** (mean of max class probability per pixel).
  - **Average objects detected** (unique predicted classes excluding background).
- Apply **low-light enhancement + denoising** for night images and re-evaluate.

## Method

### 1) Time-of-day classification
- Parses timestamps (supports multiple timestamp-file formats) and assigns each image to one of four buckets:
  - Morning: 05–12
  - Afternoon: 12–17
  - Evening: 17–21
  - Night: otherwise

### 2) Segmentation model (HRNet)
- Uses TensorFlow Hub model: `google/HRNet/camvid-hrnetv2-w48/1`.
- Produces segmentation masks, converts them to RGB via a class-color legend, and overlays masks on the original images for visualization.

### 3) Metrics (proxy evaluation)
Per image, the notebook computes:
- **Confidence** = mean over pixels of the maximum predicted probability across classes.
- **Objects detected** = number of unique predicted classes in the mask (excluding background).

### 4) Night enhancement (optional)
For night images, the notebook applies:
- HSV histogram equalization (brightness enhancement).
- Non-local means denoising (`fastNlMeansDenoisingColored`).

## Results

### Performance Analysis by Time of Day (HRNet baseline)

- Morning:
  - Average Confidence: 0.9465
  - Average Objects Detected: 7.50
- Afternoon:
  - Average Confidence: 0.8881
  - Average Objects Detected: 8.00
- Evening:
  - Average Confidence: 0.9049
  - Average Objects Detected: 7.50
- Night:
  - Average Confidence: 0.8210
  - Average Objects Detected: 6.00

### Night performance after enhancement + denoising

- Night:
  - Average Confidence: 0.8308
  - Average Objects Detected: 8.00

> Note: The notebook currently evaluates a **small random sample** per category (e.g., `numsamples=2` in the provided code), so these averages reflect the sampled subset unless you modify the notebook to run over all images.

## How to run

1. Open `IA_Project.ipynb` (Google Colab recommended; notebook uses `google.colab.drive`).
2. Mount Google Drive and set your dataset paths:
   - `basedir`
   - `imagedir`
   - `timestampdir`
3. Run all cells to:
   - Build `imagetimepairs` and `timecategories`
   - Load HRNet model from TensorFlow Hub
   - Run time-of-day analysis and view overlays
   - Optionally run night enhancement + denoising and re-check night metrics

## Repository files

- `IA_Project.ipynb` — end-to-end pipeline: timestamp parsing, time-of-day grouping, HRNet inference, metrics, and night enhancements.
- `Image_Analysis_Final_Report.docx` — project write-up (design + discussion).

## Limitations

- No ground-truth masks are available, so metrics are **proxy** measures (not IoU/mAP).
- HRNet is pre-trained on a different dataset/label space; results may not perfectly match outdoor webcam scenes.
- Results depend on the **sampling size** unless you evaluate all images per category.

## Suggested improvements

- Aggregate metrics over **all images** per category and export CSV for reproducibility.
- Add more enhancement baselines for night images and compare systematically.
- If labeled masks become available, compute standard segmentation metrics (IoU per class, mIoU).
