# Wildfire Detection Demo

## Overview
This is a quick demo project to classify wildfire and non-wildfire images using **OpenCV preprocessing** and a **simple CNN model**.  
We use **500 wildfire images** and **500 non-wildfire images** to train a model and visualize predictions.

## Features
- **Random image sampling**: 500 images per class for a lightweight demo.
- **OpenCV preprocessing**:
  - Convert images to **grayscale** to highlight intensity details.
  - Apply **Canny edge detection** to emphasize boundaries and shapes.
- **CNN model**: Simple Convolutional Neural Network to classify wildfire vs no-wildfire.
- **Visualization**: Compare original, grayscale, and edge-detected images side by side with true and predicted labels.

## Dataset
- Dataset used: [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) from Kaggle.
- 500 images for each class are randomly selected for the demo.

## Quick Start
1. Clone the repository:
   ```bash
   git clone 


2. Install required packages:
   ```bash
   pip install tensorflow keras opencv-python matplotlib scikit-learn

3. Prepare demo dataset folder:
   ```bash
   - wildfiredemo/input/wildfire
   - wildfiredemo/input/no_wildfire
4. Run the notebook:
   ```bash
   - firedetection.ipynb


## **Output**

Trained CNN model (wildfire_demo_cnn.h5) saved.

## **Visualization of predictions:**

- Original image
- Grayscale version (highlights intensity details)
- Edge-detected image (highlights boundaries like fire edges)
- Model predicts wildfire vs no-wildfire with good accuracy on demo data.

## **Conclusion**

- OpenCV preprocessing helps the model focus on important visual features like fire edges, smoke, and burnt areas.
- CNN can classify wildfire vs no-wildfire efficiently even with a small demo dataset.
- Visualization provides a clear understanding of how the model interprets the images.