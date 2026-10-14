# EndoVision: Uncertainty-Aware Endometriosis Image Classification

EndoVision is a deep learning-based medical image classification project that uses **EfficientNetB0 and Transfer Learning** to classify images into two categories: endometriosis and no endometriosis.

The project combines **Monte Carlo Dropout** for predictive uncertainty estimation and **Grad-CAM** for visual explainability. The system can flag high-uncertainty predictions for manual review, supporting a more interpretable research workflow.

## Features

- Binary classification of endometriosis and no-endometriosis images
- Transfer learning using pretrained EfficientNetB0
- Image preprocessing and data augmentation
- Model training and fine-tuning
- Monte Carlo Dropout for uncertainty estimation
- Uncertainty-based flagging for manual review
- Grad-CAM heatmaps for visual interpretation
- Confusion matrix and classification report
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Export of image-level predictions and uncertainty scores to CSV

## Dataset

The project uses a labeled image dataset containing two classes.

### Dataset Structure

```text
dataset/
├── endometriosis/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no_endometriosis/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Update the dataset path in the configuration file according to your local setup.

## Dataset Split

| Subset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 15% |
| Testing | 25% |

The validation set is used for model selection and, where implemented, uncertainty-threshold selection. The test set is reserved for final performance evaluation.

If multiple images originate from the same patient or surgical procedure, patient-level splitting should be considered to reduce the risk of data leakage.

## Model Architecture

EndoVision uses **EfficientNetB0**, pretrained on ImageNet, as the feature extraction backbone.

The classification pipeline consists of:

1. EfficientNetB0 feature extraction
2. Global Average Pooling
3. Dropout layer
4. Sigmoid classification output

Training is performed in stages, beginning with the classification head and followed by fine-tuning selected layers of the pretrained network.

## Uncertainty Estimation

Monte Carlo Dropout estimates predictive variability by performing multiple stochastic forward passes with dropout enabled during inference.

The system can calculate:

- **Mean probability:** Average predicted probability across repeated passes.
- **Uncertainty score:** Standard deviation of the predicted probabilities.
- **Manual-review flag:** Identifies predictions that exceed a threshold selected using validation data.

The current project configuration uses 30 stochastic predictions and a threshold based on the 90th percentile of validation uncertainty, if enabled in the implementation.

Uncertainty is an estimate of predictive variability and does not guarantee that a prediction is correct or incorrect.

## Grad-CAM Explainability

Grad-CAM generates a heatmap highlighting image regions that contribute to the model's classification score.

These heatmaps help users inspect the model's visual attention and understand its predictions.

**Note:** Grad-CAM is not a lesion segmentation method and does not confirm the presence or location of endometriosis.

## Results

The model was evaluated on a held-out test set of **256 images**, achieving an accuracy of **92.19%**.

### Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | 92.19% |
| Precision | 92.16% |
| Recall | 92.21% |
| Endometriosis F1-score | 92.48% |
| ROC-AUC | 0.9819 |


## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/Kumaaragurubaran-del/Endovision.git
cd Endovision
```

Install the dependencies:

```bash
pip install -r requirements.txt
```



