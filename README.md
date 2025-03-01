# Garbage Classification using deep learning

## Overview
This assignment focuses on developing a multimodal deep learning model for garbage classification. The goal is to classify garbage images into one of four categories: Blue, Black, Green, or TTR. The dataset used is the CVPR 2024 Garbage Classification Dataset, which consists of images labeled according to different recycling categories. The project integrates visual and textual data by combining features extracted from images (using ResNet-50) and text descriptions (using DistilBERT) to improve garbage classification accuracy.

## Model Architecture

- Image Feature Extraction:
    
    A pre-trained ResNet-50 model is used as the backbone for extracting visual features from images. The final fully connected layer of ResNet-50 is replaced with an identity layer, producing a 2048-dimensional feature vector.

- Text Feature Extraction: 
    
    A pre-trained DistilBERT model is employed to process text descriptions associated with the images. The output is the embedding of the [CLS] token, which provides a 768-dimensional feature vector summarizing the text.

- Feature Processing:

    Separate branches process image and text features: 
    - Image Branch: Reduces the 2048-dimensional image feature vector to 512 dimensions using a fully connected layer with ReLU activation and dropout.
    - Text Branch: Similarly, reduces the 768-dimensional text feature vector to 512 dimensions with a fully connected layer, ReLU activation, and dropout.

- Fusion and Classification: 

    The processed image and text features are normalized using L2 normalization and concatenated into a single 1024-dimensional vector (512 from each modality). A final linear layer maps this fused feature vector to one of four output classes: Blue, Black, Green, or TTR.

## Training and Hyperparameters
- Optimizer: AdamW
- Learning Rate: 5e-5
- Weight Decay: 0.01
- Batch Size: 32
- Epochs: 10
- Normalization: ImageNet mean and standard deviation
- Loss Function: Cross-Entropy Loss
- Hardware: TALC Cluster (GPU Node)

## Results and Evaluation
### Training Performance
![training and validation loss](/analysis/loss_plot.png)
![training and validation accuracy](/analysis/accuracy_plot.png)
The training results indicate that the model performs well in classifying garbage categories using a multimodal approach. Training loss steadily decreases from 1.1015 to 0.1191 over 15 epochs, while training accuracy improves from 78.91% to 97.31%, demonstrating effective learning on the training data. Validation loss decreases initially, reaching its lowest point at epoch 11 (0.3520), and validation accuracy peaks at 90.06% in epochs 11 and 14, suggesting good generalization with minimal overfitting. However, slight fluctuations in validation metrics toward the end may indicate the need for early stopping to achieve optimal performance.

### Final Evaluation
On the test data, the model achieved an overall accuracy of 85%, with a weighted average F1-score of 0.85, indicating strong performance across all classes. The macro-average precision and recall of 0.85 and 0.84, respectively, suggest balanced performance, though slight variations exist between classes.

Here is detailed break down of performance by class.

#### Confusion Matrix
![confusion matrix](/output-job-35539/confusion_martrix.png)
#### Classification Performance Metrics
![classification metrics](/output-job-35539/performance_matrix.png)

#### Class-wise Performance:

Blue: Precision (0.78) and recall (0.74) indicate moderate performance, with the confusion matrix showing misclassifications primarily into "Black" (114 cases) and "TTR" (58 cases).

Black: The model performs best for this class, with high precision (0.80), recall (0.93), and F1-score (0.86). Misclassifications are minimal, primarily into "Blue" (47 cases).

Green: Precision (0.94) is excellent, but recall (0.88) shows room for improvement. Misclassifications are mainly into "Black" (54 cases).

TTR: Precision (0.90) is high, but recall (0.79) is slightly lower due to misclassifications into "Black" (91 cases) and "Blue" (63 cases).

These findings suggest that while the model performs well overall, further fine-tuning or additional data could help improve differentiation between closely related classes like "Blue" and "Black."


### Note on usage of jupyter notebook vs python script in TALC

We tried to use jupyter notebook to train and evaluate the model but it was taking a long time to execute. So, we switched to python script file. That sped up the training process.

Sample attempt to run jupyter notebook: Job was not finish even after 6 hours and was terminated due to reaching the time limit set in the slurm file. 
![output from trying to run jupyter notebook](/jupyter_notebook_trial.png)
