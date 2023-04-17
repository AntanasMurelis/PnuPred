# Interoperable AI: Classification of X-ray Images

## Introduction

The primary goal of this project is to develop an interoperable framework for classifying X-ray images using multiple Convolutional Neural Network (CNN) models. Emphasis is placed on the ease of switching between different models, comparing their performance, and the ability to visualize and understand the models' decision-making process. In this report, we present the methodology, results, and insights obtained during the project.

## Data Preparation and Preprocessing

### Dataset
The dataset used in this project consists of X-ray images with labels indicating the presence or absence of a specific condition. The dataset is divided into a training set and a test set to ensure proper model evaluation.

Dataset can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 

Insert Figure: Example images from the dataset

### Preprocessing
Prior to training the CNN models, the images are preprocessed by resizing, normalizing, and augmenting the dataset. The data augmentation techniques applied include rotation, flipping, and zooming. This step helps to increase the dataset's size and diversity, reducing the risk of overfitting and improving the model's generalization capabilities.

## Methodology

### Interoperable Framework
To facilitate the interoperability of various CNN models, we have designed a unified interface for training and evaluation. This interface allows seamless switching between models and easy comparison of their performance. The framework is designed to handle popular CNN architectures such as VGG, ResNet, and MobileNet.

### CNN Models
For this project, we have chosen to focus on the following CNN models:

1. VGG
2. ResNet
3. Shallow MobileNet

These models were chosen due to their popularity, proven performance in image classification tasks, and diverse architectural design. This variety allows for a thorough comparison and assessment of their suitability for X-ray image classification.

### Training and Evaluation
Each of the selected CNN models is trained using the preprocessed training dataset. The models' performance is then evaluated on the test dataset to determine their accuracy, loss, and other relevant metrics.

Insert Figure: Training and evaluation results for each CNN model

## Model Comparison and Insights

### Performance Comparison
The results of the evaluation process are used to compare the performance of the various CNN models. Factors such as accuracy, loss, and training time are taken into consideration to determine the most suitable model for the given task.

Insert Figure: Performance comparison of CNN models

### Model Interpretability and Visualization
To gain insights into the decision-making process of the CNN models, we employ visualization techniques to analyze the attributions of individual layers. This helps to identify the regions in the input images that the models find most relevant for classification.

Insert Figure: Layer attributions for each CNN model

## Conclusion

The project successfully demonstrates the development of an interoperable framework for classifying X-ray images using multiple CNN models. The unified interface enables easy model switching and performance comparison, allowing for more informed decisions when selecting a suitable model for a given task. The visualization techniques employed in this project provide valuable insights into the models' decision-making processes, further enhancing our understanding of the models and improving their interpretability.