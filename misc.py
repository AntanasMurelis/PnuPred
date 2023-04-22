import cv2
import os
import random
import torch
import matplotlib.pyplot as plt

# Preprocess lung images for CNN classification
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

import os

def LoadLungImages(path):
    X = []
    y = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file != '.DS_Store':
                    image = cv2.imread(os.path.join(folder_path, file))
                    X.append(preprocess_image(image))
                    if folder == "NORMAL":
                        y.append(0)
                    else:
                        y.append(1)
    return X, y



def get_probabilities(model, dataloader):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.cuda().unsqueeze(1)
            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1)[:, 1]  # Get the probabilities of the positive class
            probabilities.extend(preds.cpu().numpy())

    return probabilities

def sample_images_with_labels(X, y, label, num_samples):
    """
    Randomly sample a specified number of images from a dataset with a given label.
    
    Args:
    - X: list of numpy arrays representing the images
    - y: list of labels corresponding to each image
    - label: integer label of images to sample (0 or 1)
    - num_samples: integer number of images to sample
    
    Returns:
    - images: list of numpy arrays representing the sampled images
    - labels: list of labels corresponding to each sampled image
    """
    indices = [i for i in range(len(y)) if y[i] == label]
    samples = random.sample(indices, num_samples)
    images = [X[i] for i in samples]
    labels = [y[i] for i in samples]
    return images, labels

def display_side_by_side(healthy_images, pneumonia_images, n=1):
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

    for i in range(n):
        if n == 1:
            ax_h = axes[0]
            ax_p = axes[1]
        else:
            ax_h = axes[i, 0]
            ax_p = axes[i, 1]

        ax_h.imshow(healthy_images[i], cmap='gray')
        ax_h.set_title('Healthy')
        ax_h.axis('off')

        ax_p.imshow(pneumonia_images[i], cmap='gray')
        ax_p.set_title('Pneumonia')
        ax_p.axis('off')

    plt.tight_layout()
    plt.show()
