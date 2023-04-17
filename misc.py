import cv2
import os
import random
import torch

# Preprocess lung images for CNN classification
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

def LoadLungImages(path):
    X = []
    y = []
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            image = cv2.imread(os.path.join(path, folder, file))
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
