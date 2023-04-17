import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from train import CNNClassification, ShallowResNetClassification, ShallowMobileNetClassification
from Dataloaders import CreateDataLoaders


def get_predictions(model, dataloader, return_probs=False):
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda().unsqueeze(1), labels.cuda()
            outputs = model(inputs)
            if return_probs:
                probs = F.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    if return_probs:
        return predictions, np.array(probabilities), true_labels
    else:
        return predictions, true_labels


def optimize_hyperparameters(X, y, model_type='cnn', num_epochs=10, k_folds=5):

    X = np.array(X)
    y = np.array(y)

    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]

    best_parameters = None
    best_auc = 0

    kfold = KFold(n_splits=k_folds, shuffle = True)

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            aucs = []

            for train_index, val_index in kfold.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                if model_type == 'cnn':
                    model_function = CNNClassification
                elif model_type == 'resnet':
                    model_function = ShallowResNetClassification
                elif model_type == 'mobilenet':
                    model_function = ShallowMobileNetClassification
                else:
                    raise ValueError("Invalid model type. Must be 'cnn', 'resnet', or 'mobilenet'.")

                model, _, _ = model_function(
                    X_train, y_train, X_val, y_val,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    preprocess=True
                )

                train_loader, val_loader = CreateDataLoaders(
                    X_train, y_train, X_val, y_val,
                    batch_size=batch_size,
                    preprocess=True
                )

                _, prob_preds, true_labels = get_predictions(model, val_loader, return_probs=True)
                true_labels = np.array(true_labels)
                prob_pred = np.array(prob_preds)
                auc_score = roc_auc_score(true_labels, prob_preds)
                aucs.append(auc_score)

            average_auc = np.mean(aucs)

            if average_auc > best_auc:
                best_auc = average_auc
                best_parameters = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "auc": best_auc
                }

            print("Model Type: {}, Learning Rate: {}, Batch Size: {}, Validation AUC: {:.4f}".format(
                model_type, learning_rate, batch_size, average_auc
            ))

    return best_parameters
