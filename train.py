from models import CNN, ShallowMobileNet, ShallowResNet
from Dataloaders import CreateDataLoaders
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import numpy as np


def TrainModel(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, verbose = False):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # Run the forward pass
            outputs = model(X.unsqueeze(1).float())  # add unsqueeze to add channel dimension
            loss = criterion(outputs, y[:, 1].long())  # cast to long
            loss_list.append(loss.item())
            
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track the accuracy
            total = y.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y[:, 1].long()).sum().item()  # cast to long
            acc_list.append(correct / total)
            
            if (i+1) % 100 == 0 and verbose:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item(),
                               (correct / total) * 100))
                
    # Test the model
    model.eval()
    y_true = []
    y_pred = []
    y_pred_binary = []  # Add this line
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X.unsqueeze(1).float())  # add unsqueeze to add channel dimension
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y[:, 1].long()).sum().item()  # cast to long

            # Collect true labels and predicted probabilities for AUC calculation
            y_true.extend(y[:, 1].cpu().numpy())
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred.extend(probabilities)
            y_pred_binary.extend((probabilities > 0.5).astype(int))  # Add this line

        test_accuracy = (correct / total) * 100
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(test_accuracy))

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)

        # Calculate F1 score using binary predictions
        f1 = f1_score(y_true, y_pred_binary)  # Update this line

        print('AUC: {:.4f}'.format(auc_score))

    return loss_list, acc_list, test_accuracy, auc_score, fpr, tpr, f1


def CNNClassification(X_train, y_train, X_test, y_test, num_epochs = 10, learning_rate = 0.001, batch_size = 32, preprocess = True):
    train_loader, test_loader = CreateDataLoaders(X_train, y_train, X_test, y_test, batch_size = batch_size, preprocess = preprocess)
    model = CNN()
    loss_list, acc_list, test_accuracy, auc_score, fpr, tpr, f1_score = TrainModel(model, train_loader, test_loader, num_epochs = num_epochs, learning_rate = learning_rate)
    return model, loss_list, acc_list, test_accuracy, auc_score, fpr, tpr, f1_score

def ShallowResNetClassification(X_train, y_train, X_test, y_test, num_epochs=10, learning_rate=0.001, batch_size=32, preprocess=True):
    train_loader, test_loader = CreateDataLoaders(X_train, y_train, X_test, y_test, batch_size=batch_size, preprocess=preprocess)
    model = ShallowResNet()
    loss_list, acc_list, test_accuracy, auc_score, fpr, tpr, f1_score = TrainModel(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    return model, loss_list, acc_list, test_accuracy, auc_score, fpr, tpr, f1_score

def ShallowMobileNetClassification(X_train, y_train, X_test, y_test, num_epochs=10, learning_rate=0.001, batch_size=32, preprocess=True):
    train_loader, test_loader = CreateDataLoaders(X_train, y_train, X_test, y_test, batch_size=batch_size, preprocess=preprocess)
    model = ShallowMobileNet()
    loss_list, acc_list, test_accuracy, auc_score, fpr, tpr, f1_score = TrainModel(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    return model, loss_list, acc_list, test_accuracy, auc_score, fpr, tpr, f1_score