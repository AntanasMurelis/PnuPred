from models import CNN, ShallowMobileNet, ShallowResNet
from Dataloaders import CreateDataLoaders
import torch
import torch.nn as nn
import torch.optim as optim

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
            
            if (i+1) % 100 == 0 and verbose == True:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item(),
                               (correct / total) * 100))
                
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X.unsqueeze(1).float())  # add unsqueeze to add channel dimension
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y[:, 1].long()).sum().item()  # cast to long

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
        
    # Save the model and plot
    torch.save(model.state_dict(), 'model.ckpt')
    
    return loss_list, acc_list


def CNNClassification(X_train, y_train, X_test, y_test, num_epochs = 10, learning_rate = 0.001, batch_size = 32, preprocess = True):
    train_loader, test_loader = CreateDataLoaders(X_train, y_train, X_test, y_test, batch_size = batch_size, preprocess = preprocess)
    model = CNN()
    loss_list, acc_list = TrainModel(model, train_loader, test_loader, num_epochs = num_epochs, learning_rate = learning_rate)
    return model, loss_list, acc_list

def ShallowResNetClassification(X_train, y_train, X_test, y_test, num_epochs=10, learning_rate=0.001, batch_size=32, preprocess=True):
    train_loader, test_loader = CreateDataLoaders(X_train, y_train, X_test, y_test, batch_size=batch_size, preprocess=preprocess)
    model = ShallowResNet()
    loss_list, acc_list = TrainModel(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    return model, loss_list, acc_list

def ShallowMobileNetClassification(X_train, y_train, X_test, y_test, num_epochs=10, learning_rate=0.001, batch_size=32, preprocess=True):
    train_loader, test_loader = CreateDataLoaders(X_train, y_train, X_test, y_test, batch_size=batch_size, preprocess=preprocess)
    model = ShallowMobileNet()
    loss_list, acc_list = TrainModel(model, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    return model, loss_list, acc_list