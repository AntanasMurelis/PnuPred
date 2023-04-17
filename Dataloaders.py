from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from misc import preprocess_image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# functor to preprocess images fro CNN classification

class PreprocessImage(object):
    def __call__(self, image):
        return preprocess_image(image)

class PneumoniaDataset(Dataset):
    """Pneumonia dataset."""
    
    def __init__(self, X, y, transform = None, preprocess = False):
        self.X = X
        self.y = y
        self.transform = transform
        if(preprocess):
            self.preprocess = PreprocessImage()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.X[idx]
        y = self.y[idx]
        y = F.one_hot(torch.tensor(y), num_classes = 2)
        if self.transform:
            X = self.transform(X)
        
        return X, y
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, X):
        return torch.from_numpy(X).float()


def CreateDataLoaders(X_train, y_train, X_test, y_test, batch_size = 32, preprocess = True):

    train_dataset = PneumoniaDataset(X_train, y_train, transform = ToTensor())
    test_dataset = PneumoniaDataset(X_test, y_test, transform = ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    return train_loader, test_loader