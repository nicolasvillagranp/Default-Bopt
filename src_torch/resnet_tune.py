from src_torch.utils import load_data, accuracy
from src_torch.resnet import ResNet18Classifier  # Assuming the ResNet model is in this file
import torch
from torch.utils.data import DataLoader



device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(lr_model, lr_linear, dropout, batch_size):
    # Train with given hyperparams.
    train_dataloader, val_dataloader = load_data(path="data", batch_size=batch_size)
    model = ResNet18Classifier(10, p=dropout)
    model = train(model, train_dataloader,  15, lr_model, lr_linear)
    total = len(val_dataloader)
    # Test model to optimize.
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)  
            correct += accuracy(logits, labels)
    
    accuracy = correct / total
    return accuracy



def train(model: torch.nn.Module, train_dataloader: DataLoader, epochs: int = 10, 
                              lr_model: float = 1e-4, lr_linear: float = 1e-3)-> ResNet18Classifier:
        """
        This function trains the model with a progress bar and plots training/validation loss.

        Args:
            model: The model to train.
            train_dataloader: Dataloader for training data.
            val_dataloader: Dataloader for validation data.
            epochs: Number of training epochs. Default: 10.
            learning_rate: Learning rate for the optimizer. Default: 1e-3.
            device: Device to train the model on ('cuda' or 'cpu'). Default: 'cuda' if available.
        """
        model.train() 
        model.to(device)  
        optimizer = torch.optim.Adam([{'params': model.res.parameters(), 'lr': lr_model}, 
                                      {'params': model.linear.parameters(), 'lr': lr_linear}])  
        criterion = torch.nn.CrossEntropyLoss() 

        for _ in range(epochs):
            # Initialize progress bar for the training loop
            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(
                    device
                )  # Send data to device
                optimizer.zero_grad()  
                outputs = model(images.squeeze())  
                loss = criterion(outputs, labels)  

                loss.backward()  
                optimizer.step() 
        return model


if __name__ == '__main__':
    # Example
    acc = evaluate(1e-4, 1e-3, 0.2, 128)
    print(acc)



