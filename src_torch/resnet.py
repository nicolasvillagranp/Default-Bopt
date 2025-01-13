import torch
import torch.nn as nn
import torchvision.models as models


"""
In practice, the community tends to overwrite the last linear layer of
the model.
As we want to avoid overfitting, we include a dropout 
regularizer.
In future tests we may include a second linear 
layer and a ReLU right after the pooling.
"""

class ResNet18Classifier(nn.Module):
    """
    This class implements a ResNet-18 based model.
    """
    def __init__(self, output_size: int = 10, hidden_size: int = 512,  p: float = 0.2):
        """
        Constructor of custom ResNet18Classifier class.

        Args:
            output_size: Output size for the model (e.g., number of classes).
            hidden_size: Intermediate size for hidden activation.
        """

        super().__init__()
        self.res = models.resnet18(pretrained=True)
        self.hidden = torch.nn.Linear(self.res.fc.out_features, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p)
        self.classifier = torch.nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass.

        Args:
            inputs: Batch of images. Dimensions: [batch, channels, height, width].

        Returns:
            Batch of logits. Dimensions: [batch, number of classes].
        """
        # Forward pass through the ResNet-18 model
        features = self.res(inputs) 
        features = self.dropout(features)
        hidden_features = self.relu(self.hidden(features))
        return self.classifier(hidden_features)



