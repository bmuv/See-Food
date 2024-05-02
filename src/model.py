import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Simple CNN model for benchmarking the performance of different models

# class BenchmarkCNN(nn.Module):
#     def __init__(self):
#         super(BenchmarkCNN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(12)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(24)
#         self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(24)

#         # Fully connected layer
#         self.fc1 = nn.Linear(269664, 1)

#     def forward(self, input):
#         output = F.relu(self.bn1(self.conv1(input)))      
#         output = F.relu(self.bn2(self.conv2(output)))     
#         output = self.pool(output)                        
#         output = F.relu(self.bn4(self.conv4(output)))     
#         output = F.relu(self.bn5(self.conv5(output)))    
#         output = output.view(output.size(0), -1)
#         output = self.fc1(output)

#         # Sigmoid activation for binary classification
#         output = torch.sigmoid(output) # Squeeze the output to match the target's shape


#         return output

# def get_model():
#     return BenchmarkCNN()


# Basic Block for ResNet18 | FROM SCRATCH

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class ResNet18(nn.Module):
#     def __init__(self, block, layers, num_classes=2):
#         super(ResNet18, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # ResNet layers
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
#         # Fully connected layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x

# def get_model():
#     return ResNet18(BasicBlock, [2, 2, 2, 2])



# def get_model(num_classes=2):
#     weights = ResNet18_Weights.IMAGENET1K_V1
#     model = models.resnet18(weights=weights)

#     # Freeze early layers
#     for param in model.parameters():
#         param.requires_grad = False

#     # Replace the final fully connected layer
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Dropout(0.5),
#         nn.Linear(num_ftrs, num_classes)
#     )
    
#     return model


def get_model(num_classes=2):
    """
    Initializes a pretrained MobileNetV2 model with a new classifier layer.

    Args:
        num_classes (int): Number of classes for the final output layer.

    Returns:
        torch.nn.Module: A MobileNetV2 model with a replaced classifier head adapted for the specified number of classes.
    """
    # Load a pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    
    # Replace the classifier head with a new one adjusted for the desired number of classes
    # The classifier[1] is the final fully connected layer in MobileNetV2's default classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

def load_trained_model(weights_path, num_classes=2, use_cuda=False):
    """
    Loads a trained model from a specified path using a given device (CUDA if available).

    Args:
        weights_path (str): Path to the model's state dictionary.
        num_classes (int): Number of classes used in the model. Must match the model when it was saved.
        use_cuda (bool): Flag to determine whether to use CUDA (if available).

    Returns:
        torch.nn.Module: The loaded model, transferred to CUDA if specified and available.
    """
    # Determine the device to use: use CUDA if available and requested; otherwise, use CPU.
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    # Initialize the model architecture with the appropriate number of classes
    model = get_model(num_classes=num_classes)
    
    # Load the trained model weights into the model
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Transfer the model to the designated device
    model.to(device)
    
    # Set the model to evaluation mode, which disables dropout and batch normalization layers during inference
    model.eval()
    
    return model