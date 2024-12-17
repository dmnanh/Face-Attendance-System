import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
        def __init__(self):
            super(EmbeddingNet, self).__init__()
            # Convolutional layers
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Fully connected layers
            self.fc1 = nn.Linear(256 * 16 * 16, 256)  # Adjusted to match the flattened size
            self.bn_fc1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)

        def forward(self, x):
            # Convolutional layers
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)

            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)

            # Flatten the tensor
            x = x.view(x.size(0), -1)

            # Fully connected layers
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = self.fc2(x)
            return x
    

class TripletNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
            # Generate embeddings for each input
        anchor_embedding = self.embedding_net(anchor)
        positive_embedding = self.embedding_net(positive)
        negative_embedding = self.embedding_net(negative)
        return anchor_embedding, positive_embedding, negative_embedding