import torch
import torch.nn as nn
import torch.nn.functional as F

# Build Network
class OtherLeNet(nn.Module):

    def __init__(self, num_classes):
        emb_dim = 490
        '''
        Define the initialization function of LeNet, this function defines
        the basic structure of the neural network
        '''
        super(OtherLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.embedding = nn.Conv2d(16, 10, kernel_size=5, stride=1, padding=2)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, multi_res=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if multi_res:
            embeddings_2 = torch.flatten(x, 1)
        
        embeddings = torch.flatten(self.embedding(x), 1)
        logits = self.classifier(embeddings)
        if multi_res:
            return logits, embeddings, embeddings_2
        else:
            return logits, embeddings

    def name(self):
        return 'lenet-magnet'
