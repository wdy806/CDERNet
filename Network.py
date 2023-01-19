from torch import nn
from torchvision import models


class ResNet18_for_embedding(nn.Module):
    def __init__(self, pretrained=True, embedding_size=256, drop_rate=0):
        super(ResNet18_for_embedding, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, embedding_size)


    def forward(self, x):
        x = self.features(x)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out


class Decoder(nn.Module):
    def __init__(self, embedding_size, class_num):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(embedding_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, class_num)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):   
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)
        return out


class CDERNet(nn.Module):
    def __init__(self, pretrained=True, embedding_size=256, class_num=3, drop_rate=0):
        super(CDERNet, self).__init__()
        self.resnet18_for_embedding = ResNet18_for_embedding(pretrained, embedding_size, drop_rate)
        self.fcnet = Decoder(embedding_size, class_num)

    def forward(self, x):
        x = self.resnet18_for_embedding(x)

        out = self.fcnet(x)

        return x, out


if __name__=='__main__':
    model = CDERNet()
    print(model)
