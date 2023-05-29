from datasets import load_dataset 
import torch
import torch.nn as nn
import PIL
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def get_data():
    dataset = load_dataset('mnist', split='train')
    return dataset['image'], dataset['label']

def transforms(examples):
    # each example is a PIL.PngImagePlugin.PngImageFile
    # resize to 28x28
    resize = RandomResizedCrop(size=(28, 28))
    # convert to tensor
    to_tensor = ToTensor()
    # normalize
    normalize = Normalize(mean=[0.5], std=[0.5])
    # compose
    transforms = Compose([resize, to_tensor, normalize])
    examples = [transforms(example) for example in examples]
    examples = torch.stack(examples)
    return examples

def preprocess_data(images, labels):
    # each image is a PIL.PngImagePlugin.PngImageFile

    # open it and turn it into a 2d tensor
    images = transforms(images)

    return images, labels

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # batch is a 4d tensor of shape (batch_size, 1, 28, 28)
        # predict the digit likelihood 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # batch is a 4d tensor of shape (batch_size, 16, 28, 28)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # batch is a 4d tensor of shape (batch_size, 32, 28, 28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # batch is a 4d tensor of shape (batch_size, 32, 14, 14)
        self.fc1 = nn.Linear(in_features=32*14*14, out_features=10)
        # batch is a 2d tensor of shape (batch_size, 10)

        # last steps to get the digit likelihood
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, batch):
        batch = self.conv1(batch)
        batch = self.conv2(batch)
        batch = self.pool(batch)
        batch = batch.view(-1, 32*14*14)
        batch = self.fc1(batch)
        batch = self.softmax(batch)
        return batch 

if __name__ == "__main__":
    images, labels = get_data()
    images, labels = preprocess_data(images, labels)
    num_epochs = 1
    batch_size = 32
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            labels = labels[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch)
            # output is a 2d tensor of shape (batch_size, 10)
            # labels is a 1d tensor of shape (batch_size)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            print("Epoch: ", epoch, " Loss: ", loss.item())

    



