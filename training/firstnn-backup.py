import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CNN import CNNet

##ToTensor(): Converts images to PyTorch tensors (shape: [C, H, W]).
##Normalize(mean, std): Normalizes pixel values to range [-1, 1] for better training stability.
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

##4 images per batch 
batch_size = 4

##CIFAR-10: A dataset of 60,000 32×32 color images in 10 classes.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
##DataLoader: Loads data in batches for training/testing.
##batch_size=4: Each batch has 4 images.
##shuffle=True: Randomizes training data order.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

if torch.cuda.is_available():
    print( "GPU available")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Running on CPU.")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

##Create CNN
window=10
joints=15
num_classes=50

net = CNNet(window_size=window, num_joints=joints, num_class=num_classes, drop_prob=0.5)

##Loss Function & Optimizer
    #CrossEntropyLoss: Combines LogSoftmax + NLLLoss.
    #SGD: Stochastic Gradient Descent with momentum.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

##Training Loop
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) #Forward Pass- compute predictions
        loss = criterion(outputs, labels)
        loss.backward() #Backward Pass - compute gradients
        optimizer.step() #updates weights

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

##Saves trained weights (next two lines)
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = CNNet()

##Loads weights and predicts on test images
net.load_state_dict(torch.load(PATH, weights_only=True))
outputs = net(images)
_, predicted = torch.max(outputs, 1)


print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

##Accuracy Calculation
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')