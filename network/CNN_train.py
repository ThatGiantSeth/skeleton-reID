import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transformers
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import tqdm
import os
import sys
import argparse
import CNN as cnn





parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store')
parser.add_argument('--save', action='store')
parser.add_argument('--epoch', action='store', default=5000, type=int)
parser.add_argument('--check', action='store', default=100, type=int)
parser.add_argument('--lr', action='store', default=0.0005, type=float)
parser.add_argument('--joints', action='store', default=13, type=int)
parser.add_argument('--window', action='store', default=10, type=int)
parser.add_argument('--nclass', action='store', default=10, type=int)
args = parser.parse_args()






        
class Feeder(torch.utils.data.Dataset):

	def __init__(self, data, labels):
		self.data=data
		self.labels=labels
		
	def __len__(self):
		x=self.labels.size()
		return x[0]

	def __iter__(self):
		return self

	def __getitem__(self, index):
		return self.data[index], self.labels[index]

# Dynamic function to contain data preprocessing code
def data_process(x):
    if len(x.shape)==5:
        x=x[:,:,:,:,0]
    #x=x.permute([0,2,3,1])
    #x=torch.unsqueeze(x,4)
    return x


      
data_root=args.data
save_source=args.save
num_classes=10
num_epochs=args.epoch
checkpoint_interval=args.check
lr=args.lr
criterion = nn.CrossEntropyLoss()
device='cuda'



for fold in range(0,10,1):
    
    model = cnn.CNNet(window_size=args.window, num_joints=args.joints, num_class=args.nclass, device=device)

        
    model.train()
    
    # Initialize model
    model.to(device)
    model.double()

    # Define optimizer
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Make save directories
    
    save_root=save_source+"fold"+str(fold)+"/"
    # Make save directory if it DNE
    isExist = os.path.exists(save_root)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(save_root)
    
    save_raw=save_root+"Raw/"
    # Make save directory if it DNE
    isExist = os.path.exists(save_raw)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(save_raw)
    
    save_check=save_root+"Checkpoints/"
    # Make save directory if it DNE
    isExist = os.path.exists(save_check)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(save_check)
    
    # Load/process train data
    x=np.load(data_root+"train_data_fold-"+str(fold)+".npy")
    y=np.load(data_root+"train_labels_fold-"+str(fold)+".npy")
    plt.hist(y+1, bins=int(np.max(y)+1))
    plt.title("Train Data")
    plt.savefig(save_root+"histogram_train.png")
    plt.clf()
    x=torch.tensor(x, device=device)
    x=data_process(x)
    print(x.shape)
    
    y=torch.tensor(y, device=device)
    y=y.type(torch.long)


    # Load/process test data
    x_test=np.load(data_root+"test_data_fold-"+str(fold)+".npy")
    y_test=np.load(data_root+"test_labels_fold-"+str(fold)+".npy")
    plt.hist(y_test+1, bins=int(np.max(y_test)+1))
    plt.title("Test Data")
    plt.savefig(save_root+"histogram_test.png")
    plt.clf()
    x_test=torch.tensor(x_test, device=device)
    x_test=data_process(x_test)
    y_test=torch.tensor(y_test, device=device)
    y_test=y_test.type(torch.long)
    
    dataloader = torch.utils.data.DataLoader(Feeder(x,y), batch_size=64, shuffle=True, num_workers=0)
    soft=nn.Softmax(dim=1)
    
    train_acc_arr=[]
    test_acc_arr=[]
    loss_arr=[]
    
    for e in range(num_epochs):
        loss_val=0
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs,_ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_val=loss_val+loss.cpu().detach().numpy()
       
        
        if (e+1)%checkpoint_interval==0:
            torch.save(model.state_dict(),save_check+"epoch_"+str(e))
        
        
        # Get test accuracy
        model.eval()
        outputs,_=model(x_test)
        _, predicted = torch.max(soft(outputs.data), 1)
        test_acc = (torch.sum(predicted == y_test))
        test_acc=test_acc/y_test.shape[0]
        test_acc=test_acc.cpu().detach().numpy()
        
        
        # Get train accuracy
        outputs,_=model(x)
        _, predicted = torch.max(soft(outputs.data), 1)
        train_acc = (torch.sum(predicted == y))       
        train_acc=train_acc/y.shape[0]
        train_acc=train_acc.cpu().detach().numpy()
        model.train()
        # Train metrics outputs
        print("Fold: ",fold,"  Epoch: ",e,"  Train Accuracy: ",train_acc,"  Test Accuracy: ",test_acc,"   Loss: ",loss_val/i)
        
        # Log train metrics
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)
        loss_arr.append(loss_val/i)
    
    # Save train metrics
    np.save(save_raw+"train_accuracy",np.array(train_acc_arr))
    np.save(save_raw+"test_accuracy",np.array(test_acc_arr))
    np.save(save_raw+"train_loss",np.array(loss_arr))
    
    # Plot train metrics
    x=range(0,num_epochs,1)

    plt.plot(x,np.array(train_acc_arr), label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.title("Train Accuracy/Epoch")
    plt.legend()
    plt.savefig(save_root+"train_accuracy.png")
    plt.clf()
    
    plt.plot(x,np.array(test_acc_arr), label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy/Epoch")
    plt.legend()
    plt.savefig(save_root+"test_accuracy.png")
    plt.clf()
    
    plt.plot(x,np.array(loss_arr), label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss/Epoch")
    plt.legend()
    plt.savefig(save_root+"train_loss.png")
    plt.clf()


