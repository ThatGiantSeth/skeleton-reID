import torch
from CNN import CNNet
import torch.nn as nn
import torch.optim as optim

from preprocessing import get_arrays, normalize_skeleton

WINDOW_SIZE = 50
STRIDE = 3
EPOCHS = 20
LR = 0.0003

# create custom dataset from recorded arrays
class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, skeletons, labels, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True):
        self.xdata = []
        self.ydata = []

        for data, label in zip(skeletons, labels):
            ## preprocessing
            if normalize:
                data = normalize_skeleton(data)

            num_frames = data.shape[0]
            
            ## apply sliding window to the data when adding to the dataset
            for window_start in range(0, num_frames - window_size + 1, stride):

                window = data[window_start:window_start + window_size]
                
                tensor = torch.from_numpy(window).float()
                tensor = tensor.permute(2, 0, 1)
                
                self.xdata.append(tensor)
                self.ydata.append(label)
        
        print(f"Created {len(self.xdata)} windows from data")
            
    def __len__(self):
        return len(self.xdata)
    
    def __getitem__(self, idx):
        return self.xdata[idx], self.ydata[idx]



## main program (probably needs to be refactored into more helper functions)
def main():
    
    # get training data and create dataset
    train_arrays, train_labels, people = get_arrays("./data")

    train_dataset = SkeletonDataset(train_arrays, train_labels, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # same operations as above but loading data to validate during training
    val_arrays, val_labels, _ = get_arrays("./data_val")
    val_dataset = SkeletonDataset(val_arrays, val_labels, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


    # Create CNN
    num_classes = len(set(train_labels))
    net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=0.5).to(device)

    ## Below code is mostly modified from the standard PyTorch training example for image classification
        # This excludes the confusion matrix function, see notes below
    
    ##Loss Function & Optimizer
        #CrossEntropyLoss: Combines LogSoftmax + NLLLoss.
        #SGD: Stochastic Gradient Descent with momentum.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

    best_acc = 0.0
    model_file = "skeleton_model_best.pth"

    ##Training Loop
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        net.train()  # Set to training mode
        print(f"\nStarting epoch {epoch+1}")

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, encode_out = net(inputs) #Forward Pass- compute predictions
            loss = criterion(outputs, labels)
            loss.backward() #Backward Pass - compute gradients
            optimizer.step() #updates weights

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0

        ## use a separate folder of numpy arrays to test the accuracy after each epoch
            # also save the model with the highest accuracy
        val_acc = validate(net, val_loader, criterion, device, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), model_file)
            print(f"New best acc {best_acc:.2f}% saved to {model_file}")

    print('Finished Training')

    ## this is just for inspecting the final model's confusion matrix for debugging and checking model biases
    net.load_state_dict(torch.load(model_file, map_location=device))
    print(f"Loaded best model (acc {best_acc:.2f}%) from {model_file}")
    print_confusion_matrix(net, val_loader, device, num_classes)

## majority of this function also taken from PyTorch example
def validate(net, val_loader, criterion, device, epoch):
    net.eval()
    correct = 0
    total = 0
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # calculate outputs by running validation data through the network
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\nValidation - Epoch {epoch + 1} accuracy: {accuracy:.2f}%')
    return accuracy

#
# copilot wrote this function on its own, i have no idea how it works, but it's only used for debugging and bias testing
#
def print_confusion_matrix(net, val_loader, device, num_classes):
    """Compute and print a confusion matrix to spot class bias."""
    net.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = net(inputs)
            preds = outputs.argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf[t.long(), p.long()] += 1

    # try to recover class names from the label mapping used in training
    # by looking at the dataset folder names; fallback to indices
    # Note: we assume the mapping in get_arrays was alphabetical by filename order
    class_names = [f"class_{i}" for i in range(num_classes)]

    print("Confusion matrix (rows=true, cols=pred):")
    header = ["true/pred"] + class_names
    rows = []
    conf_np = conf.cpu().numpy()
    for idx, row in enumerate(conf_np):
        rows.append([class_names[idx]] + row.tolist())

    # pretty print as a simple table
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]
    def fmt_row(r):
        return " | ".join(str(x).ljust(w) for x, w in zip(r, col_widths))

    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))
          
            
if __name__ == "__main__":
    main()