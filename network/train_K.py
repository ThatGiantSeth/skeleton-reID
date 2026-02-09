from pathlib import Path
import sys
import torch
from CNN import CNNet
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np

from preprocessing import get_arrays, normalize_skeleton

WINDOW_SIZE = 50
STRIDE = 2  # Reduced from 3 to create more training windows
EPOCHS = 10  
LR = 0.001  # Increased from 0.0003 for faster learning
K_FOLDS = 5  # Number of folds for K-fold cross validation

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


class SkeletonDatasetSubset(torch.utils.data.Dataset):
    """Dataset that uses a subset of indices from the base dataset."""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


## main program with K-fold cross validation
def main():
    
    # get training data from both data and data_val directories
    train_arrays, train_labels, people = get_arrays("./data")
    val_arrays, val_labels, _ = get_arrays("./data_val")
    
    # Combine all data for K-fold cross validation
    all_arrays = train_arrays + val_arrays
    all_labels = train_labels + val_labels
    
    num_classes = len(set(all_labels))
    
    # Convert labels to numpy array for K-fold split
    all_labels_np = np.array(all_labels)
    
    # use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize K-fold cross validation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    # K-fold cross validation
    for fold, (train_indices, val_indices) in enumerate(kfold.split(all_labels_np), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{K_FOLDS}")
        print(f"{'='*60}")
        
        # Create train dataset with train indices
        train_dataset = SkeletonDataset(
            [all_arrays[i] for i in train_indices],
            [all_labels[i] for i in train_indices],
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            normalize=True
        )
        
        # Create validation dataset with val indices
        val_dataset = SkeletonDataset(
            [all_arrays[i] for i in val_indices],
            [all_labels[i] for i in val_indices],
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            normalize=True
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # Create CNN for this fold
        net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=0.3).to(device)
        
        # Loss function & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)

        best_fold_acc = 0.0
        fold_model_file = f"skeleton_model_fold_{fold}.pth"

        # Training loop for this fold
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            net.train()  # Set to training mode
            
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, encode_out = net(inputs)  # Forward Pass
                loss = criterion(outputs, labels)
                loss.backward()  # Backward Pass
                optimizer.step()  # Update weights

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:    # print every 50 mini-batches
                    print(f'[Epoch {epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0

            # Validate after each epoch
            val_acc = validate(net, val_loader, criterion, device, epoch)
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save(net.state_dict(), fold_model_file)
                print(f"New best fold acc {best_fold_acc:.2f}% saved to {fold_model_file}")
        
        fold_results.append({
            'fold': fold,
            'best_acc': best_fold_acc,
            'model_file': fold_model_file
        })
        
        # Load best model for this fold and print confusion matrix
        net.load_state_dict(torch.load(fold_model_file, map_location=device))
        print(f"\nLoaded best model for fold {fold} (acc {best_fold_acc:.2f}%) from {fold_model_file}")
        print_confusion_matrix(net, val_loader, device, num_classes)

    # Print summary of all folds
    print(f"\n{'='*60}")
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*60}")
    for result in fold_results:
        print(f"Fold {result['fold']}: {result['best_acc']:.2f}%")
    
    avg_acc = np.mean([result['best_acc'] for result in fold_results])
    std_acc = np.std([result['best_acc'] for result in fold_results])
    print(f"\nAverage accuracy: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")


## Validation function
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
    print(f'Validation - Epoch {epoch + 1} accuracy: {accuracy:.2f}%')
    return accuracy


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
