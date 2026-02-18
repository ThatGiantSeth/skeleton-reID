import torch
from CNN import CNNet
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import os
import json

from preprocessing import get_arrays, normalize_skeleton

WINDOW_SIZE = 50
STRIDE = 2
EPOCHS = 30
LR = 0.0002
K_FOLDS = 7
DROP_PROB = 0.5
BATCH_SIZE = 8

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



## main program
def main():
    
    # get training data and create dataset
    train_arrays, train_labels, people = get_arrays("./data_split", trim_front=0)
    
    with open('people_map.json', 'w') as f:
        json.dump(people, f, indent = 4)
    
    fold_models_dir = "fold_models"
    os.makedirs(fold_models_dir, exist_ok=True)
    
    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    num_classes = len(set(train_labels))
    
    fold_results = []
    best_overall_acc = 0.0
    best_model_file = "skeleton_model_best_k.pth"
    
    # Loop through each fold
    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_arrays, train_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{K_FOLDS}")
        print(f"{'='*50}")
        
        # Split samples into train and validation
        fold_train_arrays = [train_arrays[i] for i in train_indices]
        fold_train_labels = [train_labels[i] for i in train_indices]
        fold_val_arrays = [train_arrays[i] for i in val_indices]
        fold_val_labels = [train_labels[i] for i in val_indices]
        
        print(f"Train samples: {len(fold_train_arrays)}, Val samples: {len(fold_val_arrays)}")
        
        # Create datasets
        train_dataset = SkeletonDataset(fold_train_arrays, fold_train_labels, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True)
        val_dataset = SkeletonDataset(fold_val_arrays, fold_val_labels, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Create CNN
        net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=DROP_PROB).to(device)
    
        ##Loss Function & Optimizer
        #CrossEntropyLoss: Combines LogSoftmax + NLLLoss.
        #SGD: Stochastic Gradient Descent with momentum. - would Adam be better? see Joe's example
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        
        best_fold_acc = 0.0
        
        # Training loop
        for epoch in range(EPOCHS):
            net.train()
            print(f"  Epoch {epoch+1}/{EPOCHS}")
            
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs, _ = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 50 == 49:
                    print(f'    [{i + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0
            
            # Validate on this fold
            fold_val_acc = validate(net, val_loader, criterion, device, epoch)
            if fold_val_acc > best_fold_acc:
                best_fold_acc = fold_val_acc
                
                # Save fold's best model
                fold_model_file = os.path.join(fold_models_dir, f"fold_{fold}_best.pth")
                torch.save(net.state_dict(), fold_model_file)
                print(f"  Fold {fold + 1} best model saved: {best_fold_acc:.2f}%")
                
                # Save best overall model
                if fold_val_acc > best_overall_acc:
                    best_overall_acc = fold_val_acc
                    torch.save(net.state_dict(), best_model_file)
                    print(f"  New overall best acc {best_overall_acc:.2f}% saved to {best_model_file}")
        
        fold_results.append(best_fold_acc)
        print(f"Fold {fold + 1} best accuracy: {best_fold_acc:.2f}%")
        
        # Print confusion matrix
        print(f"\nConfusion Matrix for Fold {fold + 1}:")
        print_confusion_matrix(net, val_loader, device, num_classes)
    
    # Print summary
    print(f"\n{'='*50}")
    print("K-Fold Cross-Validation Summary")
    print(f"{'='*50}")
    for fold, acc in enumerate(fold_results):
        print(f"Fold {fold + 1}: {acc:.2f}%")
    print(f"Mean Accuracy: {sum(fold_results) / len(fold_results):.2f}%")
    print(f"Best Accuracy: {max(fold_results):.2f}%")
    print(f"\nBest model saved to {best_model_file}")
    
    print(f"\n{'='*50}")
    print("Final Model Evaluation on Test Set:")
    print(f"{'='*50}")
    
    # Create CNN
    net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=DROP_PROB).to(device)
    net.load_state_dict(torch.load(best_model_file, map_location=device))
    
    # Load separate test data
    test_arrays, test_labels, _ = get_arrays("./data_val_split", trim_front=0)
    print(f"Loaded {len(test_arrays)} test samples")
    
    test_dataset = SkeletonDataset(test_arrays, test_labels, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Compute test accuracy
    test_acc = validate(net, test_loader, criterion, device, epoch=0)
    print(f"\nFinal Test Accuracy on data_val: {test_acc:.2f}%")
    
    # Print confusion matrix
    print("\nConfusion Matrix on Test Set (data_val):")
    print_confusion_matrix(net, test_loader, device, num_classes)

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