import torch
from CNN import CNNet
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import os
import json

from preprocessing import combine_recordings, normalize_skeleton, window_sequence

## maybe make argument processor for these options for HPCs

WINDOW_SIZE = 10
STRIDE = 2
EPOCHS = 50
LR = 0.0008
K_FOLDS = 10
DROP_PROB = 0.4
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-5

# create custom dataset from recorded arrays
class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, windows, labels):
        self.xdata = []
        self.ydata = []

        for window, label in zip(windows, labels):
            tensor = torch.from_numpy(window).float()
            tensor = tensor.permute(2, 0, 1)
            self.xdata.append(tensor)
            self.ydata.append(int(label))
        
        print(f"Created {len(self.xdata)} windows from data")
            
    def __len__(self):
        return len(self.xdata)
    
    def __getitem__(self, idx):
        return self.xdata[idx], self.ydata[idx]

## main program
def main():
    
    # combine recordings and normalize
    train_x, train_y, people = combine_recordings("./data", trim_front=499)
    train_x = normalize_skeleton(train_x)
    
    #window data
    windows, window_labels = window_sequence(train_x, train_y, window_size=WINDOW_SIZE, stride=STRIDE)
    print(f"Total training windows for k-fold: {len(windows)}")
    
    # save list of people
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
    
    num_classes = len(people)
    class_names = [f"class_{i}" for i in range(num_classes)]
    for person_name, class_idx in people.items():
        if 0 <= class_idx < num_classes:
            class_names[class_idx] = person_name
    
    fold_results = []
    
    # Loop through each fold
    for fold, (train_indices, val_indices) in enumerate(kfold.split(windows, window_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{K_FOLDS}")
        print(f"{'='*50}")
        
        # Split samples into train and validation
        fold_train_windows = windows[train_indices]
        fold_train_labels = window_labels[train_indices]
        fold_val_windows = windows[val_indices]
        fold_val_labels = window_labels[val_indices]
        
        print(f"Train windows: {len(fold_train_windows)}, Val windows: {len(fold_val_windows)}")
        
        # Create datasets
        train_dataset = SkeletonDataset(fold_train_windows, fold_train_labels)
        val_dataset = SkeletonDataset(fold_val_windows, fold_val_labels)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Create CNN
        net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=DROP_PROB).to(device)
    
        ##Loss Function & Optimizer
        #CrossEntropyLoss: Combines LogSoftmax + NLLLoss.
        #SGD: Stochastic Gradient Descent with momentum. - would Adam be better? see Joe's example
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
        
        best_fold_acc = 0.0
        fold_best_model_file = os.path.join(fold_models_dir, f"fold_{fold}_best.pth")
        
        # Training loop
        for epoch in range(EPOCHS):
            net.train()
            
            running_loss = 0.0
            num_batches = 0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs, _ = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = running_loss / max(num_batches, 1)

            # get training accuracy
            train_acc = evaluate_accuracy(net, train_loader, device)

            # validate and get validation accuracy
            fold_val_acc, fold_val_loss = validate(net, val_loader, criterion, device)
            scheduler.step(fold_val_loss)

            print(
                f"Fold: {fold + 1}  Epoch: {epoch + 1}/{EPOCHS}  "
                f"Train Accuracy: {train_acc:.2f}%  "
                f"Validation Accuracy: {fold_val_acc:.2f}%  "
                f"Train Loss: {avg_train_loss:.4f}  "
                f"Validation Loss: {fold_val_loss:.4f}  "
            )
            if fold_val_acc > best_fold_acc:
                best_fold_acc = fold_val_acc
                torch.save(net.state_dict(), fold_best_model_file)

        fold_results.append(best_fold_acc)
        print(f"Fold {fold + 1} best validation accuracy: {best_fold_acc:.2f}%")
        
        # Print confusion matrix
        print(f"\nConfusion Matrix for Fold {fold + 1}:")
        print_confusion_matrix(net, val_loader, device, num_classes, class_names=class_names)
    
    # Print summary
    print(f"\n{'='*50}")
    print("K-Fold Cross-Validation Summary")
    print(f"{'='*50}")
    for fold, acc in enumerate(fold_results):
        print(f"Fold {fold + 1}: {acc:.2f}%")
    print(f"Mean Accuracy: {sum(fold_results) / len(fold_results):.2f}%")
    print(f"Best Accuracy: {max(fold_results):.2f}%")
    
    print(f"\n{'='*50}")
    print("Final Model Evaluation on Test Set:")
    print(f"{'='*50}")
    
    # load and window separate test data
    test_dir = "./data_val"
    test_x, test_y, _ = combine_recordings(test_dir, trim_front=499, people_map=people)
    test_x = normalize_skeleton(test_x)
    test_windows, test_window_labels = window_sequence(test_x, test_y, window_size=WINDOW_SIZE, stride=STRIDE)
    
    test_dataset = SkeletonDataset(test_windows, test_window_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    #get test accuracy
    for fold in range(K_FOLDS):
        # Create CNN
        model = os.path.join(fold_models_dir, f"fold_{fold}_best.pth");
        net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=DROP_PROB).to(device)
        net.load_state_dict(torch.load(model, map_location=device))
        
        # evaluate accuracy
        test_acc, test_loss = validate(net, test_loader, criterion, device)
        print(f"\nTest Accuracy on data_val for fold {fold}: {test_acc:.2f}%")
        print(f"Final Test Loss on data_val for fold {fold}: {test_loss:.4f}")
        
        # Print confusion matrix
        print(f"\nConfusion Matrix on Test Set (data_val) for fold {fold}:")
        print_confusion_matrix(net, test_loader, device, num_classes, class_names=class_names)

## majority of these validation functions also taken from PyTorch example
def evaluate_accuracy(net, loader, device):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / max(total, 1)


def validate(net, val_loader, criterion, device):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0
    
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # calculate outputs by running validation data through the network
            outputs, _ = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            num_batches += 1
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / max(total, 1)
    avg_loss = running_loss / max(num_batches, 1)
    return accuracy, avg_loss

#
# copilot wrote this function on its own, i have no idea how it works, but it's only used for debugging and bias testing
#
def print_confusion_matrix(net, val_loader, device, num_classes, class_names=None):
    net.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = net(inputs)
            _, predicted = torch.max(outputs, 1)

            labels_flat = labels.view(-1).long()
            predicted_flat = predicted.view(-1).long()
            flat_indices = labels_flat * num_classes + predicted_flat
            conf += torch.bincount(flat_indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    print("Confusion matrix (rows=true, cols=pred):")
    header = ["true/pred"] + class_names
    rows = []
    conf_np = conf.cpu().numpy()
    for idx, row in enumerate(conf_np):
        rows.append([class_names[idx]] + row.tolist())

    # print results as a table
    col_widths = [max(len(str(x)) for x in col) for col in zip(*([header] + rows))]
    def fmt_row(r):
        return " | ".join(str(x).ljust(w) for x, w in zip(r, col_widths))

    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))
          
            
if __name__ == "__main__":
    main()