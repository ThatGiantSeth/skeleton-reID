import torch
from CNN import CNNet
import torch.nn as nn
import argparse

from preprocessing import get_arrays, normalize_skeleton

## just copied from train_K.py so that i can test the model without running training again 

WINDOW_SIZE = 50
STRIDE = 2

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


def validate(net, val_loader, criterion, device, epoch=0):
    """Validate the model on a validation set."""
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
    print(f'\nValidation accuracy: {accuracy:.2f}%')
    return accuracy

## copilot function
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


def main():
    # add some arguments so i can change which fold model to load
    parser = argparse.ArgumentParser(description='Evaluate a trained skeleton model on test data')
    parser.add_argument('--model', type=str, default='skeleton_model_best_k.pth',
                        help='Path to the model file to evaluate (default: skeleton_model_best_k.pth)')
    parser.add_argument('--data', type=str, default='./data_val_split',
                        help='Path to the test data folder (default: ./data_val_split)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation (default: 16)')
    
    args = parser.parse_args()
    
    # use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n{'='*50}")
    print("Model Evaluation on Test Set")
    print(f"{'='*50}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}\n")
    
    # Load test data
    test_arrays, test_labels, _ = get_arrays(args.data, trim_front=0)
    print(f"Loaded {len(test_arrays)} test samples")
    
    num_classes = len(set(test_labels))
    
    # Create dataset and dataloader
    test_dataset = SkeletonDataset(test_arrays, test_labels, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load model
    net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=0.5).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")
    
    # Loss function for reference
    criterion = nn.CrossEntropyLoss()
    
    # Compute test accuracy
    test_acc = validate(net, test_loader, criterion, device, epoch=0)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Print confusion matrix
    print("\nConfusion Matrix on Test Set:")
    print_confusion_matrix(net, test_loader, device, num_classes)


if __name__ == "__main__":
    main()
