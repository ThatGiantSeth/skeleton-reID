import torch
from CNN import CNNet
import torch.nn as nn
import argparse
import json
import os

from preprocessing import combine_recordings, normalize_skeleton, window_sequence

## copied from train_k_v2

WINDOW_SIZE = 10
STRIDE = 2
DROP_PROB = 0.4

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


def validate(net, val_loader, criterion, device):
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
    
    accuracy = 100 * correct / max(total, 1)
    print(f'\nValidation accuracy: {accuracy:.2f}%')
    return accuracy

## copilot function
def print_confusion_matrix(net, val_loader, device, class_names):
    net.eval()
    num_classes = len(class_names)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = net(inputs)
            preds = outputs.argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf[t.long(), p.long()] += 1

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
    parser.add_argument('--data', type=str, default='./data_val',
                        help='Path to the test data folder (default: ./data_val)')
    parser.add_argument('--people-map', type=str, default='./people_map.json',
                        help='Path to people_map JSON from training (default: ./people_map.json)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    
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
    
    # load people mapping if available (best for consistent class indexing)
    people = None
    if os.path.exists(args.people_map):
        with open(args.people_map, 'r') as f:
            people = json.load(f)
        print(f"Loaded people map from {args.people_map}")
    else:
        print(f"Warning: people map not found at {args.people_map}; inferring classes from labels")

    # load and window test data using the same preprocessing as training
    test_x, test_y, _ = combine_recordings(args.data, trim_front=499, people_map=people)
    test_x = normalize_skeleton(test_x)
    test_windows, test_window_labels = window_sequence(
        test_x,
        test_y,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        require_single_label=True
    )
    print(f"Loaded {len(test_windows)} test windows")

    if isinstance(people, dict) and len(people) > 0:
        num_classes = len(people)
        class_names = [f"class_{i}" for i in range(num_classes)]
        for name, idx in people.items():
            if isinstance(idx, int) and 0 <= idx < num_classes:
                class_names[idx] = name
    else:
        inferred_classes = sorted(set(test_window_labels))
        num_classes = len(inferred_classes)
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    # Create dataset and dataloader
    test_dataset = SkeletonDataset(test_windows, test_window_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Load model
    net = CNNet(in_channel=3, num_joints=15, window_size=WINDOW_SIZE, num_class=num_classes, drop_prob=DROP_PROB).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")
    
    # Loss function for reference
    criterion = nn.CrossEntropyLoss()
    
    # Compute test accuracy
    test_acc = validate(net, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Print confusion matrix
    print("\nConfusion Matrix on Test Set:")
    print_confusion_matrix(net, test_loader, device, class_names)


if __name__ == "__main__":
    main()
