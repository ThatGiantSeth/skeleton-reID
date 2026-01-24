import torch
import numpy as np
from CNN import CNNet

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device: {device}")

model = CNNet(window_size=10, num_joints=15, num_class=50, drop_prob=0.5).to(device)

