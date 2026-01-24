import torch
import CNN as cnn
import time


window=10
joints=15
num_classes=50

def classifier_model():
    model = cnn.CNNet(window_size=window, num_joints=joints, num_class=num_classes, drop_prob=0.5)
    return model
    

model=classifier_model()


input = torch.randn(1, 3, window, joints)








start_time = time.process_time()
output = model(input)
end_time = time.process_time()

forward_time = end_time - start_time

print("Forward pass time:", forward_time)

