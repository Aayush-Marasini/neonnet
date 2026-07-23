import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import struct
import random
import numpy as np
import os

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print(f"random seed set to {seed}")

# Convert images to Tensors and scale pixels to [0.0 - 1.0]

transform = transforms.ToTensor()


# Get the dataset

train_dataset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle = True)

# Download and load the testing data

test_dataset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1000, shuffle = False)

print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} test images.")


# Neural Network Definition

class NeonNet(nn.Module):
    def __init__(self):

        super(NeonNet, self).__init__()

        self.flatten = nn.Flatten()

        self.layer1 = nn.Linear(784, 128)

        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

model = NeonNet()
print(model)


# Grader and the Updater setup 

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


# Training Loop

epochs = 9
print("starting training....")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Average Loss : {avg_loss: .4f}")

print("Training complete")

# Check Accuracy

print("Testing the model accuracy ...")

model.eval()

correct = 0 
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final test accuracy is {accuracy: .2f}")


# Exporting Weights to Binary

print("Exporting weights to C binary format...")

os.makedirs("artifacts", exist_ok=True)

W1 = model.layer1.weight.detach().numpy().T
b1 = model.layer1.bias.detach().numpy()
W2 = model.layer2.weight.detach().numpy().T
b2 = model.layer2.bias.detach().numpy()

with open("artifacts/weights.bin", "wb") as f:

    f.write(struct.pack('<I', 0x4E454F4E)) 
    f.write(struct.pack('<I', 4))


    for tensor in [W1, b1, W2, b2]:

        if len(tensor.shape)== 1:
            rows = 1
            cols = tensor.shape[0]
        else:
            rows = tensor.shape[0]
            cols = tensor.shape[1]
            
        f.write(struct.pack('<I', rows))
        f.write(struct.pack('<I', cols))
        
       
        f.write(tensor.astype(np.float32).tobytes())

print("Weights exported successfully to artifacts/weights.bin!")


# Export Test Data for C Engine Validation

print("Exporting test data for C verification ...")

images, labels = next(iter(test_loader))

num_samples = 700
images = images[:num_samples]
labels = labels[:num_samples]

with torch.no_grad():
    outputs = model(images)

with open("artifacts/test_samples.bin", "wb") as f:

    f.write(struct.pack('<I',0x44415441))
    f.write(struct.pack('<I', num_samples))


    for i in range(num_samples):

        f.write(struct.pack('<I', labels[i].item()))
        
        f.write(images[i].numpy().astype(np.float32).tobytes())
        
        f.write(outputs[i].numpy().astype(np.float32).tobytes())

print("Test data exported to artifacts/test_samples.bin!")
