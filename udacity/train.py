import argparse
import torch
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
import json

# Define command line arguments
parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
parser.add_argument('data_dir', metavar='data_dir', type=str, help='Path to the data directory')
parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth', help='Directory to save checkpoints')
parser.add_argument('--arch', dest='arch', action='store', default='vgg16', help='Model architecture (vgg16 or densenet121)')
parser.add_argument('--learning_rate', dest='learning_rate', action='store', default=0.003, type=float, help='Learning rate')
parser.add_argument('--hidden_units', dest='hidden_units', action='store', default=512, type=int, help='Number of hidden units')
parser.add_argument('--epochs', dest='epochs', action='store', default=20, type=int, help='Number of epochs')
parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

# Load data
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Load a pre-trained model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_size = 1024
else:
    print("Invalid model architecture. Please use 'vgg16' or 'densenet121'.")
    exit()

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Define classifier
classifier = nn.Sequential(nn.Linear(input_size, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

model.classifier = classifier

# Define criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Use GPU if available
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Train the classifier
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 40

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Loss: {running_loss/print_every:.3f}")
            
            running_loss = 0

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {
    'arch': args.arch,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
}

torch.save(checkpoint, args.save_dir)
print("Model trained and saved successfully!")
