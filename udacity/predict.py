import argparse
import torch
from torchvision import models
from torch import nn
from torchvision import transforms
import json
from PIL import Image

# Define command line arguments
parser = argparse.ArgumentParser(description='Predict.py')
parser.add_argument('image_path', metavar='image_path', type=str, help='Path to the image file')
parser.add_argument('checkpoint', metavar='checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--top_k', dest='top_k', action='store', type=int, default=3, help='Return top K most likely classes')
parser.add_argument('--category_names', dest='category_names', action='store', default='cat_to_name.json', help='Path to the mapping of categories to real names')
parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

# Load the mapping of categories to real names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the checkpoint
checkpoint = torch.load(args.checkpoint)

# Load a pre-trained model
if checkpoint['arch'] == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif checkpoint['arch'] == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_size = 1024
else:
    print("Invalid model architecture in the checkpoint.")
    exit()

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Define classifier
classifier = nn.Sequential(nn.Linear(input_size, 512),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(512, 102),
                            nn.LogSoftmax(dim=1))

model.classifier = classifier
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

# Process the image
def process_image(image):
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = img_transforms(image)
    return image

# Make predictions
def predict(image_path, model, topk=5):
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0)
    image = image.float()
    
    if args.gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    else:
        model.cpu()
        image = image.cpu()
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        probs, indices = torch.topk(torch.exp(output), topk)
        probs = probs.cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]
        classes = [cat_to_name[str(model.class_to_idx[str(idx)])] for idx in indices]
    return probs, classes

# Perform prediction
probs, classes = predict(args.image_path, model, topk=args.top_k)

# Print the result
for i in range(len(probs)):
    print(f"{classes[i]}: {probs[i]*100:.2f}%")

