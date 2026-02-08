import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        # Do not download pretrained weights by default to avoid SSL/download issues.
        # If you want pretrained weights, set `weights` to a valid enum, e.g.
        # `torchvision.models.DenseNet121_Weights.DEFAULT` and ensure network access.
        self.model = torchvision.models.densenet121(weights=None)
        # CHANGED: Output layer reduced to 2 (Normal vs TB). 
        # Removed Sigmoid here because we use BCEWithLogitsLoss for numerical stability.
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    def save_param(self):
        torch.save(self.model.state_dict(), "tb_model.pt")

    def load_param(self):
        self.model.load_state_dict(torch.load("tb_model.pt"))

class DataPreprocessing(Dataset):
    # Use integer class labels for CrossEntropyLoss: 0=Normal, 1=Tuberculosis
    classEncoding = {
        'Normal': 0,
        'Tuberculosis': 1
    }

    def __init__(self, root_dir="data/images"):
        self.image_names = []
        self.labels = []
        
        categories = ['Normal', 'Tuberculosis']
        
        for category in categories:
            class_folder = os.path.join(root_dir, category)
            if not os.path.exists(class_folder):
                continue
                
            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(class_folder, file_name)
                    self.image_names.append(full_path)
                    # store as integer tensor (LongTensor) for CrossEntropyLoss
                    self.labels.append(torch.tensor(self.classEncoding[category], dtype=torch.long))

    def __getitem__(self, index):
        image_path = self.image_names[index]
        image = Image.open(image_path).convert('RGB')
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # 10-crop augmentation
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
        image = preprocess(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_names)


class Train():
    def __init__(self, trainset, model, device=torch.device('cpu')):
        # reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Reduced batch size slightly to be safe on smaller GPUs
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

        # Use CrossEntropyLoss with integer class indices
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        model.to(device)

        print("Starting training...")
        # epochs
        epochs = 10
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(trainloader, 0):

                optimizer.zero_grad()

                n_batches, n_crops, channels, height, width = images.size()
                image_batch = images.view(-1, channels, height, width).to(device)

                # Tile labels to match the 10 crops -> result is LongTensor of shape (batch*n_crops,)
                labels = tile(labels, 0, n_crops).to(device)

                outputs = model(image_batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Print every 10 batches
                if i % 10 == 9:
                    print('[Epoch: %d, Batch: %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

            scheduler.step()

        print('Finished Training')
        model.save_param()
        
class Test():
    def __init__(self, testset, model, device=torch.device('cpu')):
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
        num_classes = 2
        conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
        correct = 0
        total = 0

        model.to(device)
        model.eval()

        print("Starting testing...")
        with torch.no_grad():
            for (images, labels) in testloader:
                # images: (batch=1, n_crops, C, H, W)
                n_batches, n_crops, channels, height, width = images.size()
                image_batch = images.view(-1, channels, height, width).to(device)

                outputs = model(image_batch)  # (n_crops, num_classes)

                # aggregate across crops per image by averaging logits
                outputs = outputs.view(n_crops, -1)
                avg_output = outputs.mean(dim=0, keepdim=True)  # (1, num_classes)
                predicted = torch.argmax(avg_output, dim=1).item()

                truth = labels.item() if isinstance(labels, torch.Tensor) else int(labels)

                conf_matrix[truth, predicted] += 1
                total += 1
                correct += int(predicted == truth)

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total if total > 0 else 0))
        print('Confusion matrix (rows=true, cols=pred):')
        print(conf_matrix)

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def get_label_for_image(model, image_path):
    classes = ['Normal', 'Tuberculosis']
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    with torch.no_grad():
        output = model(input_batch)
    
    index_tensor = torch.argmax(output)
    index = index_tensor.item()
    return classes[index]    

def main():
    # 1. Setup Data
    data = DataPreprocessing()
    if len(data) == 0:
        print("No images found! Please run 'organize_data.py' first.")
        return

    # 2. Split Data (80% Train, 20% Test)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_set, test_set = random_split(data, [train_size, test_size])
    
    # 3. Initialize Model
    model = DenseNet121()
    
    # 4. Train
    print(f"Training on {len(train_set)} images...")
    train = Train(train_set, model)
    
    # 5. Test
    print(f"Testing on {len(test_set)} images...")
    test = Test(test_set, model)

if __name__ == '__main__':
    main()