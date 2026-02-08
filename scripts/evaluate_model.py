import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
import model
import sys
import os
from PIL import Image

# Set seed for reproducibility
torch.manual_seed(42)

# Custom Validation Dataset to avoid training augmentations
class ValDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
        ])
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

def evaluate():
    print("Initializing evaluation...")
    
    # We need to extract the raw data from DataPreprocessing to split it manually
    # creating a wrapper or accessing internal lists
    
    raw_dataset = model.DataPreprocessing(root_dir=os.path.join(os.path.dirname(__file__), '../data/images'))
    
    # Access internal lists (assuming they are populated in __init__)
    # Based on previous view_code_item, they are self.image_names and self.labels
    
    full_size = len(raw_dataset.image_names)
    train_size = int(0.8 * full_size)
    val_size = full_size - train_size
    
    # Generate indices
    indices = list(range(full_size))
    # Shuffle with seed
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_indices = indices[train_size:]
    
    # specific validation data
    val_paths = [raw_dataset.image_names[i] for i in val_indices]
    val_labels = [raw_dataset.labels[i] for i in val_indices]
    
    val_set = ValDataset(val_paths, val_labels)
    
    print(f"Validation set size: {len(val_set)}")
    
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    
    # Load Model
    net = model.DenseNet121()
    checkpoint_path = os.path.join(os.path.dirname(__file__), '../models/tb_model_best.pt')
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        net.model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    net.model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running predictions...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            if i % 10 == 0:
                print(f"Batch {i}/{len(val_loader)}")
            
    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    
    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Tuberculosis'])
    print(report)
    
    output_metrics_path = os.path.join(os.path.dirname(__file__), '../docs/metrics.txt')
    with open(output_metrics_path, 'w') as f:
        f.write(report)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'TB'], 
                yticklabels=['Normal', 'TB'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    output_cm_path = os.path.join(os.path.dirname(__file__), '../docs/confusion_matrix.png')
    plt.savefig(output_cm_path)
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
