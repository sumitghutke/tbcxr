import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
from model import DenseNet121

# --- Configuration ---
TB_DIR = "/home/sumit/Documents/archive/TBX11K/imgs/tb"
NORMAL_DIR = "/home/sumit/Documents/archive/TBX11K/imgs/health"
MODEL_PATH = "tb_model_best.pt"

class CustomDataset(Dataset):
    def __init__(self, tb_dir, normal_dir, max_samples_per_class=100):
        self.image_paths = []
        self.labels = []
        import random
        
        # Load TB images (Label 1)
        tb_images = []
        if os.path.exists(tb_dir):
            for fname in os.listdir(tb_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('._'):
                    tb_images.append(os.path.join(tb_dir, fname))
            
            # Randomly sample
            if len(tb_images) > max_samples_per_class:
                print(f"Subsampling TB images from {len(tb_images)} to {max_samples_per_class}")
                tb_images = random.sample(tb_images, max_samples_per_class)
            
            for path in tb_images:
                self.image_paths.append(path)
                self.labels.append(1)
        else:
            print(f"Warning: TB directory not found: {tb_dir}")

        # Load Normal images (Label 0)
        normal_images = []
        if os.path.exists(normal_dir):
            for fname in os.listdir(normal_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('._'):
                    normal_images.append(os.path.join(normal_dir, fname))

            # Randomly sample
            if len(normal_images) > max_samples_per_class:
                print(f"Subsampling Normal images from {len(normal_images)} to {max_samples_per_class}")
                normal_images = random.sample(normal_images, max_samples_per_class)
                
            for path in normal_images:
                self.image_paths.append(path)
                self.labels.append(0)
        else:
            print(f"Warning: Normal directory not found: {normal_dir}")
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy image or handle error appropriately.
            # For simplicity, let's just return a black image
            image = Image.new('RGB', (224, 224))
            
        # Matches the transforms in model.py (DataPreprocessing) for validation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # CenterCrop is standard for validation/testing
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = transform(image)
        return image, label

def progress_bar(iterable, desc="Progress"):
    try:
        total = len(iterable)
    except:
        total = None
        
    for i, item in enumerate(iterable):
        yield item
        if total:
            # Simple progress print
            if i % 10 == 0 or i == total - 1:
                percent = (i + 1) / total * 100
                sys.stdout.write(f"\r{desc}: {i+1}/{total} ({percent:.1f}%)")
                sys.stdout.flush()
    print()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Dataset
    print(f"Scanning datasets...")
    print(f"TB: {TB_DIR}")
    print(f"Normal: {NORMAL_DIR}")
    
    dataset = CustomDataset(TB_DIR, NORMAL_DIR)
    if len(dataset) == 0:
        print("No images found! Check your paths.")
        return

    print(f"Found {len(dataset)} total images.")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found!")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = DenseNet121()
    try:
        model.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    model.to(device)
    model.eval()

    # 3. Evaluate
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels in progress_bar(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            # Metrics
            for p, l in zip(predicted, labels):
                l_item = l.item()
                p_item = p.item()
                
                if l_item == 1 and p_item == 1:
                    tp += 1
                elif l_item == 0 and p_item == 0:
                    tn += 1
                elif l_item == 0 and p_item == 1:
                    fp += 1 # False Positive: Normal predicted as TB
                elif l_item == 1 and p_item == 0:
                    fn += 1 # False Negative: TB predicted as Normal

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # 4. Report
    if total > 0:
        accuracy = 100 * correct / total
        sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0 # approximate F1 formula
        f1_score = 2 * tp / (2 * tp + fp + fn) # standard F1 formula

        print("\n" + "="*30)
        print("TEST RESULTS")
        print("="*30)
        print(f"Model: {MODEL_PATH}")
        print(f"Total Images: {total}")
        print(f"Correct: {correct}")
        print("-" * 20)
        print(f"True Positives (TB identified as TB): {tp}")
        print(f"True Negatives (Normal identified as Normal): {tn}")
        print(f"False Positives (Normal identified as TB): {fp}")
        print(f"False Negatives (TB identified as Normal): {fn}")
        print("-" * 20)
        print(f"Accuracy:    {accuracy:.2f}%")
        print(f"Sensitivity: {sensitivity:.2f}% (Recall)")
        print(f"Specificity: {specificity:.2f}%")
        print(f"Precision:   {precision:.2f}%")
        print(f"F1 Score:    {f1_score*100:.2f}%")
        print("="*30)
    else:
        print("Error: Total processed images is 0.")

if __name__ == "__main__":
    main()
