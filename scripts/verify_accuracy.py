import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from model import DenseNet121, DataPreprocessing
import os
import sys

def progress_bar(iterable, desc="Progress"):
    try:
        total = len(iterable)
    except:
        total = None
        
    for i, item in enumerate(iterable):
        # Yield the item immediately
        yield item
        
        # Update progress *after* yielding
        if total:
            percent = (i + 1) / total
            bar_len = 30
            filled = int(bar_len * percent)
            bar = '=' * filled + '-' * (bar_len - filled)
            sys.stdout.write(f'\r{desc}: [{bar}] {i+1}/{total} ({percent:.1%})')
            sys.stdout.flush()
    if total:
        print()

def evaluate_model(model_path, dataset, device):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return 0.0

    print(f"Loading {model_path}...")
    model = DenseNet121()
    try:
        # The model class saves 'self.model.state_dict()', so we must load into 'model.model'
        # or use model.load_param() if we could, but here we do it manually.
        model.model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return 0.0
        
    model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar(loader, desc="Evaluating")):
            images = images.to(device)
            # data shape: (batch, C, H, W)
            
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            
            # Metrics
            preds = predicted.cpu()
            lbls = labels.cpu()
            
            for p, l in zip(preds, lbls):
                if l == 1 and p == 1:
                    tp += 1
                elif l == 0 and p == 0:
                    tn += 1
                elif l == 0 and p == 1:
                    fp += 1
                elif l == 1 and p == 0:
                    fn += 1
            
            correct += (preds == lbls).sum().item()
            total += labels.size(0)
            # Removed early stopping limit
            # if total >= 50:
            #     print("\nStopping early for quick verification (processed 50+ images).")
            #     break
    
    sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}%")
    print(f"Specificity: {specificity:.2f}%")
    
    return 100 * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check Original Model (tb_model.pt)
    print("Initializing DataPreprocessing...")
    full_data = DataPreprocessing()
    if len(full_data) == 0:
        return

    # print("Evaluating tb_model.pt (Original)...")
    # acc_orig = evaluate_model(os.path.join(os.path.dirname(__file__), "../models/tb_model.pt"), full_data, device)
    # print(f"RESULT: Accuracy (tb_model.pt) = {acc_orig:.2f}%")

    print("Evaluating tb_model_best.pt (New Best)...")
    acc_best = evaluate_model(os.path.join(os.path.dirname(__file__), "../models/tb_model_best.pt"), full_data, device)
    print(f"RESULT: Accuracy (tb_model_best.pt) = {acc_best:.2f}%")

if __name__ == "__main__":
    main()
