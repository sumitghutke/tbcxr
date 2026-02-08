import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_distribution(base_path, output_path):
    classes = ['Normal', 'Tuberculosis']
    counts = []
    
    for cls in classes:
        cls_path = os.path.join(base_path, cls)
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            counts.append(count)
        else:
            counts.append(0)
            print(f"Warning: Path not found {cls_path}")

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=classes, y=counts, palette='viridis')
    plt.title('Dataset Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    
    # Add text labels
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', va='bottom')
        
    plt.savefig(output_path)
    print(f"Created {output_path}")

if __name__ == "__main__":
    plot_data_distribution(
        '/home/sumit/Downloads/ChestXRayModel/data/images',
        '/home/sumit/Downloads/ChestXRayModel/outputimg/data_distribution.png'
    )
