# utils.py (更新后的绘图函数)
import os
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import torch
from config.config import Config

def download_dataset():
    """Download dataset"""
    dataset_path = os.path.join(Config.DATA_DIR, "irmas.zip")
    extract_path = Config.DATA_DIR
    
    if not os.path.exists(os.path.join(extract_path, Config.DATASET_NAME)):
        print("Downloading dataset...")
        urllib.request.urlretrieve(Config.DATASET_URL, dataset_path)
        
        # Extract files
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print("Dataset download completed!")
    else:
        print("Dataset already exists!")

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None, show_plot=False):
    """Plot training history charts - 修改为不自动显示"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy chart
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss chart
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
        print(f"Training chart saved to: {save_path}")
    
    # 只在明确要求时显示
    if show_plot:
        plt.show()
    else:
        plt.close()  # 关闭图表，释放内存

def analyze_model_performance(model, test_loader, label_encoder, device, save_dir):
    """Analyze model performance"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=label_encoder.classes_))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    plt.show()
    
    # Calculate accuracy for each class
    print("\nClass-wise Accuracy:")
    class_accuracy = {}
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    for i, class_name in enumerate(label_encoder.classes_):
        mask = all_labels == i
        if np.sum(mask) > 0:
            acc = np.mean(all_preds[mask] == all_labels[mask])
            class_accuracy[class_name] = acc
            print(f"  {class_name}: {acc:.3f}")
    
    return class_accuracy