# model_comparison.py (新文件)
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from config.config import Config
from src.model_manager import model_manager
from src.audio_preprocessor import AudioDataPreprocessor
from src.model_builder import create_improved_classifier
from src.advanced_models import create_advanced_classifier, create_simplified_classifier

def compare_models():
    """比较所有可用模型的性能"""
    print("=== 模型性能比较 ===")
    
    # 获取可用模型
    available_models = model_manager.get_available_models()
    if not available_models:
        print("没有找到可用的模型")
        return
    
    print(f"找到的模型: {', '.join(available_models)}")
    
    # 加载数据
    preprocessor = AudioDataPreprocessor(use_cache=True)
    train_loader, val_loader, test_loader, num_classes = preprocessor.create_data_loaders(
        use_cache=True, augment=False
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for model_type in available_models:
        print(f"\n评估 {model_type} 模型...")
        
        try:
            # 创建模型
            if model_type == 'basic':
                model = create_improved_classifier((1, 128, 130), num_classes)
            elif model_type == 'advanced':
                model = create_advanced_classifier((1, 128, 130), num_classes)
            elif model_type == 'simplified':
                model = create_simplified_classifier((1, 128, 130), num_classes)
            else:
                continue
                
            model = model.to(device)
            
            # 加载权重
            load_success, _ = model_manager.load_model(model, model_type, device)
            if not load_success:
                continue
            
            # 评估模型
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            results[model_type] = accuracy
            print(f"{model_type}模型准确率: {accuracy:.4f}")
            
        except Exception as e:
            print(f"评估{model_type}模型失败: {e}")
    
    # 绘制比较图
    if results:
        plt.figure(figsize=(10, 6))
        models = list(results.keys())
        accuracies = [results[model] for model in models]
        
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Performance Comparison', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.0)
        
        # 在柱子上添加数值
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        comparison_path = os.path.join(Config.OUTPUT_DIR, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\n比较图已保存到: {comparison_path}")
        plt.show()
        
        # 输出最佳模型
        best_model = max(results, key=results.get)
        print(f"\n🎉 最佳模型: {best_model} (准确率: {results[best_model]:.4f})")

if __name__ == "__main__":
    compare_models()