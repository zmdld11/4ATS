import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import Config
from src.model_builder import create_improved_classifier
from src.advanced_models import create_advanced_classifier, create_simplified_classifier
from src.audio_preprocessor import AudioDataPreprocessor
from src.instrument_mapper import InstrumentMapper

class MultiModelEvaluator:
    """多模型评估器"""
    
    def __init__(self, model, label_encoder, device, model_type):
        self.model = model
        self.label_encoder = label_encoder
        self.device = device
        self.model_type = model_type
        self.model.eval()
    
    def comprehensive_evaluation(self, test_loader):
        """全面评估模型"""
        print(f"\n=== {self.model_type}模型全面评估 ===")
        
        # 1. 基础指标
        accuracy, class_accuracy = self.calculate_accuracy(test_loader)
        
        # 2. 详细分类报告
        y_true, y_pred, y_prob = self.get_predictions(test_loader)
        self.print_classification_report(y_true, y_pred)
        
        # 3. 混淆矩阵
        self.plot_confusion_matrix(y_true, y_pred)
        
        # 4. ROC曲线和AUC
        self.plot_roc_curves(y_true, y_prob)
        
        # 5. 各类别性能分析
        self.analyze_class_performance(y_true, y_pred, y_prob)
        
        return {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def calculate_accuracy(self, test_loader):
        """计算准确率"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(len(self.label_encoder.classes_))}
        class_total = {i: 0 for i in range(len(self.label_encoder.classes_))}
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 各类别准确率
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        accuracy = 100 * correct / total
        class_accuracy = {}
        
        print(f"\n📊 {self.model_type}模型总体准确率: {accuracy:.2f}%")
        print(f"\n📈 各类别准确率:")
        print("-" * 50)
        
        for i in range(len(self.label_encoder.classes_)):
            instrument = self.label_encoder.inverse_transform([i])[0]
            english_name = InstrumentMapper.get_english_name(instrument)
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                class_accuracy[english_name] = acc
                print(f"  {english_name:15s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                class_accuracy[english_name] = 0
                print(f"  {english_name:15s}:   0.00% (0/0)")
        
        return accuracy, class_accuracy
    
    def get_predictions(self, test_loader):
        """获取预测结果"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def print_classification_report(self, y_true, y_pred):
        """打印详细分类报告"""
        target_names = [InstrumentMapper.get_english_name(instr) 
                       for instr in self.label_encoder.classes_]
        
        print(f"\n📋 {self.model_type}模型详细分类报告:")
        print("=" * 70)
        report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        print(report)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        target_names = [InstrumentMapper.get_english_name(instr) 
                       for instr in self.label_encoder.classes_]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'{self.model_type}模型 - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = os.path.join(Config.OUTPUT_DIR, f'confusion_matrix_{self.model_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ {self.model_type}模型混淆矩阵已保存: {output_path}")
        plt.show()
    
    def plot_roc_curves(self, y_true, y_prob):
        """绘制ROC曲线"""
        # 将标签二值化
        y_true_bin = label_binarize(y_true, classes=range(len(self.label_encoder.classes_)))
        
        # 计算每个类别的ROC曲线和AUC
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        plt.figure(figsize=(10, 8))
        
        for i in range(len(self.label_encoder.classes_)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            instrument = InstrumentMapper.get_english_name(
                self.label_encoder.inverse_transform([i])[0]
            )
            plt.plot(fpr[i], tpr[i], label=f'{instrument} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_type}模型 - ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(Config.OUTPUT_DIR, f'roc_curves_{self.model_type}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ {self.model_type}模型ROC曲线已保存: {output_path}")
        plt.show()
        
        # 显示AUC统计
        print(f"\n🎯 {self.model_type}模型AUC统计:")
        print("-" * 40)
        avg_auc = np.mean(list(roc_auc.values()))
        print(f"平均AUC: {avg_auc:.4f}")
        for i, auc_val in roc_auc.items():
            instrument = InstrumentMapper.get_english_name(
                self.label_encoder.inverse_transform([i])[0]
            )
            print(f"  {instrument:15s}: {auc_val:.4f}")
    
    def analyze_class_performance(self, y_true, y_pred, y_prob):
        """分析各类别性能"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        print(f"\n🎼 {self.model_type}模型各类别详细性能指标:")
        print("=" * 70)
        print(f"{'Instrument':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i in range(len(self.label_encoder.classes_)):
            instrument = InstrumentMapper.get_english_name(
                self.label_encoder.inverse_transform([i])[0]
            )
            support = np.sum(y_true == i)
            
            print(f"{instrument:<20} {precision_per_class[i]:<10.4f} "
                  f"{recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f} {support:<10}")
        
        # 宏观平均和加权平均
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        weighted_precision = precision_score(y_true, y_pred, average='weighted')
        weighted_recall = recall_score(y_true, y_pred, average='weighted')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("-" * 70)
        print(f"{'Macro Avg':<20} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
        print(f"{'Weighted Avg':<20} {weighted_precision:<10.4f} {weighted_recall:<10.4f} {weighted_f1:<10.4f}")

def load_model_and_evaluator(model_type, device):
    """加载模型和评估器"""
    import joblib
    
    # 加载标签编码器
    label_encoder_path = os.path.join(project_root, "model", f"model_{model_type}_label_encoder.pkl")
    if not os.path.exists(label_encoder_path):
        print(f"错误: 标签编码器文件不存在 - {label_encoder_path}")
        return None
    
    label_encoder = joblib.load(label_encoder_path)
    
    # 加载模型
    model_path = os.path.join(project_root, "model", f"model_{model_type}.pth")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 根据模型类型创建对应的模型
    input_shape = checkpoint.get('input_shape', (1, 128, 130))
    num_classes = checkpoint.get('num_classes', len(label_encoder.classes_))
    
    if model_type == 'basic':
        model = create_improved_classifier(input_shape, num_classes)
    elif model_type == 'advanced':
        model = create_advanced_classifier(input_shape, num_classes)
    elif model_type == 'simplified':
        model = create_simplified_classifier(input_shape, num_classes)
    else:
        print(f"未知的模型类型: {model_type}")
        return None
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ {model_type}模型加载成功")
    
    return MultiModelEvaluator(model, label_encoder, device, model_type)

def compare_all_models(test_loader, device):
    """比较所有模型"""
    model_types = ['basic', 'simplified', 'advanced']
    results = {}
    
    for model_type in model_types:
        evaluator = load_model_and_evaluator(model_type, device)
        if evaluator is None:
            continue
            
        result = evaluator.comprehensive_evaluation(test_loader)
        results[model_type] = result
    
    # 绘制比较图
    if len(results) > 1:
        plt.figure(figsize=(12, 6))
        
        # 准确率比较
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('模型准确率比较', fontsize=14)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # 在柱子上添加数值
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{accuracy:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # 各类别准确率比较
        plt.subplot(1, 2, 2)
        instruments = list(results[models[0]]['class_accuracy'].keys())
        x = np.arange(len(instruments))
        width = 0.25
        
        for i, model_type in enumerate(models):
            accuracies = [results[model_type]['class_accuracy'][inst] for inst in instruments]
            plt.bar(x + i*width, accuracies, width, label=model_type)
        
        plt.xlabel('乐器类别', fontsize=12)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.title('各类别准确率比较', fontsize=14)
        plt.xticks(x + width, instruments, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(Config.OUTPUT_DIR, 'model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 模型比较图已保存: {output_path}")
        plt.show()
        
        # 输出最佳模型
        best_model = max(results, key=lambda x: results[x]['accuracy'])
        print(f"\n🎉 最佳模型: {best_model} (准确率: {results[best_model]['accuracy']:.2f}%)")

def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description='多模型评估')
    parser.add_argument('--model-type', type=str, default='all', 
                       choices=['basic', 'simplified', 'advanced', 'all'],
                       help='要评估的模型类型')
    
    args = parser.parse_args()
    
    print("=== 多模型训练程度评估 ===")
    
    # 1. 初始化配置
    Config.create_directories()
    
    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 3. 加载测试数据
    print("\n加载测试数据...")
    preprocessor = AudioDataPreprocessor(use_cache=True)
    _, _, test_loader, num_classes = preprocessor.create_data_loaders(use_cache=True, augment=False)
    
    if args.model_type == 'all':
        # 比较所有模型
        compare_all_models(test_loader, device)
    else:
        # 评估单个模型
        evaluator = load_model_and_evaluator(args.model_type, device)
        if evaluator is None:
            return
            
        results = evaluator.comprehensive_evaluation(test_loader)
        
        # 模型训练程度总结
        print(f"\n🎯 {args.model_type}模型训练程度总结:")
        print("=" * 50)
        accuracy = results['accuracy']
        
        if accuracy > 90:
            print("✅ 优秀! 模型训练得很好")
        elif accuracy > 80:
            print("✅ 良好! 模型训练得不错")  
        elif accuracy > 70:
            print("⚠️ 一般! 模型需要进一步优化")
        else:
            print("❌ 较差! 建议重新训练或调整模型")
        
        print(f"总体准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()