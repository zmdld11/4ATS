# main.py (更新版本 - 中文输出)
import os
import torch
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from src.audio_preprocessor import AudioDataPreprocessor
from src.model_builder import create_improved_classifier
from src.advanced_models import create_advanced_classifier, create_simplified_classifier
from src.model_trainer import AdvancedModelTrainer
from src.model_manager import model_manager
from src.utils import download_dataset, plot_training_history, analyze_model_performance

def setup_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎉 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("❌ 使用CPU进行训练")
    return device

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI音频分析与自动扒谱系统')
    parser.add_argument('--no-cache', action='store_true', 
                       help='不使用缓存，重新预处理数据')
    parser.add_argument('--resume', type=str, default=None,
                       help='从指定模型继续训练')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                       help='训练轮数')
    parser.add_argument('--no-resume', action='store_true',
                       help='强制从头开始训练，忽略现有模型')
    parser.add_argument('--basic-model', action='store_true',
                       help='使用基础模型而不是高级模型')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='禁用数据增强')
    parser.add_argument('--model-type', type=str, default='simplified', 
                       choices=['basic', 'advanced', 'simplified', 'all'],
                       help='使用的模型类型 (basic, advanced, simplified, all)')
    parser.add_argument('--train-all', action='store_true',
                       help='训练所有模型类型')
    return parser.parse_args()

def create_model(model_type, input_shape, num_classes, device):
    """根据类型创建模型并移动到设备"""
    if model_type == 'basic':
        model = create_improved_classifier(input_shape, num_classes)
    elif model_type == 'advanced':
        try:
            model = create_advanced_classifier(input_shape, num_classes)
        except Exception as e:
            print(f"高级模型创建失败: {e}")
            print("回退到稳定高级模型")
            model = create_stable_advanced_classifier(input_shape, num_classes)
    elif model_type == 'simplified':
        model = create_simplified_classifier(input_shape, num_classes)
    elif model_type == 'stable_advanced':
        model = create_stable_advanced_classifier(input_shape, num_classes)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 立即移动到设备
    model = model.to(device)
    print(f"✅ {model_type}模型已创建并移动到{device}")
    
    return model


def train_single_model(model_type, train_loader, val_loader, test_loader, preprocessor, device, args):
    """训练单个模型"""
    print(f"\n{'='*50}")
    print(f"训练 {model_type} 模型")
    print(f"{'='*50}")
    
    # 获取输入形状
    for data, _ in train_loader:
        input_shape = data.shape[1:]  # (1, 128, 130)
        break
    
    print(f"输入形状: {input_shape}")
    print(f"使用设备: {device}")
    
    # 创建模型并立即移动到设备
    model = create_model(model_type, input_shape, len(preprocessor.label_encoder.classes_), device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 检查模型是否在正确的设备上
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 初始化训练器
    trainer = AdvancedModelTrainer(model, preprocessor, device, model_type=model_type)
    
    # 检查是否需要恢复训练
    resume_model = not args.no_resume
    if resume_model:
        load_success, checkpoint = model_manager.load_model(model, model_type, device)
        if load_success and checkpoint and 'history' in checkpoint:
            trainer.history = checkpoint['history']
            print(f"恢复训练历史，已训练 {len(trainer.history['train_loss'])} 个周期")
            print(f"✅ {model_type}模型加载成功，继续训练...")
        else:
            print(f"✅ 从头开始训练{model_type}模型...")
    else:
        print(f"✅ 强制从头开始训练{model_type}模型...")
    
    # 训练模型
    history = trainer.train(train_loader, val_loader, epochs=args.epochs, 
                           patience=Config.EARLY_STOPPING_PATIENCE)
    
    # 评估模型
    test_loss, test_acc = trainer.evaluate(test_loader)
    
    # 保存最终模型
    trainer.save_model()
    
    return test_acc, history

def setup_device():
    """设置训练设备 - 增加内存监控"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎉 使用GPU: {torch.cuda.get_device_name(0)}")
        # 打印GPU内存信息
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("❌ 使用CPU进行训练")
    return device

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=== AI音频分析与自动扒谱系统 - 多版本模型 ===")
    print(f"使用缓存: {not args.no_cache}")
    print(f"模型类型: {args.model_type}")
    print(f"使用数据增强: {not args.no_augmentation}")
    print(f"同时保存多个版本: {Config.SAVE_MULTIPLE_VERSIONS}")
    
    # 显示可用模型
    available_models = model_manager.get_available_models()
    if available_models:
        print(f"已存在的模型: {', '.join(available_models)}")
    else:
        print("没有找到已训练的模型")
    
    # 1. 初始化配置
    print("\n1. 初始化配置...")
    Config.create_directories()
    
    # 2. 设置设备
    print("2. 检查硬件配置...")
    device = setup_device()
    print(f"PyTorch版本: {torch.__version__}")
    
    # 3. 数据预处理
    print("3. 数据预处理...")
    preprocessor = AudioDataPreprocessor(use_cache=not args.no_cache)
    
    # 创建数据加载器，可选数据增强
    use_augmentation = not args.no_augmentation and Config.USE_DATA_AUGMENTATION
    train_loader, val_loader, test_loader, num_classes = preprocessor.create_data_loaders(
        use_cache=not args.no_cache, 
        augment=use_augmentation
    )
    
    print(f"类别数量: {num_classes}")
    
    # 确定要训练的模型类型
    if args.model_type == 'all' or args.train_all:
        model_types = ['basic', 'simplified', 'advanced']
    else:
        model_types = [args.model_type]
    
    # 训练结果统计
    results = {}
    
    # 4. 训练模型
    for model_type in model_types:
        try:
            test_acc, history = train_single_model(
                model_type, train_loader, val_loader, test_loader, 
                preprocessor, device, args
            )
            results[model_type] = {
                'test_accuracy': test_acc,
                'history': history
            }
            
            # 可视化结果
            print(f"8. 生成{model_type}模型可视化结果...")
            plot_training_history(
                history['train_loss'], history['val_loss'],
                history['train_acc'], history['val_acc'],
                os.path.join(Config.OUTPUT_DIR, f'training_curves_{model_type}.png'),
                show_plot=False  # 添加这个参数
            )
            
            # 性能分析（只在最后一个模型上执行，避免重复）
            if model_type == model_types[-1]:
                print("9. 性能分析...")
                # 重新加载最佳模型进行评估
                best_model = create_model(model_type, train_loader.dataset[0][0].shape, num_classes)
                best_model = best_model.to(device)
                load_success, _ = model_manager.load_model(best_model, model_type, device)
                
                if load_success:
                    analyze_model_performance(
                        best_model, test_loader, preprocessor.label_encoder, device, Config.OUTPUT_DIR
                    )
            
        except Exception as e:
            print(f"❌ 训练{model_type}模型时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. 输出训练总结
    print("\n" + "="*60)
    print("训练总结")
    print("="*60)
    
    for model_type, result in results.items():
        print(f"{model_type:>10}模型: 测试准确率 = {result['test_accuracy']:.4f}")
    
    print(f"\n模型保存位置: {Config.MODEL_DIR}")
    print(f"输出文件位置: {Config.OUTPUT_DIR}")
    
    # 显示最终可用的模型
    final_models = model_manager.get_available_models()
    print(f"最终可用模型: {', '.join(final_models)}")

if __name__ == "__main__":
    main()