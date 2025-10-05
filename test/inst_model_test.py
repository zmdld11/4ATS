import os
import sys
import torch
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import Config
from src.model_builder import create_improved_classifier
from src.advanced_models import create_advanced_classifier, create_simplified_classifier
from src.instrument_mapper import InstrumentMapper

class MultiModelTester:
    """多模型测试类"""
    
    def __init__(self, model_type='simplified'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        print(f"使用设备: {self.device}")
        print(f"测试模型: {model_type}")
        
        # 加载标签编码器
        label_encoder_path = os.path.join(project_root, "model", f"model_{model_type}_label_encoder.pkl")
        if not os.path.exists(label_encoder_path):
            print(f"错误: 标签编码器文件不存在 - {label_encoder_path}")
            return
        
        self.label_encoder = joblib.load(label_encoder_path)
        print(f"标签编码器加载成功，类别: {list(self.label_encoder.classes_)}")
        
        # 加载模型
        self.model = self.load_model(model_type)
        if self.model:
            self.model.eval()  # 设置为评估模式
            print(f"{model_type}模型加载成功!")
    
    def load_model(self, model_type):
        """加载指定类型的模型"""
        model_path = os.path.join(project_root, "model", f"model_{model_type}.pth")
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在 - {model_path}")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 获取模型参数
            input_shape = checkpoint.get('input_shape', (1, 128, 130))
            num_classes = checkpoint.get('num_classes', len(self.label_encoder.classes_))
            
            print(f"模型参数 - 输入形状: {input_shape}, 类别数: {num_classes}")
            
            # 根据模型类型创建对应的模型实例
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
            model.to(self.device)
            
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    
    def preprocess_audio(self, audio_path):
        """预处理音频文件"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=Config.TARGET_SAMPLE_RATE)
            
            # 确保音频长度一致
            y = librosa.util.fix_length(y, size=Config.TARGET_SAMPLE_RATE * Config.AUDIO_DURATION)
            
            # 提取Mel频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=Config.N_MELS, fmax=8000, 
                n_fft=2048, hop_length=512
            )
            log_mel = librosa.power_to_db(mel_spec)
            
            # 标准化
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            
            return log_mel
            
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def predict_single_audio(self, audio_path):
        """预测单个音频文件的乐器种类"""
        if self.model is None:
            print("模型未正确加载")
            return None, None
            
        print(f"\n分析音频文件: {audio_path}")
        
        # 预处理音频
        features = self.preprocess_audio(audio_path)
        if features is None:
            return None, None
        
        # 转换为模型输入格式
        input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 130)
        input_tensor = input_tensor.to(self.device)
        print(f"输入张量形状: {input_tensor.shape}")
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=3)  # 获取前3个预测
        
        # 转换为numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # 解码预测结果
        predictions = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            instrument = self.label_encoder.inverse_transform([idx])[0]
            predictions.append({
                'rank': i+1,
                'instrument': instrument,
                'probability': prob
            })
        
        return predictions, features
    
    def visualize_prediction(self, predictions, features, audio_name):
        """可视化预测结果"""
        # 乐器名称映射字典
        instrument_names = {
            'cel': 'Cello',
            'cla': 'Clarinet', 
            'flu': 'Flute',
            'gac': 'Acoustic Guitar',
            'gel': 'Electric Guitar',
            'org': 'Organ',
            'pia': 'Piano',
            'sax': 'Saxophone',
            'tru': 'Trumpet',
            'vio': 'Violin',
            'voi': 'Voice'
        }
        
        plt.figure(figsize=(12, 5))
        
        # 绘制Mel频谱图
        plt.subplot(1, 2, 1)
        librosa.display.specshow(features, sr=Config.TARGET_SAMPLE_RATE, 
                                hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - {audio_name}\n({self.model_type}模型)')
        
        # 绘制预测概率
        plt.subplot(1, 2, 2)
        # 将缩写转换为英文名称
        instruments = [instrument_names.get(p['instrument'], p['instrument']) for p in predictions]
        probabilities = [p['probability'] for p in predictions]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(predictions)))
        bars = plt.bar(instruments, probabilities, color=colors)
        
        # 在柱状图上添加数值标签
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.title(f'{self.model_type}模型预测结果')
        plt.xlabel('Instrument')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存结果
        output_path = os.path.join(Config.OUTPUT_DIR, f"prediction_{audio_name}_{self.model_type}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存: {output_path}")
        
        plt.show()

def test_all_models(audio_path):
    """测试所有三个模型"""
    model_types = ['basic', 'simplified', 'advanced']
    all_results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"测试 {model_type} 模型")
        print(f"{'='*50}")
        
        tester = MultiModelTester(model_type)
        if tester.model is None:
            print(f"跳过 {model_type} 模型（加载失败）")
            continue
            
        predictions, features = tester.predict_single_audio(audio_path)
        
        if predictions:
            print(f"\n=== {model_type}模型预测结果 ===")
            for pred in predictions:
                instrument_name = InstrumentMapper.get_english_name(pred['instrument'])
                print(f"{pred['rank']}. {instrument_name}: {pred['probability']:.3f}")
            
            # 可视化结果
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            tester.visualize_prediction(predictions, features, audio_name)
            
            # 存储结果
            all_results[model_type] = {
                'top_prediction': predictions[0],
                'all_predictions': predictions
            }
    
    return all_results

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='多模型测试')
    parser.add_argument('--model-type', type=str, default='all', 
                       choices=['basic', 'simplified', 'advanced', 'all'],
                       help='要测试的模型类型')
    parser.add_argument('--audio-path', type=str, 
                       default=os.path.join(project_root, "music", "3.flac"),
                       help='测试音频路径')
    
    args = parser.parse_args()
    
    print("=== AI音频分析与自动扒谱系统 - 多模型测试 ===")
    
    # 1. 初始化配置
    Config.create_directories()
    
    # 2. 检查音频文件是否存在
    if not os.path.exists(args.audio_path):
        print(f"错误: 测试音频文件不存在 - {args.audio_path}")
        print("请确保音频文件存在")
        return
    
    if args.model_type == 'all':
        # 测试所有模型
        results = test_all_models(args.audio_path)
        
        # 输出比较结果
        print(f"\n{'='*60}")
        print("模型比较结果")
        print(f"{'='*60}")
        
        for model_type, result in results.items():
            top_pred = result['top_prediction']
            instrument_name = InstrumentMapper.get_english_name(top_pred['instrument'])
            print(f"{model_type:>10}模型: {instrument_name:15s} (置信度: {top_pred['probability']:.3f})")
    
    else:
        # 测试单个模型
        tester = MultiModelTester(args.model_type)
        if tester.model is None:
            return
            
        predictions, features = tester.predict_single_audio(args.audio_path)
        
        if predictions:
            print(f"\n=== {args.model_type}模型预测结果 ===")
            for pred in predictions:
                instrument_name = InstrumentMapper.get_english_name(pred['instrument'])
                print(f"{pred['rank']}. {instrument_name}: {pred['probability']:.3f}")
            
            # 可视化结果
            audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
            tester.visualize_prediction(predictions, features, audio_name)
            
            # 输出最可能的乐器
            top_prediction = predictions[0]
            instrument_name = InstrumentMapper.get_english_name(top_prediction['instrument'])
            print(f"\n🎵 最可能的乐器: {instrument_name} (置信度: {top_prediction['probability']:.3f})")

if __name__ == "__main__":
    main()