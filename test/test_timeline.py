import os
import sys
import torch
import joblib
import librosa
import numpy as np
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import Config
from src.model_builder import create_improved_classifier
from src.advanced_models import create_advanced_classifier, create_simplified_classifier
from src.time_series_analyzer import TimeSeriesAnalyzer

def load_model_with_fix(model_type, device):
    """加载指定类型的模型并修复输入形状问题"""
    # 加载标签编码器
    label_encoder_path = os.path.join(project_root, "model", f"model_{model_type}_label_encoder.pkl")
    if not os.path.exists(label_encoder_path):
        print(f"错误: 标签编码器文件不存在 - {label_encoder_path}")
        return None, None
    
    label_encoder = joblib.load(label_encoder_path)
    
    # 加载模型
    model_path = os.path.join(project_root, "model", f"model_{model_type}.pth")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 修复输入形状
    input_shape = (1, 128, 130)  # (channels, height, width)
    num_classes = len(label_encoder.classes_)
    
    print(f"使用输入形状: {input_shape}")
    print(f"类别数量: {num_classes}")
    
    # 根据模型类型创建对应的模型
    if model_type == 'basic':
        model = create_improved_classifier(input_shape, num_classes)
    elif model_type == 'advanced':
        model = create_advanced_classifier(input_shape, num_classes)
    elif model_type == 'simplified':
        model = create_simplified_classifier(input_shape, num_classes)
    else:
        print(f"未知的模型类型: {model_type}")
        return None, None
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, label_encoder

def ensure_feature_shape(features, target_shape=(128, 130)):
    """确保特征形状一致"""
    if features.shape != target_shape:
        # 调整到目标形状
        if features.shape[1] < target_shape[1]:
            # 填充
            pad_width = target_shape[1] - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        elif features.shape[1] > target_shape[1]:
            # 截断
            features = features[:, :target_shape[1]]
    
    return features

class FixedTimeSeriesAnalyzer(TimeSeriesAnalyzer):
    """修复的时间序列分析器"""
    
    def __init__(self, model, label_encoder, device, model_type):
        super().__init__(model, label_encoder, device)
        self.model_type = model_type
    
    def _extract_features(self, audio, sr):
        """提取音频特征"""
        try:
            # 提取Mel频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=128, fmax=8000, 
                n_fft=2048, hop_length=512
            )
            log_mel = librosa.power_to_db(mel_spec)
            
            # 标准化
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            
            # 确保特征尺寸一致
            log_mel = ensure_feature_shape(log_mel, (128, 130))
            
            return log_mel
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def _predict_single_window(self, features):
        """预测单个窗口"""
        # 确保特征形状正确
        features = ensure_feature_shape(features, (128, 130))
        
        # 转换为模型输入格式: (1, 1, 128, 130)
        input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # 获取所有类别的概率
        probs = probabilities.cpu().numpy()[0]
        results = {}
        
        for idx, prob in enumerate(probs):
            instrument = self.label_encoder.inverse_transform([idx])[0]
            results[instrument] = float(prob)
        
        return results
    
    def generate_report(self, timeline, audio_duration):
        """生成分析报告"""
        from src.instrument_mapper import InstrumentMapper
        
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
        
        print(f"\n{'='*60}")
        print(f"      {self.model_type}模型 - Instrument Timeline Analysis Report")
        print(f"{'='*60}")
        
        # 统计活跃乐器
        active_instruments = []
        for instrument, data in timeline.items():
            if data['segments']:
                english_name = instrument_names.get(instrument, instrument)
                active_instruments.append((english_name, data['total_duration'], instrument))
        
        # 按持续时间排序
        active_instruments.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Active Instruments Statistics ({len(active_instruments)} instruments):")
        print("-" * 60)
        
        for english_name, duration, original_name in active_instruments:
            percentage = (duration / audio_duration) * 100
            max_conf = timeline[original_name]['max_confidence']
            avg_conf = timeline[original_name]['average_confidence']
            segment_count = len(timeline[original_name]['segments'])
            
            print(f"🎵 {english_name:15s} | {duration:6.1f}s ({percentage:5.1f}%) | "
                f"Max Confidence: {max_conf:.3f} | Segments: {segment_count}")
        
        return active_instruments

def test_timeline_with_model(model_type, audio_path):
    """使用指定模型测试时间线分析"""
    print(f"\n{'='*50}")
    print(f"使用 {model_type} 模型进行时间线分析")
    print(f"{'='*50}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和标签编码器
    model, label_encoder = load_model_with_fix(model_type, device)
    if model is None or label_encoder is None:
        return None
    
    print(f"乐器类别: {list(label_encoder.classes_)}")
    print("模型加载成功!")
    
    # 创建时间序列分析器
    analyzer = FixedTimeSeriesAnalyzer(model, label_encoder, device, model_type)
    
    # 分析音频时间线
    print("\n开始时间线分析...")
    timeline = analyzer.analyze_audio_timeline(
        audio_path, 
        window_size=1.5,      # 减小窗口大小，提高时间分辨率
        hop_size=0.3,         # 减小跳跃步长，增加采样密度
        threshold=0.15         # 降低阈值，捕捉更多弱信号
    )
    
    # 获取音频时长用于可视化
    y, sr = librosa.load(audio_path, sr=22050)
    audio_duration = len(y) / sr
    
    # 生成报告
    active_instruments = analyzer.generate_report(timeline, audio_duration)
    
    # 可视化时间线
    output_path = os.path.join(Config.OUTPUT_DIR, f"instrument_timeline_{model_type}.png")
    analyzer.visualize_timeline(timeline, audio_duration, output_path)
    
    print(f"\n✅ {model_type}模型分析完成!")
    print(f"📊 发现了 {len(active_instruments)} 种活跃乐器")
    print(f"📁 时间线图已保存: {output_path}")
    
    return active_instruments

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='多模型时间线分析')
    parser.add_argument('--model-type', type=str, default='all', 
                       choices=['basic', 'simplified', 'advanced', 'all'],
                       help='要测试的模型类型')
    parser.add_argument('--audio-path', type=str, 
                       default=os.path.join(project_root, "music", "3.flac"),
                       help='测试音频路径')
    
    args = parser.parse_args()
    
    print("=== AI音频分析与自动扒谱系统 - 多模型时间线分析 ===")
    
    # 1. 初始化配置
    Config.create_directories()
    
    # 2. 检查音频文件是否存在
    if not os.path.exists(args.audio_path):
        print(f"错误: 测试音频文件不存在 - {args.audio_path}")
        print("请确保音频文件存在")
        return
    
    if args.model_type == 'all':
        # 测试所有模型
        model_types = ['basic', 'simplified', 'advanced']
        all_results = {}
        
        for model_type in model_types:
            result = test_timeline_with_model(model_type, args.audio_path)
            if result:
                all_results[model_type] = result
        
        # 输出比较结果
        print(f"\n{'='*60}")
        print("时间线分析模型比较结果")
        print(f"{'='*60}")
        
        for model_type, instruments in all_results.items():
            print(f"\n{model_type:>10}模型: 检测到 {len(instruments)} 种乐器")
            for i, (name, duration, _) in enumerate(instruments[:3]):  # 显示前3种
                print(f"            {i+1}. {name}: {duration:.1f}s")
    
    else:
        # 测试单个模型
        test_timeline_with_model(args.model_type, args.audio_path)

if __name__ == "__main__":
    main()