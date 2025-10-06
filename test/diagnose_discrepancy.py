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
from config.music_file_loader import load_music_files
from src.instrument_mapper import InstrumentMapper

def diagnose_analysis_batch():
    """批量诊断分析不一致问题"""
    # 加载音乐文件
    music_files = load_music_files()
    if not music_files:
        print("没有找到可诊断的音乐文件")
        return
    
    print(f"开始批量诊断分析 {len(music_files)} 个音乐文件...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, audio_path in enumerate(music_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(music_files)}] 诊断分析: {os.path.basename(audio_path)}")
        print(f"{'='*60}")
        
        # 这里可以调用原来的诊断函数，但需要调整以支持批量处理
        # diagnose_single_analysis(audio_path, device)
        
        # 简化版的诊断分析
        try:
            # 加载第一个可用模型进行诊断
            model_type = 'simplified'
            model_path = os.path.join(project_root, "model", f"model_{model_type}.pth")
            label_encoder_path = os.path.join(project_root, "model", f"model_{model_type}_label_encoder.pkl")
            
            if os.path.exists(model_path) and os.path.exists(label_encoder_path):
                diagnose_single_analysis(audio_path, model_path, label_encoder_path, device)
            else:
                print(f"模型文件不存在，跳过诊断")
                
        except Exception as e:
            print(f"诊断过程中出错: {e}")

def diagnose_single_analysis(audio_path, model_path, label_encoder_path, device):
    """诊断单个文件的分析不一致问题"""
    # 加载模型和标签编码器
    label_encoder = joblib.load(label_encoder_path)
    checkpoint = torch.load(model_path, map_location=device)
    
    from src.model_builder import ImprovedInstrumentClassifier
    model = ImprovedInstrumentClassifier((1, 128, 130), len(label_encoder.classes_))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 加载音频
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    print(f"音频时长: {duration:.2f}秒")
    
    # 测试1: 整体分析
    print("\n1. 整体分析（3秒片段）:")
    test_overall_analysis(y, sr, model, label_encoder, device)
    
    # 测试2: 多个窗口分析
    print("\n2. 多窗口分析:")
    test_multiple_windows(y, sr, model, label_encoder, device)
    
    # 测试3: 置信度分布
    print("\n3. 置信度分布分析:")
    test_confidence_distribution(y, sr, model, label_encoder, device)

# 保留原有的测试函数（test_overall_analysis, test_multiple_windows等）
def test_overall_analysis(y, sr, model, label_encoder, device):
    """测试整体分析"""
    segment = y[:3*sr] if len(y) > 3*sr else y
    features = extract_features(segment, sr)
    if features is not None:
        probs = predict_single_window(features, model, label_encoder, device)
        print_top_predictions(probs, label_encoder, "整体分析")

def test_multiple_windows(y, sr, model, label_encoder, device):
    """测试多个窗口"""
    window_size = 3.0
    hop_size = 1.0
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    all_predictions = []
    
    for start in range(0, min(len(y) - window_samples, 10 * hop_samples), hop_samples):
        end = start + window_samples
        segment = y[start:end]
        features = extract_features(segment, sr)
        if features is not None:
            probs = predict_single_window(features, model, label_encoder, device)
            all_predictions.append(probs)
    
    if all_predictions:
        avg_probs = np.mean(all_predictions, axis=0)
        print_top_predictions(avg_probs, label_encoder, "多窗口平均")

def test_confidence_distribution(y, sr, model, label_encoder, device):
    """测试置信度分布"""
    window_size = 3.0
    hop_size = 1.0
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    instrument_confidences = {inst: [] for inst in label_encoder.classes_}
    
    for start in range(0, min(len(y) - window_samples, 20 * hop_samples), hop_samples):
        end = start + window_samples
        segment = y[start:end]
        features = extract_features(segment, sr)
        if features is not None:
            probs = predict_single_window(features, model, label_encoder, device)
            for idx, prob in enumerate(probs):
                instrument = label_encoder.inverse_transform([idx])[0]
                instrument_confidences[instrument].append(prob)
    
    print("\n置信度统计:")
    for instrument, confidences in instrument_confidences.items():
        if confidences:
            english_name = InstrumentMapper.get_english_name(instrument)
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            print(f"  {english_name:15s}: 平均{avg_conf:.3f}, 最大{max_conf:.3f}")

def extract_features(audio, sr, target_shape=(128, 130)):
    """提取特征"""
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=8000, 
            n_fft=2048, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
        
        if log_mel.shape[1] < target_shape[1]:
            log_mel = np.pad(log_mel, ((0, 0), (0, target_shape[1] - log_mel.shape[1])), mode='constant')
        else:
            log_mel = log_mel[:, :target_shape[1]]
            
        return log_mel
    except Exception as e:
        print(f"特征提取错误: {e}")
        return None

def predict_single_window(features, model, label_encoder, device):
    """预测单个窗口"""
    input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    return probabilities.cpu().numpy()[0]

def print_top_predictions(probs, label_encoder, title):
    """打印前3个预测"""
    top_indices = np.argsort(probs)[-3:][::-1]
    
    print(f"\n{title}:")
    for i, idx in enumerate(top_indices):
        instrument = label_encoder.inverse_transform([idx])[0]
        english_name = InstrumentMapper.get_english_name(instrument)
        prob = probs[idx]
        print(f"  {i+1}. {english_name:15s}: {prob:.3f}")

def main():
    """主诊断函数"""
    parser = argparse.ArgumentParser(description='批量诊断分析')
    
    args = parser.parse_args()
    
    print("=== 批量分析不一致诊断 ===")
    
    # 初始化配置
    Config.create_directories()
    
    # 执行批量诊断
    diagnose_analysis_batch()

if __name__ == "__main__":
    main()