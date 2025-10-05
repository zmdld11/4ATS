import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import joblib

class TimeSeriesAnalyzer:
    """时间序列乐器分析器"""
    
    def __init__(self, model, label_encoder, device):
        self.model = model
        self.label_encoder = label_encoder
        self.device = device
        self.model.eval()
    
    def analyze_audio_timeline(self, audio_path, window_size=3.0, hop_size=1.0, threshold=0.3):
        """
        分析音频时间线，检测乐器出现的时间段
        
        Args:
            audio_path: 音频文件路径
            window_size: 分析窗口大小（秒）
            hop_size: 窗口跳跃大小（秒）
            threshold: 置信度阈值
        """
        print(f"开始分析音频时间线: {audio_path}")
        
        # 加载完整音频
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        print(f"音频时长: {duration:.2f}秒, 采样率: {sr}Hz")
        
        # 计算窗口参数
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        # 滑动窗口分析
        predictions = []
        timestamps = []
        
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            # 提取当前窗口
            end = start + window_samples
            window_audio = y[start:end]
            timestamp = start / sr  # 当前窗口开始时间
            
            # 提取特征
            features = self._extract_features(window_audio, sr)
            if features is not None:
                # 预测
                prediction = self._predict_single_window(features)
                predictions.append(prediction)
                timestamps.append(timestamp)
            
            # 显示进度
            if len(predictions) % 10 == 0:
                progress = min(100, (end / len(y)) * 100)
                print(f"分析进度: {progress:.1f}%")
        
        print("时间序列分析完成!")
        return self._process_timeline_results(predictions, timestamps, window_size, threshold)
    
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
            target_frames = 130  # 与训练时一致
            if log_mel.shape[1] < target_frames:
                # 填充
                pad_width = target_frames - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
            elif log_mel.shape[1] > target_frames:
                # 截断
                log_mel = log_mel[:, :target_frames]
            
            return log_mel
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def _predict_single_window(self, features):
        """预测单个窗口"""
        # 转换为模型输入格式
        input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 130)
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
    
    def _process_timeline_results(self, predictions, timestamps, window_size, threshold):
        """处理时间线结果，合并连续的时间段"""
        timeline = {}
        
        # 为每个乐器初始化时间线
        for instrument in self.label_encoder.classes_:
            timeline[instrument] = {
                'segments': [],
                'total_duration': 0.0,
                'max_confidence': 0.0,
                'average_confidence': 0.0
            }
        
        # 分析每个时间点的预测
        for i, (timestamp, pred_dict) in enumerate(zip(timestamps, predictions)):
            # 找到当前窗口最可能的乐器
            if pred_dict:
                best_instrument = max(pred_dict.items(), key=lambda x: x[1])
                instrument, confidence = best_instrument
                
                # 只记录置信度高于阈值的预测
                if confidence >= threshold:
                    timeline[instrument]['segments'].append({
                        'start': timestamp,
                        'end': timestamp + window_size,
                        'confidence': confidence
                    })
        
        # 合并连续的时间段并计算统计信息
        for instrument in timeline.keys():
            segments = timeline[instrument]['segments']
            if segments:
                # 按开始时间排序
                segments.sort(key=lambda x: x['start'])
                
                # 合并连续的时间段
                merged_segments = []
                current_segment = segments[0]
                
                for segment in segments[1:]:
                    if segment['start'] <= current_segment['end'] + 0.5:  # 允许0.5秒间隙
                        # 时间段连续，合并
                        current_segment['end'] = max(current_segment['end'], segment['end'])
                        current_segment['confidence'] = max(current_segment['confidence'], segment['confidence'])
                    else:
                        # 时间段不连续，保存当前段并开始新段
                        merged_segments.append(current_segment)
                        current_segment = segment
                
                merged_segments.append(current_segment)
                
                # 更新timeline
                timeline[instrument]['segments'] = merged_segments
                
                # 计算统计信息
                total_duration = sum(seg['end'] - seg['start'] for seg in merged_segments)
                confidences = [seg['confidence'] for seg in merged_segments]
                
                timeline[instrument]['total_duration'] = total_duration
                timeline[instrument]['max_confidence'] = max(confidences) if confidences else 0
                timeline[instrument]['average_confidence'] = np.mean(confidences) if confidences else 0
        
        return timeline
    
    def visualize_timeline(self, timeline, audio_duration, save_path=None):
        """可视化时间线结果"""
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
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 使用英文名称
        english_instruments = [instrument_names.get(instr, instr) for instr in timeline.keys()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(english_instruments)))
        
        # 为每个乐器创建时间线
        for i, (instrument, data) in enumerate(timeline.items()):
            english_name = instrument_names.get(instrument, instrument)
            segments = data['segments']
            if segments:
                for segment in segments:
                    # 绘制时间段矩形
                    rect = Rectangle(
                        (segment['start'], i - 0.4),
                        segment['end'] - segment['start'],
                        0.8,
                        facecolor=colors[i],
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=1
                    )
                    ax.add_patch(rect)
                    
                    # 添加置信度文本
                    if segment['end'] - segment['start'] > 2:  # 只在不小的段上添加文本
                        ax.text(
                            (segment['start'] + segment['end']) / 2,
                            i,
                            f'{segment["confidence"]:.2f}',
                            ha='center',
                            va='center',
                            fontsize=8,
                            fontweight='bold'
                        )
        
        # 设置坐标轴
        ax.set_xlim(0, audio_duration)
        ax.set_ylim(-0.5, len(english_instruments) - 0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Instrument')
        ax.set_title('Instrument Timeline Analysis')
        
        # 设置y轴刻度（使用英文名称）
        ax.set_yticks(range(len(english_instruments)))
        ax.set_yticklabels(english_instruments)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = []
        for i, (instrument, data) in enumerate(timeline.items()):
            english_name = instrument_names.get(instrument, instrument)
            if data['segments']:
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, 
                                label=f"{english_name} ({data['total_duration']:.1f}s)")
                )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline chart saved: {save_path}")
        
        plt.show()
        
        return fig

    def generate_report(self, timeline, audio_duration):
        """生成分析报告"""
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
        
        print("\n" + "="*60)
        print("               Instrument Timeline Analysis Report")
        print("="*60)
        
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
        
        print(f"\n📈 Detailed Time Segments:")
        print("-" * 50)
        
        for english_name, duration, original_name in active_instruments:
            segments = timeline[original_name]['segments']
            
            print(f"\n{english_name}:")
            for i, segment in enumerate(segments, 1):
                print(f"  Segment {i}: {segment['start']:6.1f}s - {segment['end']:6.1f}s "
                    f"(Confidence: {segment['confidence']:.3f})")
        
        return active_instruments