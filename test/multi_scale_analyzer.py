import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MultiScaleTimeSeriesAnalyzer:
    """多尺度时间序列分析器"""
    
    def __init__(self, model, label_encoder, device):
        self.model = model
        self.label_encoder = label_encoder
        self.device = device
        self.model.eval()
    
    def multi_scale_analysis(self, audio_path, scales=None):
        """
        多尺度分析
        
        Args:
            audio_path: 音频文件路径
            scales: 分析尺度配置
        """
        if scales is None:
            scales = [
                {'window_size': 3.0, 'hop_size': 1.0, 'threshold': 0.3, 'weight': 1.0, 'name': 'Coarse'},
                {'window_size': 1.5, 'hop_size': 0.5, 'threshold': 0.2, 'weight': 0.8, 'name': 'Medium'},
                {'window_size': 1.0, 'hop_size': 0.3, 'threshold': 0.15, 'weight': 0.6, 'name': 'Fine'}
            ]
        
        print("开始多尺度分析...")
        all_results = []
        
        for scale in scales:
            print(f"\n🔧 运行 {scale['name']} 尺度分析:")
            print(f"   窗口: {scale['window_size']}s, 跳跃: {scale['hop_size']}s, 阈值: {scale['threshold']}")
            
            timeline = self._single_scale_analysis(audio_path, scale)
            all_results.append({
                'scale': scale,
                'timeline': timeline
            })
        
        # 融合多尺度结果
        fused_timeline = self._fuse_multi_scale_results(all_results)
        return fused_timeline
    
    def _single_scale_analysis(self, audio_path, scale_config):
        """单尺度分析"""
        y, sr = librosa.load(audio_path, sr=22050)
        
        window_size = scale_config['window_size']
        hop_size = scale_config['hop_size']
        threshold = scale_config['threshold']
        
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        predictions = []
        timestamps = []
        
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            end = start + window_samples
            window_audio = y[start:end]
            timestamp = start / sr
            
            features = self._extract_features(window_audio, sr)
            if features is not None:
                prediction = self._predict_single_window(features)
                predictions.append(prediction)
                timestamps.append(timestamp)
        
        return self._process_timeline_results(predictions, timestamps, window_size, threshold)
    
    def _fuse_multi_scale_results(self, all_results):
        """融合多尺度结果"""
        fused_timeline = {}
        
        # 初始化时间线
        for instrument in self.label_encoder.classes_:
            fused_timeline[instrument] = {
                'segments': [],
                'total_duration': 0.0,
                'max_confidence': 0.0,
                'average_confidence': 0.0,
                'scale_scores': []  # 记录来自不同尺度的得分
            }
        
        # 融合逻辑
        for result in all_results:
            scale = result['scale']
            timeline = result['timeline']
            weight = scale['weight']
            
            for instrument, data in timeline.items():
                for segment in data['segments']:
                    # 加权融合
                    weighted_segment = segment.copy()
                    weighted_segment['confidence'] *= weight
                    weighted_segment['scale'] = scale['name']
                    weighted_segment['original_confidence'] = segment['confidence']
                    
                    fused_timeline[instrument]['segments'].append(weighted_segment)
        
        # 合并和过滤融合后的时间段
        for instrument in fused_timeline.keys():
            segments = fused_timeline[instrument]['segments']
            if segments:
                # 按开始时间排序
                segments.sort(key=lambda x: x['start'])
                
                # 合并重叠时间段
                merged_segments = []
                current_segment = segments[0]
                
                for segment in segments[1:]:
                    if self._segments_overlap(current_segment, segment):
                        # 合并重叠段，取最大置信度
                        current_segment['end'] = max(current_segment['end'], segment['end'])
                        current_segment['confidence'] = max(current_segment['confidence'], segment['confidence'])
                        if 'scale_scores' not in current_segment:
                            current_segment['scale_scores'] = []
                        current_segment['scale_scores'].append({
                            'scale': segment.get('scale', 'unknown'),
                            'confidence': segment.get('original_confidence', segment['confidence'])
                        })
                    else:
                        merged_segments.append(current_segment)
                        current_segment = segment
                
                merged_segments.append(current_segment)
                
                # 过滤低置信度段
                filtered_segments = [seg for seg in merged_segments if seg['confidence'] >= 0.15]
                
                fused_timeline[instrument]['segments'] = filtered_segments
                
                # 计算统计信息
                if filtered_segments:
                    total_duration = sum(seg['end'] - seg['start'] for seg in filtered_segments)
                    confidences = [seg['confidence'] for seg in filtered_segments]
                    
                    fused_timeline[instrument]['total_duration'] = total_duration
                    fused_timeline[instrument]['max_confidence'] = max(confidences)
                    fused_timeline[instrument]['average_confidence'] = np.mean(confidences)
        
        return fused_timeline
    
    def _segments_overlap(self, seg1, seg2, gap_tolerance=0.5):
        """判断两个时间段是否重叠"""
        return seg1['end'] + gap_tolerance >= seg2['start'] and seg2['end'] + gap_tolerance >= seg1['start']
    
    def _extract_features(self, audio, sr, target_shape=(128, 130)):
        """提取音频特征"""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=128, fmax=8000, 
                n_fft=2048, hop_length=512
            )
            log_mel = librosa.power_to_db(mel_spec)
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            
            # 调整形状
            if log_mel.shape[1] < target_shape[1]:
                log_mel = np.pad(log_mel, ((0, 0), (0, target_shape[1] - log_mel.shape[1])), mode='constant')
            else:
                log_mel = log_mel[:, :target_shape[1]]
                
            return log_mel
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def _predict_single_window(self, features):
        """预测单个窗口"""
        input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        probs = probabilities.cpu().numpy()[0]
        results = {}
        
        for idx, prob in enumerate(probs):
            instrument = self.label_encoder.inverse_transform([idx])[0]
            results[instrument] = float(prob)
        
        return results
    
    def _process_timeline_results(self, predictions, timestamps, window_size, threshold):
        """处理时间线结果"""
        timeline = {}
        
        for instrument in self.label_encoder.classes_:
            timeline[instrument] = {
                'segments': [],
                'total_duration': 0.0,
                'max_confidence': 0.0,
                'average_confidence': 0.0
            }
        
        for timestamp, pred_dict in zip(timestamps, predictions):
            if pred_dict:
                best_instrument, best_confidence = max(pred_dict.items(), key=lambda x: x[1])
                if best_confidence >= threshold:
                    timeline[best_instrument]['segments'].append({
                        'start': timestamp,
                        'end': timestamp + window_size,
                        'confidence': best_confidence
                    })
        
        # 合并连续时间段
        for instrument in timeline.keys():
            segments = timeline[instrument]['segments']
            if segments:
                segments.sort(key=lambda x: x['start'])
                merged_segments = []
                current_segment = segments[0]
                
                for segment in segments[1:]:
                    if segment['start'] <= current_segment['end'] + 0.5:
                        current_segment['end'] = max(current_segment['end'], segment['end'])
                        current_segment['confidence'] = max(current_segment['confidence'], segment['confidence'])
                    else:
                        merged_segments.append(current_segment)
                        current_segment = segment
                
                merged_segments.append(current_segment)
                timeline[instrument]['segments'] = merged_segments
                
                # 计算统计
                total_duration = sum(seg['end'] - seg['start'] for seg in merged_segments)
                confidences = [seg['confidence'] for seg in merged_segments]
                
                timeline[instrument]['total_duration'] = total_duration
                timeline[instrument]['max_confidence'] = max(confidences) if confidences else 0
                timeline[instrument]['average_confidence'] = np.mean(confidences) if confidences else 0
        
        return timeline