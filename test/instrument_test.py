import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
from pathlib import Path
import soundfile as sf
warnings.filterwarnings('ignore')

class InstrumentAnalyzer:
    def __init__(self, data_path=None):
        self.project_root = Path(__file__).parent.parent
        # self.data_path = data_path
        # self.data_path = self.project_root / 'model'
        self.data_path = self.project_root / 'model/IRMAS-TrainingData'
        self.model = None
        self.instruments = []
        self.feature_names = []
        
    def explore_dataset(self):
        """探索数据集结构"""
        if not os.path.exists(self.data_path):
            print(f"数据集路径不存在: {self.data_path}")
            return False
            
        # 获取所有文件夹，但只保留包含WAV文件的文件夹
        all_dirs = os.listdir(self.data_path)
        self.instruments = []

        for instrument in all_dirs:
            instrument_path = os.path.join(self.data_path, instrument)
            if os.path.isdir(instrument_path):
                # 检查文件夹中是否有WAV文件
                wav_files = [f for f in os.listdir(instrument_path) if f.endswith('.wav')]
                if wav_files:
                    self.instruments.append(instrument)

        print("有效的乐器类别:", self.instruments)

        total_files = 0
        for instrument in self.instruments:
            instrument_path = os.path.join(self.data_path, instrument)
            files = [f for f in os.listdir(instrument_path) if f.endswith('.wav')]
            print(f"{instrument}: {len(files)}个文件")
            total_files += len(files)
    
        print(f"\n数据集总计: {total_files}个音频文件")
        return True
    
    def extract_features(self, audio_path):
        """提取音频特征"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)  # 统一采样率
            
            # 提取多种音频特征
            features = {}
            
            # MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 频谱特征
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # 时域特征
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zero_crossing_rate)
            features['zcr_std'] = np.std(zero_crossing_rate)
            
            # 色度特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # 节奏特征
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # 将特征展平为一维向量
            feature_vector = []
            for key in sorted(features.keys()):
                if isinstance(features[key], np.ndarray):
                    feature_vector.extend(features[key])
                else:
                    feature_vector.append(features[key])
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"处理音频文件时出错 {audio_path}: {e}")
            return None
    
    def prepare_training_data(self, max_samples_per_class=200):
        """准备训练数据"""
        features_list = []
        labels_list = []
        
        print("开始提取特征...")
        for instrument in self.instruments:
            instrument_path = os.path.join(self.data_path, instrument)
            if not os.path.isdir(instrument_path):
                continue
                
            files = [f for f in os.listdir(instrument_path) if f.endswith('.wav')]
            files = files[:max_samples_per_class]  # 限制每类样本数
            
            print(f"处理 {instrument}...")
            instrument_count = 0  # 添加计数器
            
            for i, file in enumerate(files):
                file_path = os.path.join(instrument_path, file)
                features = self.extract_features(file_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(instrument)
                    instrument_count += 1  # 计数有效样本
                
                if (i + 1) % 50 == 0:
                    print(f"  已处理 {i + 1}/{len(files)} 个文件")
            print(f"  {instrument} 有效样本: {instrument_count}/{len(files)}")  # 添加输出
        
        return np.array(features_list), np.array(labels_list)
    
    def train_classifier(self, test_size=0.2):
        """训练分类器"""
        if not self.explore_dataset():
            return False
        
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("没有提取到有效特征！")
            return False
        
        print(f"\n特征矩阵形状: {X.shape}")
        print(f"标签数量: {len(y)}")
        print(f"实际类别: {np.unique(y)}")  # 添加调试信息
        
        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 训练随机森林分类器
        print("训练分类器中...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, labels=np.unique(y)))
        
        return True
    
    def analyze_audio(self, audio_path, segment_duration=3.0):
        """分析音频文件的乐器组成"""
        if self.model is None:
            print("请先训练模型！")
            return None
        
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            duration = len(y) / sr
            
            print(f"音频长度: {duration:.2f}秒")
            print(f"分段分析 (每段{segment_duration}秒)...")
            
            # 分段分析
            segment_length = int(segment_duration * sr)
            segments = []
            
            for start in range(0, len(y), segment_length):
                end = start + segment_length
                if end > len(y):
                    break
                segments.append(y[start:end])
            
            print(f"共分成 {len(segments)} 个片段")
            
            # 对每个片段进行预测
            predictions = []
            confidences = []
            
            for i, segment in enumerate(segments):
                # 临时保存片段并提取特征
                temp_path = f"temp_segment_{i}.wav"
                sf.write(temp_path, segment, sr)
                
                features = self.extract_features(temp_path)
                if features is not None:
                    # 预测并获取概率
                    # # proba = self.model.predict_proba([features])[0]
                    predicted_idx = np.argmax(proba)
                    predicted_instrument = self.model.classes_[predicted_idx]
                    confidence = proba[predicted_idx]
                    
                    predictions.append(predicted_instrument)
                    confidences.append(confidence)
                
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                if (i + 1) % 10 == 0:
                    print(f"  已分析 {i + 1}/{len(segments)} 个片段")
            
            # 统计结果
            if predictions:
                unique, counts = np.unique(predictions, return_counts=True)
                total_segments = len(predictions)
                
                print(f"\n=== 音频分析结果 ===")
                print(f"分析片段总数: {total_segments}")
                print(f"平均置信度: {np.mean(confidences):.4f}")
                print(f"\n乐器分布:")
                
                results = {}
                for instrument, count in zip(unique, counts):
                    percentage = (count / total_segments) * 100
                    instrument_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == instrument]
                    avg_confidence = np.mean(instrument_confidences) if instrument_confidences else 0
                    
                    results[instrument] = {
                        'percentage': percentage,
                        'confidence': avg_confidence,
                        'count': count
                    }
                    
                    print(f"  {instrument}: {percentage:.1f}% (置信度: {avg_confidence:.4f})")
                
                # 可视化结果
                self.visualize_results(results, audio_path)
                return results
            else:
                print("无法分析该音频文件")
                return None
                
        except Exception as e:
            print(f"分析音频时出错: {e}")
            return None
    
    def visualize_results(self, results, audio_path):
        """可视化分析结果"""
        instruments = list(results.keys())
        percentages = [results[inst]['percentage'] for inst in instruments]
        confidences = [results[inst]['confidence'] for inst in instruments]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 饼图显示乐器分布
        ax1.pie(percentages, labels=instruments, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'乐器分布 - {os.path.basename(audio_path)}')
        
        # 柱状图显示置信度
        bars = ax2.bar(instruments, confidences, color='skyblue', alpha=0.7)
        ax2.set_ylabel('平均置信度')
        ax2.set_title('各乐器识别置信度')
        ax2.set_ylim(0, 1)
        
        # 在柱状图上添加数值
        for bar, confidence in zip(bars, confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{confidence:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path='instrument_classifier.pkl'):
        """保存训练好的模型"""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'instruments': self.instruments
            }, model_path)
            print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path='instrument_classifier.pkl'):
        """加载训练好的模型"""
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.instruments = data['instruments']
            print(f"模型已从 {model_path} 加载")
            return True
        else:
            print(f"模型文件不存在: {model_path}")
            return False

# 使用示例
def main():
    # 初始化分析器
    analyzer = InstrumentAnalyzer(data_path="models/IRMAS-TrainingData")
    
    # 检查是否有保存的模型
    if not analyzer.load_model():
        print("未找到保存的模型，开始训练新模型...")
        # 训练分类器
        if analyzer.train_classifier():
            # 保存模型
            analyzer.save_model()
    
    # 分析新的音频文件
    project_path = Path(__file__).parent.parent
    test_audio_path = Path(__file__).parent.parent / "music/4.flac"  # 替换为你的音频文件路径
    
    if os.path.exists(test_audio_path):
        results = analyzer.analyze_audio(test_audio_path)
        if results:
            print("\n分析完成！")
    else:
        print(f"测试音频文件不存在: {test_audio_path}")
        print("请将 'your_audio_file.wav' 替换为你的音频文件路径")

if __name__ == "__main__":
    main()