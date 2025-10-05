# audio_preprocessor.py (添加数据增强 - 中文注释)
import os
import librosa
import numpy as np
import pickle
import hashlib
import torch
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config.config import Config

class AudioDataset(Dataset):
    """带有数据增强的PyTorch音频数据集"""
    
    def __init__(self, features, labels, transform=None, augment=False):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx].copy()  # 重要：创建副本
        label = self.labels[idx]
        
        # 训练期间的数据增强
        if self.augment:
            feature = self._apply_augmentation(feature)
        
        # 转换为PyTorch张量
        feature = torch.FloatTensor(feature).unsqueeze(0)  # 添加通道维度
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label
    
    def _apply_augmentation(self, feature):
        """对音频特征应用数据增强"""
        # 时间掩码（类似于SpecAugment）
        if random.random() > 0.5:
            max_mask_width = feature.shape[1] // 4
            mask_width = random.randint(1, max_mask_width)
            mask_start = random.randint(0, feature.shape[1] - mask_width)
            feature[:, mask_start:mask_start + mask_width] = 0
        
        # 频率掩码
        if random.random() > 0.5:
            max_mask_height = feature.shape[0] // 8
            mask_height = random.randint(1, max_mask_height)
            mask_start = random.randint(0, feature.shape[0] - mask_height)
            feature[mask_start:mask_start + mask_height, :] = 0
        
        # 添加小噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.01, feature.shape)
            feature = feature + noise
        
        return feature

class AudioDataPreprocessor:
    """带有增强功能和数据增强的音频数据预处理器"""
    
    def __init__(self, target_sr=Config.TARGET_SAMPLE_RATE, 
                 duration=Config.AUDIO_DURATION,
                 use_cache=True):
        self.target_sr = target_sr
        self.duration = duration
        self.label_encoder = LabelEncoder()
        self.use_cache = use_cache
        self.cache_dir = os.path.join(Config.DATA_DIR, "preprocessed_cache")
        
        # 创建缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def extract_enhanced_features(self, audio_path):
        """提取带有多种表示的增强音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 确保音频长度一致
            y = librosa.util.fix_length(y, size=self.target_sr * self.duration)
            
            # 提取多种特征
            # 1. Mel频谱图（主要特征）
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=Config.N_MELS, fmax=8000, 
                n_fft=2048, hop_length=512
            )
            log_mel = librosa.power_to_db(mel_spec)
            
            # 2. MFCC特征
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # 3. Chroma特征
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 组合特征（可以选择使用哪些）
            # 选项1：仅使用mel频谱图（与当前架构兼容）
            combined_features = log_mel
            
            # 选项2：堆叠多个特征（需要更改模型架构）
            # combined_features = np.vstack([log_mel, mfcc, chroma])
            
            # 标准化
            combined_features = (combined_features - np.mean(combined_features)) / (np.std(combined_features) + 1e-8)
            
            return combined_features
            
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def create_data_loaders(self, batch_size=Config.BATCH_SIZE, use_cache=True, augment=True):
        """创建带有可选数据增强的PyTorch数据加载器"""
        # 加载音频样本
        audio_paths, labels = self.load_audio_samples(Config.DATA_DIR)
        
        # 提取特征（使用缓存）
        X, y = self.prepare_dataset(audio_paths, labels, use_cache=use_cache)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.VALIDATION_SPLIT, 
            random_state=42, stratify=y
        )
        
        # 进一步划分训练集用于验证
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, 
            random_state=42, stratify=y_train
        )
        
        # 创建数据集
        train_dataset = AudioDataset(X_train, y_train, augment=augment)
        val_dataset = AudioDataset(X_val, y_val, augment=False)
        test_dataset = AudioDataset(X_test, y_test, augment=False)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(val_dataset)}")
        print(f"测试样本数: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, len(self.label_encoder.classes_)

    def _get_cache_filename(self, data_dir):
        """生成缓存文件名"""
        config_str = f"{data_dir}_{self.target_sr}_{self.duration}_{Config.N_MELS}"
        hash_obj = hashlib.md5(config_str.encode())
        return os.path.join(self.cache_dir, f"preprocessed_{hash_obj.hexdigest()}.pkl")
    
    def _save_to_cache(self, cache_file, X, y, label_encoder):
        """保存预处理结果到缓存"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'features': X,
                    'labels': y,
                    'label_encoder': label_encoder,
                    'config': {
                        'target_sr': self.target_sr,
                        'duration': self.duration,
                        'n_mels': Config.N_MELS
                    }
                }, f)
            print(f"✅ 预处理数据已缓存到: {cache_file}")
            return True
        except Exception as e:
            print(f"❌ 缓存保存失败: {e}")
            return False
    
    def _load_from_cache(self, cache_file):
        """从缓存加载预处理结果"""
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 验证配置是否匹配
            config = cache_data['config']
            if (config['target_sr'] == self.target_sr and 
                config['duration'] == self.duration and 
                config['n_mels'] == Config.N_MELS):
                
                print("✅ 从缓存加载预处理数据")
                return cache_data['features'], cache_data['labels'], cache_data['label_encoder']
            else:
                print("⚠️ 缓存配置不匹配，重新预处理")
                return None, None, None
                
        except Exception as e:
            print(f"❌ 缓存加载失败: {e}")
            return None, None, None
    
    def load_audio_samples(self, data_dir):
        """加载音频样本和标签"""
        samples = []
        labels = []
        
        dataset_path = os.path.join(data_dir, Config.DATASET_NAME)
        print(f"从 {dataset_path} 加载数据...")
        
        # 遍历每个乐器文件夹
        for instrument in os.listdir(dataset_path):
            instrument_path = os.path.join(dataset_path, instrument)
            if os.path.isdir(instrument_path):
                for audio_file in os.listdir(instrument_path):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(instrument_path, audio_file)
                        samples.append(audio_path)
                        labels.append(instrument)
        
        print(f"找到 {len(samples)} 个音频样本")
        print(f"乐器类别: {set(labels)}")
        
        return samples, labels
    
    def prepare_dataset(self, audio_paths, labels, use_cache=True):
        """准备训练数据集 - 带缓存功能"""
        cache_file = self._get_cache_filename(Config.DATA_DIR)
        
        # 尝试从缓存加载
        if use_cache and os.path.exists(cache_file):
            X, y, label_encoder = self._load_from_cache(cache_file)
            if X is not None and y is not None:
                self.label_encoder = label_encoder
                return X, y
        
        # 缓存不存在或不可用，重新处理
        print("🔄 预处理音频数据（这可能需要一些时间）...")
        features = []
        valid_labels = []
        
        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            if i % 100 == 0:  # 每100个样本显示进度
                print(f"已处理 {i}/{len(audio_paths)} 个样本 ({i/len(audio_paths)*100:.1f}%)")
                
            feature = self.extract_enhanced_features(path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(label)
        
        # 转换为numpy数组
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        
        print(f"最终数据集形状: {X.shape}")
        
        # 保存到缓存
        if use_cache:
            self._save_to_cache(cache_file, X, y, self.label_encoder)
        
        return X, y