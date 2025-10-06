当然可以！我来为您详细介绍机器学习模型训练的全流程，包括平台选择、数据获取、训练步骤等。

## 1. 训练平台选择

### 1.1 免费平台（推荐初学者）

**Google Colab（最推荐）**
- 优点：免费GPU、预装环境、在线运行、容易分享
- 缺点：有使用时长限制（需要重新连接）
- 网址：colab.research.google.com

**Kaggle Kernels**
- 优点：免费GPU、内置数据集、竞赛环境
- 缺点：GPU时间限制更严格
- 网址：kaggle.com/kernels

**本地训练**
- 优点：完全控制、无时间限制
- 缺点：需要较好的硬件（GPU）、环境配置复杂

### 1.2 付费平台（项目后期考虑）

- **AWS SageMaker**
- **Google Cloud AI Platform**
- **Azure Machine Learning**

## 2. 数据获取途径

### 2.1 公开音频数据集

**乐器分类专用数据集：**
1. **MedleyDB**（最推荐）
   - 包含多乐器混合音频及分离音轨
   - 网址：medleydb.weebly.com
   - 特点：专业标注，适合乐器识别

2. **NSynth**（谷歌发布）
   - 30万条乐器音符样本
   - 网址：magenta.tensorflow.org/datasets/nsynth
   - 特点：音高准确，适合基础训练

3. **IRMAS**（乐器识别数据集）
   - 包含11种乐器的独奏片段
   - 网址：www.upf.edu/web/mtg/irmas
   - 特点：专门为乐器识别设计

4. **Freesound Dataset**
   - 用户上传的各类声音
   - 网址：freesound.org
   - 特点：数据量大但质量不一

### 2.2 数据准备工具

```python
# 数据下载和预处理示例
import os
import urllib.request
import tarfile
import librosa
import numpy as np

def download_and_prepare_dataset():
    """下载和预处理数据集"""
    
    # 下载数据集（以IRMAS为例）
    dataset_url = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip"
    local_path = "data/irmas.zip"
    
    if not os.path.exists(local_path):
        print("下载数据集...")
        urllib.request.urlretrieve(dataset_url, local_path)
        
        # 解压
        with tarfile.open(local_path, 'r:zip') as tar:
            tar.extractall("data/")
    
    return "data/IRMAS-TrainingData/"

def load_audio_samples(data_dir):
    """加载音频样本和标签"""
    samples = []
    labels = []
    
    # 遍历每个乐器文件夹
    for instrument in os.listdir(data_dir):
        instrument_path = os.path.join(data_dir, instrument)
        if os.path.isdir(instrument_path):
            for audio_file in os.listdir(instrument_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(instrument_path, audio_file)
                    samples.append(audio_path)
                    labels.append(instrument)
    
    return samples, labels
```

## 3. 模型训练完整流程

### 3.1 环境设置（Google Colab示例）

```python
# 在Colab中运行这些命令来设置环境
!pip install tensorflow
!pip install librosa
!pip install matplotlib
!pip install scikit-learn

import tensorflow as tf
print("GPU可用:", tf.test.is_gpu_available())
```

### 3.2 数据预处理流程

```python
import tensorflow as tf
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class AudioDataPreprocessor:
    def __init__(self, target_sr=22050, duration=3):
        self.target_sr = target_sr
        self.duration = duration  # 每个样本的秒数
        self.label_encoder = LabelEncoder()
    
    def extract_features(self, audio_path):
        """提取音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 确保音频长度一致
            if len(y) < self.target_sr * self.duration:
                y = np.pad(y, (0, self.target_sr * self.duration - len(y)))
            else:
                y = y[:self.target_sr * self.duration]
            
            # 提取Mel频谱图（最常用的特征）
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=8000
            )
            log_mel = librosa.power_to_db(mel_spectrogram)
            
            # 标准化
            log_mel = (log_mel - np.mean(log_mel)) / np.std(log_mel)
            
            return log_mel
            
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def prepare_dataset(self, audio_paths, labels):
        """准备训练数据集"""
        features = []
        valid_labels = []
        
        for path, label in zip(audio_paths, labels):
            feature = self.extract_features(path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(label)
        
        # 转换为numpy数组
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        
        # 添加通道维度（CNN需要）
        X = X[..., np.newaxis]
        
        return X, y
```

### 3.3 模型构建

```python
from tensorflow.keras import layers, models

def create_instrument_classifier(input_shape, num_classes):
    """创建乐器分类CNN模型"""
    
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 第三个卷积块
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 全局平均池化 + 全连接层
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# 创建模型
input_shape = (128, 130, 1)  # Mel频谱图的形状
num_classes = 10  # 根据您的乐器类别数量调整
model = create_instrument_classifier(input_shape, num_classes)

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### 3.4 训练过程

```python
def train_model():
    """完整的训练流程"""
    
    # 1. 准备数据
    preprocessor = AudioDataPreprocessor()
    audio_paths, labels = load_audio_samples("data/IRMAS-TrainingData/")
    X, y = preprocessor.prepare_dataset(audio_paths, labels)
    
    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 数据增强
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # 4. 训练模型
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # 5. 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试准确率: {test_acc:.4f}")
    
    # 6. 保存模型
    model.save("instrument_classifier.h5")
    import joblib
    joblib.dump(preprocessor.label_encoder, "label_encoder.pkl")
    
    return model, preprocessor.label_encoder
```

## 4. 实际代码使用

### 4.1 解压训练资源库，打开模型保存路径

```pyt
from google.colab import drive
import zipfile
import os

# 挂载Google Drive
drive.mount('/content/drive')

# 假设你的IRMAS数据在Google Drive的这个路径
dataset_path = '/content/drive/MyDrive/IRMAS-TrainingData.zip'

# 复制到Colab环境并解压
!cp "{dataset_path}" /content/
!unzip -q /content/IRMAS-TrainingData.zip -d /content/data/

# 定义模型保存路径
model_save_dir = "/content/drive/MyDrive/4ATS/model"
# 如果模型保存路径不存在，则创建它
os.makedirs(model_save_dir, exist_ok=True)
print(f"模型保存目录已准备就绪: {model_save_dir}")
```

### 4.2 配置环境

```py
# 安装库
!pip install TensorFlow
!pip install librosa
!pip install matplotlib
!pip install scikit-learn

# 导入库
import tensorflow as tf
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import zipfile

print("GPU可用:", tf.test.is_gpu_available())

# 数据准备代码（上面修复后的文件路径代码）
```

### 4.3 数据预处理流程 - 定义类

```py
# 3.2 数据预处理流程 - 定义类
class AudioDataPreprocessor:
    def __init__(self, target_sr=22050, duration=3):
        self.target_sr = target_sr
        self.duration = duration
        self.label_encoder = LabelEncoder()
    
    def extract_features(self, audio_path):
        """提取音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 确保音频长度一致
            if len(y) < self.target_sr * self.duration:
                y = np.pad(y, (0, self.target_sr * self.duration - len(y)))
            else:
                y = y[:self.target_sr * self.duration]
            
            # 提取Mel频谱图
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=8000
            )
            log_mel = librosa.power_to_db(mel_spectrogram)
            
            # 标准化
            log_mel = (log_mel - np.mean(log_mel)) / np.std(log_mel)
            
            return log_mel
            
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def prepare_dataset(self, audio_paths, labels):
        """准备训练数据集"""
        features = []
        valid_labels = []
        
        for path, label in zip(audio_paths, labels):
            feature = self.extract_features(path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(label)
        
        # 转换为numpy数组
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        
        # 添加通道维度（CNN需要）
        X = X[..., np.newaxis]
        
        return X, y

# 数据加载函数
def load_audio_samples(data_dir):
    """加载音频样本和标签"""
    samples = []
    labels = []
    
    # 遍历每个乐器文件夹
    for instrument in os.listdir(data_dir):
        instrument_path = os.path.join(data_dir, instrument)
        if os.path.isdir(instrument_path):
            for audio_file in os.listdir(instrument_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(instrument_path, audio_file)
                    samples.append(audio_path)
                    labels.append(instrument)
    
    return samples, labels
```

### 4.4 模型构建

```py
# 3.3 模型构建
from tensorflow.keras import layers, models

def create_instrument_classifier(input_shape, num_classes):
    """创建乐器分类CNN模型"""
    
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 第二个卷积块
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 第三个卷积块
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # 全局平均池化 + 全连接层
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

print("模型构建完成")
```

### 4.5 训练流程

```py
# 3.4 训练流程
def train_model():
    """完整的训练流程"""
    
    # 1. 准备数据
    data_dir = "/content/data/IRMAS-TrainingData/"
    print("数据目录内容:", os.listdir(data_dir))
    
    preprocessor = AudioDataPreprocessor()
    audio_paths, labels = load_audio_samples(data_dir)
    print(f"找到 {len(audio_paths)} 个音频样本")
    
    X, y = preprocessor.prepare_dataset(audio_paths, labels)
    print(f"特征形状: {X.shape}, 标签形状: {y.shape}")
    
    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 创建模型
    num_classes = len(np.unique(y))
    model = create_instrument_classifier(X.shape[1:], num_classes)
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. 数据增强
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # 5. 训练模型
    print("开始训练...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # 6. 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试准确率: {test_acc:.4f}")
    
    # 7. 保存模型
    model_save_path = os.path.join(model_save_dir, "instrument_classifier.h5")
    label_encoder_save_path = os.path.join(model_save_dir, "label_encoder.pkl")
    
    model.save(model_save_path)
    import joblib
    joblib.dump(preprocessor.label_encoder, label_encoder_save_path)
    
    print(f"模型已保存到: {model_save_path}")
    print(f"标签编码器已保存到: {label_encoder_save_path}")
    
    return model, preprocessor.label_encoder, history

# 调用训练函数
print("开始调用训练函数...")
model, label_encoder, history = train_model()
print("训练完成！")
```

## 5 改进代码version1.0

### 5.1 google drive挂载

```py
from google.colab import drive
import zipfile
import os

# 挂载Google Drive
drive.mount('/content/drive')

# 假设你的IRMAS数据在Google Drive的这个路径
dataset_path = '/content/drive/MyDrive/4ATS/model/IRMAS-TrainingData.zip'

# 复制到Colab环境并解压
!cp "{dataset_path}" /content/
!unzip -q /content/IRMAS-TrainingData.zip -d /content/data/

# 定义模型保存路径
model_save_dir = "/content/drive/MyDrive/4ATS/model"
# 如果模型保存路径不存在，则创建它
os.makedirs(model_save_dir, exist_ok=True)
print(f"模型保存目录已准备就绪: {model_save_dir}")
```

### 5.2 环境配置

```py
# 4.2 配置环境 - 修改安装命令
# 安装库（修正TensorFlow的安装名称）
!pip install tensorflow
!pip install librosa
!pip install matplotlib
!pip install scikit-learn
!pip install joblib  # 添加缺失的库

# 导入库
import tensorflow as tf
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib  # 添加这个导入
import matplotlib.pyplot as plt  # 添加可视化

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", len(tf.config.list_physical_devices('GPU')) > 0)
```

### 5.3 数据预处理流程

```py
# 4.3 数据预处理流程 - 替换为增强版本
class EnhancedAudioDataPreprocessor:
    def __init__(self, target_sr=22050, duration=3):
        self.target_sr = target_sr
        self.duration = duration
        self.label_encoder = LabelEncoder()
    
    def extract_enhanced_features(self, audio_path):
        """提取增强的音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 确保音频长度一致
            y = librosa.util.fix_length(y, size=self.target_sr * self.duration)
            
            # 1. 主要特征：Mel频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=8000, n_fft=2048, hop_length=512
            )
            log_mel = librosa.power_to_db(mel_spec)
            
            # 2. 附加特征：MFCC（增加上下文信息）
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            
            # 3. 节奏特征
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 合并特征（简单版本：只使用log_mel，保持向后兼容）
            # 标准化
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            
            return log_mel
            
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def prepare_dataset(self, audio_paths, labels):
        """准备训练数据集"""
        features = []
        valid_labels = []
        
        print("正在提取特征...")
        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            if i % 500 == 0:  # 每500个样本显示进度
                print(f"已处理 {i}/{len(audio_paths)} 个样本")
                
            feature = self.extract_enhanced_features(path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(label)
        
        # 转换为numpy数组
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        
        # 添加通道维度（CNN需要）
        X = X[..., np.newaxis]
        
        print(f"最终数据集形状: {X.shape}")
        return X, y

# 更新数据加载函数以使用新的预处理器
preprocessor = EnhancedAudioDataPreprocessor()
```

### 5.4 定义类 & 模型构建

```py
# 3.2 数据预处理流程 - 定义类
class AudioDataPreprocessor:
    def __init__(self, target_sr=22050, duration=3):
        self.target_sr = target_sr
        self.duration = duration
        self.label_encoder = LabelEncoder()
    
    def extract_features(self, audio_path):
        """提取音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 确保音频长度一致
            if len(y) < self.target_sr * self.duration:
                y = np.pad(y, (0, self.target_sr * self.duration - len(y)))
            else:
                y = y[:self.target_sr * self.duration]
            
            # 提取Mel频谱图
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=8000
            )
            log_mel = librosa.power_to_db(mel_spectrogram)
            
            # 标准化
            log_mel = (log_mel - np.mean(log_mel)) / np.std(log_mel)
            
            return log_mel
            
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return None
    
    def prepare_dataset(self, audio_paths, labels):
        """准备训练数据集"""
        features = []
        valid_labels = []
        
        for path, label in zip(audio_paths, labels):
            feature = self.extract_features(path)
            if feature is not None:
                features.append(feature)
                valid_labels.append(label)
        
        # 转换为numpy数组
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        
        # 添加通道维度（CNN需要）
        X = X[..., np.newaxis]
        
        return X, y

# 数据加载函数
def load_audio_samples(data_dir):
    """加载音频样本和标签"""
    samples = []
    labels = []
    
    # 遍历每个乐器文件夹
    for instrument in os.listdir(data_dir):
        instrument_path = os.path.join(data_dir, instrument)
        if os.path.isdir(instrument_path):
            for audio_file in os.listdir(instrument_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(instrument_path, audio_file)
                    samples.append(audio_path)
                    labels.append(instrument)
    
    return samples, labels

# 4.3 模型构建 - 替换为改进版本
from tensorflow.keras import layers, models

def create_improved_classifier(input_shape, num_classes):
    """创建改进的乐器分类CNN模型"""
    
    model = models.Sequential([
        # 第一个卷积块 - 增加滤波器数量
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第二个卷积块
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三个卷积块
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 全局池化 + 全连接层
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

print("改进的模型构建完成")
```

### 5.5 模型训练流程

```py
# 4.4 训练流程 - 替换为改进版本
def train_improved_model():
    """改进的训练流程"""
    
    # 1. 准备数据
    data_dir = "/content/data/IRMAS-TrainingData/"
    print("数据目录内容:", os.listdir(data_dir))
    
    audio_paths, labels = load_audio_samples(data_dir)
    print(f"找到 {len(audio_paths)} 个音频样本")
    print("乐器类别:", set(labels))
    
    X, y = preprocessor.prepare_dataset(audio_paths, labels)
    print(f"特征形状: {X.shape}, 标签形状: {y.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    
    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. 创建改进的模型
    num_classes = len(np.unique(y))
    model = create_improved_classifier(X.shape[1:], num_classes)
    
    # 编译模型 - 使用更低的初始学习率
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. 增强的数据增强
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='constant'
    )
    
    # 5. 改进的回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=15, 
            restore_best_weights=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=8,
            min_lr=1e-7,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_save_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # 6. 训练模型
    print("开始训练改进模型...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=100,  # 增加epoch数量
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. 评估模型
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"最终测试准确率: {test_acc:.4f}")
    
    # 8. 保存模型和训练历史
    model_save_path = os.path.join(model_save_dir, "improved_instrument_classifier.h5")
    label_encoder_save_path = os.path.join(model_save_dir, "improved_label_encoder.pkl")
    history_save_path = os.path.join(model_save_dir, "training_history.npy")
    
    model.save(model_save_path)
    joblib.dump(preprocessor.label_encoder, label_encoder_save_path)
    np.save(history_save_path, history.history)
    
    print(f"改进模型已保存到: {model_save_path}")
    print(f"标签编码器已保存到: {label_encoder_save_path}")
    print(f"训练历史已保存到: {history_save_path}")
    
    # 9. 绘制训练曲线
    plot_training_history(history)
    
    return model, preprocessor.label_encoder, history

def plot_training_history(history):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'training_curves.png'))
    plt.show()

# 调用改进的训练函数
print("开始调用改进的训练函数...")
model, label_encoder, history = train_improved_model()
print("改进模型训练完成！")
```

### 5.6 性能分析

```py
# 在训练完成后添加性能分析
def analyze_model_performance(model, X_test, y_test, label_encoder):
    """分析模型性能"""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # 分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=label_encoder.classes_))
    
    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'confusion_matrix.png'))
    plt.show()
    
    # 计算每个类别的准确率
    class_accuracy = {}
    for i, class_name in enumerate(label_encoder.classes_):
        mask = y_test == i
        if np.sum(mask) > 0:
            acc = np.mean(y_pred_classes[mask] == y_test[mask])
            class_accuracy[class_name] = acc
            print(f"{class_name}: {acc:.3f}")

# 在训练完成后调用分析函数
analyze_model_performance(model, X_test, y_test, label_encoder)
```



