# config.py (更新版本 - 中文注释)
import os

class Config:
    """项目配置类"""
    MODEL_VERSIONS = ['basic', 'simplified', 'advanced', 'stable_advanced']  # 支持的模型版本

    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # 数据集配置
    DATASET_NAME = "IRMAS-TrainingData"
    DATASET_URL = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip"
    
    # 音频处理配置
    TARGET_SAMPLE_RATE = 22050
    AUDIO_DURATION = 3  # 秒
    N_MELS = 128
    
    # 训练配置
    BATCH_SIZE = 32
    EPOCHS = 150  # 增加以获得更好的收敛
    LEARNING_RATE = 0.001  # 为AdamW调整
    VALIDATION_SPLIT = 0.2
    
    # 模型配置
    INPUT_SHAPE = (128, 130, 1)  # Mel频谱图形状
    
    # 高级训练配置
    USE_ADVANCED_MODEL = True  # 设置为False使用基础模型
    USE_DATA_AUGMENTATION = True
    EARLY_STOPPING_PATIENCE = 20
    
    # 多版本模型配置
    SAVE_MULTIPLE_VERSIONS = True  # 同时保存多个模型版本
    MODEL_VERSIONS = ['basic', 'simplified', 'advanced']  # 支持的模型版本
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [cls.DATA_DIR, cls.MODEL_DIR, cls.OUTPUT_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"目录已创建: {directory}")