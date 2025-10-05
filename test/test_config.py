import os

class TestConfig:
    """测试配置"""
    
    # 测试音频路径
    TEST_AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "music")
    
    # 模型路径
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
    
    # 输出路径
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "test_results")
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        print(f"测试输出目录: {cls.OUTPUT_DIR}")