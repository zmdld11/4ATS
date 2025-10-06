# config/music_file_loader.py
"""
通用的音乐文件加载工具
用于从 music.txt 加载音乐文件列表
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class MusicFileLoader:
    """音乐文件加载器"""
    
    def __init__(self, music_list_file="music.txt", music_dir="music"):
        """
        初始化音乐文件加载器
        
        Args:
            music_list_file: 音乐列表文件名
            music_dir: 音乐文件夹名
        """
        self.music_list_file = os.path.join(project_root, music_list_file)
        self.music_dir = os.path.join(project_root, music_dir)
        
    def load_music_files(self):
        """从music.txt加载所有音乐文件路径"""
        if not os.path.exists(self.music_list_file):
            print(f"错误: 音乐列表文件不存在 - {self.music_list_file}")
            return []
        
        # 读取音乐文件列表
        try:
            with open(self.music_list_file, 'r', encoding='utf-8') as f:
                music_files = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"读取音乐列表文件失败: {e}")
            return []
        
        # 构建完整的文件路径并验证文件存在
        valid_files = []
        for music_file in music_files:
            full_path = os.path.join(self.music_dir, music_file)
            if os.path.exists(full_path):
                valid_files.append(full_path)
            else:
                print(f"警告: 音乐文件不存在 - {full_path}")
        
        print(f"找到 {len(valid_files)} 个有效的音乐文件")
        return valid_files
    
    def get_music_info(self, file_path):
        """获取音乐文件信息"""
        import librosa
        try:
            y, sr = librosa.load(file_path, sr=None)  # 不重采样以获取原始信息
            duration = len(y) / sr
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            return {
                'file_name': os.path.basename(file_path),
                'duration': duration,
                'sample_rate': sr,
                'file_size_mb': file_size,
                'channels': 1 if len(y.shape) == 1 else y.shape[0]
            }
        except Exception as e:
            print(f"获取音乐文件信息失败 {file_path}: {e}")
            return None
    
    def print_music_list(self):
        """打印音乐文件列表信息"""
        files = self.load_music_files()
        if not files:
            print("没有找到可用的音乐文件")
            return
        
        print("\n音乐文件列表:")
        print("-" * 80)
        print(f"{'文件名':<25} {'时长(s)':<10} {'采样率':<10} {'大小(MB)':<10}")
        print("-" * 80)
        
        for file_path in files:
            info = self.get_music_info(file_path)
            if info:
                print(f"{info['file_name']:<25} {info['duration']:<10.1f} {info['sample_rate']:<10} {info['file_size_mb']:<10.1f}")

# 创建全局实例
music_loader = MusicFileLoader()

# 便捷函数
def load_music_files():
    """加载所有音乐文件路径"""
    return music_loader.load_music_files()

def get_music_info(file_path):
    """获取音乐文件信息"""
    return music_loader.get_music_info(file_path)

def print_music_list():
    """打印音乐文件列表信息"""
    music_loader.print_music_list()

if __name__ == "__main__":
    # 测试功能
    print("音乐文件加载器测试:")
    print_music_list()