# src/utils/data_loader.py
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import sys

class DataManager:
    def __init__(self, expected_root=None):
        # 获取项目根目录 - 根据文件实际位置调整
        # 如果这个文件在 src/utils/ 下，那么 parent.parent 就是项目根目录
        self.project_root = Path(__file__).parent.parent
        
        # 检查项目根目录是否正确（可选）
        if expected_root:
            expected_path = Path(expected_root)
            if self.project_root != expected_path:
                print(f"错误: 项目根目录不匹配!")
                print(f"期望: {expected_path}")
                print(f"实际: {self.project_root}")
                print("请检查文件位置或调整expected_root参数")
                sys.exit(1)
        
        # 数据目录 - 根据您的需求选择 data 或 model
        self.data_dir = self.project_root / 'model'  # 改为data文件夹，或保持model
        self.setup_directories()
    
    def setup_directories(self):
        """创建数据目录结构"""
        directories = [
            self.data_dir,
            self.data_dir / 'raw',
            self.data_dir / 'processed', 
            self.data_dir / 'datasets',
            self.data_dir / 'temp'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ 目录已就绪: {directory}")
    
    def download_and_prepare_irmas(self):
        """下载和预处理IRMAS数据集"""
        dataset_url = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip"
        local_path = self.data_dir / "irmas.zip"
        extract_path = self.data_dir / "IRMAS-TrainingData"
        
        # 检查是否已经解压
        if extract_path.exists():
            print("数据集已存在，跳过下载...")
            return str(extract_path)
        
        # 下载（如果需要）
        if not local_path.exists():
            print("下载IRMAS数据集...")
            try:
                urllib.request.urlretrieve(dataset_url, local_path)
                print("下载完成!")
            except Exception as e:
                print(f"下载失败: {e}")
                return None
        
        # 解压
        print("解压数据集...")
        try:
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("解压完成!")
            
            # 可选：删除zip文件以节省空间
            # local_path.unlink()
            
        except Exception as e:
            print(f"解压失败: {e}")
            return None
        
        return str(extract_path)
    
    def load_audio_samples(self, data_dir=None):
        """加载音频样本和标签"""
        if data_dir is None:
            data_dir = self.data_dir / "IRMAS-TrainingData"
        
        data_path = Path(data_dir)
        
        # 检查数据集是否存在
        if not data_path.exists():
            print(f"错误: 数据集目录不存在: {data_path}")
            return [], []
        
        samples = []
        labels = []
        
        # 遍历每个乐器文件夹
        for instrument in os.listdir(data_path):
            instrument_path = data_path / instrument
            if instrument_path.is_dir():
                audio_files = list(instrument_path.glob('*.wav'))
                print(f"找到 {len(audio_files)} 个 {instrument} 样本")
                for audio_file in audio_files:
                    samples.append(str(audio_file))
                    labels.append(instrument)
        
        print(f"总共加载了 {len(samples)} 个音频样本")
        return samples, labels
    
    def get_dataset_path(self, dataset_name):
        """获取数据集路径"""
        return self.data_dir / "datasets" / dataset_name

# 使用示例
def main():
    # 可以传入期望的项目根目录进行检查（可选）
    dm = DataManager(expected_root=r"D:\program_project\4ATS")
    
    # 下载和准备数据集
    irmas_path = dm.download_and_prepare_irmas()
    if irmas_path:
        print(f"数据集路径: {irmas_path}")
        
        # 加载样本
        samples, labels = dm.load_audio_samples(irmas_path)
        if samples:
            print(f"加载了 {len(samples)} 个音频样本")
            print(f"乐器类别: {set(labels)}")
        else:
            print("没有找到音频样本")
    else:
        print("数据集准备失败")

if __name__ == "__main__":
    main()