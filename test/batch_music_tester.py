import os
import sys
import torch
import argparse
import time

# 添加项目根目录到Python路径 :cite[1]:cite[2]
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 向上两级到项目根目录
sys.path.insert(0, project_root) # 确保Python在导入模块时能搜索到项目根目录 :cite[1]

from config.config import Config
# 注意：如果inst_model_test.py也在test文件夹中，可能需要调整导入方式
# 如果inst_model_test.py在src文件夹中，应使用：
from inst_model_test import MultiModelTester

class BatchMusicTester:
    def __init__(self, model_type='simplified'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        print(f"测试模型: {model_type}")
        
        # 创建输出目录
        self.batch_output_dir = os.path.join(Config.OUTPUT_DIR, "batch_test_results")
        os.makedirs(self.batch_output_dir, exist_ok=True)
    
    def load_music_list(self, music_list_file):
        """从music.txt加载音乐文件列表"""
        # 音乐列表文件现在位于项目根目录
        music_list_path = os.path.join(project_root, music_list_file) # 从项目根目录查找
        
        if not os.path.exists(music_list_path):
            print(f"错误: 音乐列表文件不存在 - {music_list_path}")
            return []
        
        with open(music_list_path, 'r', encoding='utf-8') as f:
            music_files = [line.strip() for line in f if line.strip()]
        
        # 构建完整的文件路径 - music文件夹在项目根目录
        music_dir = os.path.join(project_root, "music") # 从项目根目录查找music文件夹
        full_paths = []
        
        for music_file in music_files:
            full_path = os.path.join(music_dir, music_file)
            if os.path.exists(full_path):
                full_paths.append(full_path)
            else:
                print(f"警告: 音乐文件不存在 - {full_path}")
        
        print(f"找到 {len(full_paths)} 个有效的音乐文件")
        return full_paths
    
    def test_batch_music(self, music_list_file="music.txt"):
        """批量测试音乐文件"""
        music_files = self.load_music_list(music_list_file)
        
        if not music_files:
            print("没有找到可测试的音乐文件")
            return
        
        print(f"\n开始批量测试 {len(music_files)} 个音乐文件...")
        print("=" * 60)
        
        results = []
        
        for i, music_path in enumerate(music_files, 1):
            print(f"\n[{i}/{len(music_files)}] 测试: {os.path.basename(music_path)}")
            
            try:
                # 创建测试器
                tester = MultiModelTester(self.model_type)
                if tester.model is None:
                    print(f"  ❌ 模型加载失败，跳过")
                    continue
                
                # 测试单个文件
                start_time = time.time()
                predictions, features = tester.predict_single_audio(music_path)
                test_time = time.time() - start_time
                
                if predictions:
                    # 记录结果
                    top_prediction = predictions[0]
                    result = {
                        'file_name': os.path.basename(music_path),
                        'top_instrument': top_prediction['instrument'],
                        'top_confidence': top_prediction['probability'],
                        'test_time': test_time,
                        'all_predictions': predictions
                    }
                    results.append(result)
                    
                    # 输出结果
                    instrument_name = tester.get_english_name(top_prediction['instrument'])
                    print(f"  ✅ 最可能乐器: {instrument_name} (置信度: {top_prediction['probability']:.3f})")
                    print(f"  测试耗时: {test_time:.2f}秒")
                    
                    # 保存可视化结果（可选）
                    audio_name = os.path.splitext(os.path.basename(music_path))[0]
                    output_path = os.path.join(self.batch_output_dir, 
                                             f"batch_{audio_name}_{self.model_type}.png")
                    tester.visualize_prediction(predictions, features, audio_name)
                    
                else:
                    print(f"  ❌ 预测失败")
                    
            except Exception as e:
                print(f"  ❌ 测试过程中出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成批量测试报告
        self.generate_batch_report(results)
    
    def generate_batch_report(self, results):
        """生成批量测试报告"""
        if not results:
            print("\n没有有效的测试结果")
            return
        
        report_path = os.path.join(self.batch_output_dir, f"batch_test_report_{self.model_type}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("批量音乐测试报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试模型: {self.model_type}\n")
            f.write(f"测试文件数: {len(results)}\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'文件名':<20} {'最可能乐器':<15} {'置信度':<10} {'测试时间':<10}\n")
            f.write("-" * 80 + "\n")
            
            total_time = 0
            for result in results:
                instrument_name = self.get_english_name(result['top_instrument'])
                f.write(f"{result['file_name']:<20} {instrument_name:<15} {result['top_confidence']:<10.3f} {result['test_time']:<10.2f}\n")
                total_time += result['test_time']
            
            f.write(f"\n总测试时间: {total_time:.2f}秒\n")
            f.write(f"平均测试时间: {total_time/len(results):.2f}秒/文件\n")
        
        print(f"\n✅ 批量测试报告已保存: {report_path}")
        
        # 在控制台也输出总结
        print(f"\n批量测试总结:")
        print(f"总文件数: {len(results)}")
        print(f"总测试时间: {total_time:.2f}秒")
        print(f"平均测试时间: {total_time/len(results):.2f}秒/文件")
    
    def get_english_name(self, abbreviation):
        """获取乐器英文名称"""
        from src.instrument_mapper import InstrumentMapper
        return InstrumentMapper.get_english_name(abbreviation)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量音乐测试')
    parser.add_argument('--model-type', type=str, default='simplified', 
                       choices=['basic', 'simplified', 'advanced'],
                       help='使用的模型类型')
    parser.add_argument('--music-list', type=str, default='music.txt',
                       help='音乐列表文件路径')
    
    args = parser.parse_args()
    
    print("=== AI音频分析系统 - 批量音乐测试 ===")
    
    # 初始化配置
    Config.create_directories()
    
    # 创建批量测试器
    tester = BatchMusicTester(args.model_type)
    
    # 执行批量测试
    tester.test_batch_music(args.music_list)

if __name__ == "__main__":
    main()