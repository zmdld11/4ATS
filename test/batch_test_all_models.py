import os
import sys
import torch
import argparse
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import Config
from config.music_file_loader import load_music_files
from inst_model_test import MultiModelTester

class BatchAllModelsTester:
    """批量测试所有模型"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_types = ['basic', 'simplified', 'advanced']
        
        # 创建输出目录
        self.batch_output_dir = os.path.join(Config.OUTPUT_DIR, "batch_all_models_results")
        os.makedirs(self.batch_output_dir, exist_ok=True)
    
    def test_all_models_on_music(self):
        """在所有音乐文件上测试所有模型"""
        # 加载音乐文件
        music_files = load_music_files()
        if not music_files:
            print("没有找到可测试的音乐文件")
            return
        
        print(f"\n开始批量测试 {len(music_files)} 个音乐文件，使用 {len(self.model_types)} 个模型...")
        print("=" * 70)
        
        all_results = {}
        
        for model_type in self.model_types:
            print(f"\n🔧 测试 {model_type} 模型:")
            print("-" * 50)
            
            model_results = []
            for i, music_path in enumerate(music_files, 1):
                print(f"  [{i}/{len(music_files)}] {os.path.basename(music_path)}")
                
                try:
                    tester = MultiModelTester(model_type)
                    if tester.model is None:
                        print(f"    ❌ 模型加载失败，跳过")
                        continue
                    
                    start_time = time.time()
                    predictions, _ = tester.predict_single_audio(music_path)
                    test_time = time.time() - start_time
                    
                    if predictions:
                        top_pred = predictions[0]
                        result = {
                            'file_name': os.path.basename(music_path),
                            'top_instrument': top_pred['instrument'],
                            'top_confidence': top_pred['probability'],
                            'test_time': test_time
                        }
                        model_results.append(result)
                        
                        instrument_name = tester.get_english_name(top_pred['instrument'])
                        print(f"    ✅ {instrument_name} (置信度: {top_pred['probability']:.3f}, 耗时: {test_time:.2f}s)")
                    else:
                        print(f"    ❌ 预测失败")
                        
                except Exception as e:
                    print(f"    ❌ 测试过程中出错: {e}")
            
            all_results[model_type] = model_results
        
        # 生成报告
        self.generate_comprehensive_report(all_results, music_files)
    
    def generate_comprehensive_report(self, all_results, music_files):
        """生成综合报告"""
        report_path = os.path.join(self.batch_output_dir, "batch_all_models_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("多模型批量测试报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试文件数: {len(music_files)}\n")
            f.write(f"测试模型: {', '.join(self.model_types)}\n\n")
            
            # 各模型结果
            for model_type, results in all_results.items():
                f.write(f"{model_type.upper()} 模型结果:\n")
                f.write("-" * 50 + "\n")
                
                if results:
                    for result in results:
                        f.write(f"  {result['file_name']}: {result['top_instrument']} (置信度: {result['top_confidence']:.3f})\n")
                else:
                    f.write("  无有效结果\n")
                f.write("\n")
            
            # 统计信息
            f.write("统计信息:\n")
            f.write("-" * 30 + "\n")
            for model_type, results in all_results.items():
                f.write(f"{model_type}: {len(results)}/{len(music_files)} 文件测试成功\n")
        
        print(f"\n✅ 综合测试报告已保存: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量测试所有模型')
    
    args = parser.parse_args()
    
    print("=== AI音频分析系统 - 多模型批量测试 ===")
    
    # 初始化配置
    Config.create_directories()
    
    # 创建测试器并执行测试
    tester = BatchAllModelsTester()
    tester.test_all_models_on_music()

if __name__ == "__main__":
    main()