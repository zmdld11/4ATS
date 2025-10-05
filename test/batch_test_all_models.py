import os
import sys
import subprocess
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import Config

def run_test(test_script, model_type, audio_path=None):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"运行 {test_script} - 模型: {model_type}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, test_script, '--model-type', model_type]
    if audio_path:
        cmd.extend(['--audio-path', audio_path])
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
        return False

def main():
    """批量测试所有模型"""
    parser = argparse.ArgumentParser(description='批量测试所有模型')
    parser.add_argument('--audio-path', type=str, 
                       default=os.path.join(project_root, "music", "3.flac"),
                       help='测试音频路径')
    
    args = parser.parse_args()
    
    print("=== AI音频分析与自动扒谱系统 - 批量模型测试 ===")
    
    # 1. 初始化配置
    Config.create_directories()
    
    # 2. 检查音频文件
    if not os.path.exists(args.audio_path):
        print(f"错误: 测试音频文件不存在 - {args.audio_path}")
        return
    
    model_types = ['basic', 'simplified', 'advanced']
    test_scripts = [
        os.path.join(project_root, "test", "inst_model_test.py"),
        os.path.join(project_root, "test", "model_evaluation.py"),
        os.path.join(project_root, "test", "test_timeline.py")
    ]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n🎯 测试 {model_type} 模型")
        print("-" * 40)
        
        model_results = {}
        for test_script in test_scripts:
            if os.path.exists(test_script):
                test_name = os.path.basename(test_script)
                success = run_test(test_script, model_type, args.audio_path)
                model_results[test_name] = success
            else:
                print(f"测试脚本不存在: {test_script}")
        
        results[model_type] = model_results
    
    # 输出测试总结
    print(f"\n{'='*60}")
    print("批量测试总结")
    print(f"{'='*60}")
    
    for model_type, model_results in results.items():
        print(f"\n{model_type}模型:")
        total_tests = len(model_results)
        passed_tests = sum(1 for success in model_results.values() if success)
        print(f"  通过测试: {passed_tests}/{total_tests}")
        
        for test_name, success in model_results.items():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"    {test_name}: {status}")

if __name__ == "__main__":
    main()