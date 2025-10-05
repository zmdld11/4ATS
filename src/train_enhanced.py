# train_enhanced.py (简化版本 - 中文注释)
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # 直接运行主程序，使用默认参数
    from main import main
    main()