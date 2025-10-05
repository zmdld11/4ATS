# AI音频分析与自动扒谱系统项目说明书

## 1. 项目概述

### 1.1 项目名称
AI音频分析与自动扒谱系统（AI Audio Analysis and Automatic Transcription System）

### 1.2 项目目标
开发一个能够自动分析音频内容，识别音乐元素，并生成标准乐谱的AI系统。

### 1.3 项目阶段
- **短期目标**：音频节奏检测和乐器种类识别
- **中期目标**：音源分离（乐器分离和人声提取）
- **长期目标**：音高检测和自动乐谱生成

## 2. 技术栈选择

### 2.1 核心编程语言
- Python 3.10

### 2.2 开发环境
- VS Code + Python扩展
- Conda环境管理
- Git版本控制

### 2.3 主要技术框架
```
音频处理：Librosa, PyAudio
机器学习：TensorFlow/PyTorch
音源分离：Demucs, Spleeter
音乐信息检索：Madmom, Essentia
乐谱生成：Music21, Abjad
可视化：Matplotlib, Plotly
```

## 3. 项目架构设计

### 3.1 系统模块划分
```
4ATS/
├── src/                    	# 代码
│   ├── main.py      			# 音频加载与预处理
│   └── xxx.py  	 			# ...
├── output/           			# 程序结果输出
│   ├── notation_generator.py 	# 乐谱生成
│   └── music_xml_exporter.py 	# MusicXML导出
├── env/                    	# 虚拟环境（环境名称env_310）
├── config/                 	# 配置环境
│   └── config.py            	# 配置文件
├── test/                   	# 测试文件
│   └── xxx.py 
├── data/                   	# 数据目录
|	└── IRMAS-TrainingData
├── model/                  	# 预训练模型
│   ├── model_advanced.pth
│   ├── model_advanced_label_encoder.pkl
│   ├── model_basic.pth
│   ├── model_basic_label_encoder.pkl
│   ├── model_simplified.pth
│   └── model_simplified_label_encoder.pkl
└── docs/                    	# 文档
```

## 4. 阶段详细实施计划

### 4.1 第一阶段：节奏与乐器识别（1-2个月）

#### 功能需求
- 支持常见音频格式（WAV, MP3, FLAC）
- BPM（每分钟节拍数）检测
- 节拍位置识别
- 乐器种类分类（钢琴、吉他、鼓、贝斯等）

#### 技术实现
```python
# 示例代码结构
class RhythmAnalyzer:
    def detect_bpm(self, audio_path):
        """检测音频BPM"""
        pass
    
    def find_beats(self, audio_path):
        """定位节拍位置"""
        pass

class InstrumentClassifier:
    def classify_instruments(self, audio_path):
        """识别音频中的乐器种类"""
        pass
```

#### 交付成果
- 节奏分析报告（BPM、节拍网格）
- 乐器识别结果及置信度
- 基础可视化界面

### 4.2 第二阶段：音源分离（2-3个月）

#### 功能需求
- 人声与伴奏分离
- 多乐器分离（鼓、贝斯、钢琴、其他）
- 分离质量评估

#### 技术实现
```python
class SourceSeparator:
    def separate_sources(self, audio_path, model_type='demucs'):
        """音源分离主函数"""
        pass
    
    def evaluate_separation(self, original, separated):
        """分离质量评估"""
        pass
```

#### 交付成果
- 分离后的各音轨音频文件
- 分离质量评估报告
- 音源分离API接口

### 4.3 第三阶段：音高检测与乐谱生成（3-4个月）

#### 功能需求
- 单音高检测
- 和弦识别
- 音符时长量化
- 标准乐谱生成（MusicXML格式）

#### 技术实现
```python
class PitchDetector:
    def detect_pitches(self, audio_path):
        """音高检测"""
        pass
    
    def quantize_notes(self, pitches):
        """音符量化"""
        pass

class NotationGenerator:
    def generate_sheet_music(self, notes_data):
        """生成乐谱"""
        pass
```

#### 交付成果
- 各音轨的音高序列
- MusicXML格式乐谱文件
- 可视化乐谱显示

## 5. 开发环境设置

### 5.1 Conda环境配置
```yaml
# environment.yml
name: ai-audio-transcriber
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - pip
  - numpy
  - scipy
  - matplotlib
  - jupyter
  - librosa
  - pytorch
  - torchaudio
  - tensorflow
  - pip:
    - demucs
    - spleeter
    - music21
    - essentia-tensorflow
```

### 5.2 安装步骤
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate ai-audio-transcriber

# 安装额外依赖
pip install -r requirements.txt
```

## 6. 测试计划

### 6.1 单元测试
- 各模块功能测试
- 边界条件测试
- 性能基准测试

### 6.2 集成测试
- 端到端流程测试
- 不同音乐类型兼容性测试
- 大规模数据集验证

## 7. 风险评估与应对策略

### 7.1 技术风险
- **音源分离质量不足**：准备多种分离算法备选
- **复杂音乐处理困难**：从简单音乐开始，逐步优化
- **计算资源需求大**：优化算法，支持GPU加速

### 7.2 数据风险
- **训练数据不足**：使用公开数据集+数据增强
- **标注质量不一**：建立严格的数据质量控制流程

## 8. 资料

### 8.1 IRMAS乐器缩写对照表

| 缩写 | 英文全称        | 中文名称 |
| :--- | :-------------- | :------- |
| cel  | Cello           | 大提琴   |
| cla  | Clarinet        | 单簧管   |
| flu  | Flute           | 长笛     |
| gac  | Acoustic Guitar | 原声吉他 |
| gel  | Electric Guitar | 电吉他   |
| org  | Organ           | 管风琴   |
| pia  | Piano           | 钢琴     |
| sax  | Saxophone       | 萨克斯管 |
| tru  | Trumpet         | 小号     |
| vio  | Violin          | 小提琴   |
| voi  | Human Voice     | 人声     |
