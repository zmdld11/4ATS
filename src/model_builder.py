import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config

class ImprovedInstrumentClassifier(nn.Module):
    """改进的乐器分类CNN模型 - PyTorch版本"""
    
    def __init__(self, input_shape, num_classes):
        super(ImprovedInstrumentClassifier, self).__init__()
        
        # 确保输入形状是 (channels, height, width) 格式
        if len(input_shape) == 3:
            self.input_shape = input_shape  # (channels, height, width)
        elif len(input_shape) == 2:
            self.input_shape = (1, input_shape[0], input_shape[1])  # 添加channel维度
        else:
            raise ValueError(f"不支持的输入形状: {input_shape}")
            
        self.num_classes = num_classes
        
        print(f"模型输入形状: {self.input_shape}")
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 64, 3, padding=1),  # 使用正确的输入通道数
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # 计算卷积后的特征图尺寸
        with torch.no_grad():
            self.feature_size = self._get_conv_output(self.input_shape)
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _get_conv_output(self, shape):
        """计算卷积层输出尺寸"""
        batch_size = 1
        # 创建测试输入 (batch_size, channels, height, width)
        input_tensor = torch.rand(batch_size, *shape)
        output = self.conv1(input_tensor)
        output = self.conv2(output)
        output = self.conv3(output)
        return output.view(batch_size, -1).size(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

def create_improved_classifier(input_shape, num_classes):
    """创建改进的乐器分类器"""
    return ImprovedInstrumentClassifier(input_shape, num_classes)