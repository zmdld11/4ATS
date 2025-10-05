# advanced_models.py (修复权重初始化)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.config import Config

class ResidualBlock(nn.Module):
    """带有批归一化的残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class AttentionModule(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AdvancedInstrumentClassifier(nn.Module):
    """带有残差连接和注意力机制的高级乐器分类器"""
    
    def __init__(self, input_shape, num_classes):
        super(AdvancedInstrumentClassifier, self).__init__()
        
        if len(input_shape) == 3:
            self.input_shape = input_shape
        elif len(input_shape) == 2:
            self.input_shape = (1, input_shape[0], input_shape[1])
        else:
            raise ValueError(f"不支持的输入形状: {input_shape}")
            
        self.num_classes = num_classes
        
        print(f"模型输入形状: {self.input_shape}")
        
        # 初始卷积层 - 修复：使用正确的输入通道数
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # 带有注意力的残差块
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.attention1 = AttentionModule(64)
        
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.attention2 = AttentionModule(128)
        
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.attention3 = AttentionModule(256)
        
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.attention4 = AttentionModule(512)
        
        # 全局池化和分类器
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 延迟权重初始化，在模型移动到设备后进行
        self.weights_initialized = False
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """创建残差层"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """正确初始化权重 - 修复版本"""
        print("初始化高级模型权重...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:  # 只有在偏置存在时才初始化
                    nn.init.constant_(m.bias, 0)
                    print(f"初始化卷积层偏置: {m.bias.shape}")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                print(f"初始化BN层: weight={m.weight.shape}, bias={m.bias.shape}")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # 只有在偏置存在时才初始化
                    nn.init.constant_(m.bias, 0)
                    print(f"初始化全连接层偏置: {m.bias.shape}")
        
        self.weights_initialized = True
        print("✅ 高级模型权重初始化完成")
    
    def forward(self, x):
        """前向传播"""
        # 延迟初始化权重（第一次前向传播时）
        if not self.weights_initialized:
            self._initialize_weights()
        
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_advanced_classifier(input_shape, num_classes):
    """创建高级乐器分类器"""
    return AdvancedInstrumentClassifier(input_shape, num_classes)

# 添加一个更稳定的简化高级模型
class StableAdvancedClassifier(nn.Module):
    """稳定的高级分类器 - 避免复杂的初始化问题"""
    
    def __init__(self, input_shape, num_classes):
        super(StableAdvancedClassifier, self).__init__()
        
        if len(input_shape) == 3:
            self.input_shape = input_shape
        elif len(input_shape) == 2:
            self.input_shape = (1, input_shape[0], input_shape[1])
        else:
            raise ValueError(f"不支持的输入形状: {input_shape}")
            
        self.num_classes = num_classes
        
        print(f"模型输入形状: {self.input_shape}")
        
        # 使用更简单的结构
        self.features = nn.Sequential(
            # 块1
            nn.Conv2d(self.input_shape[0], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 块2 - 残差块
            self._residual_block(64, 128, stride=2),
            
            # 块3 - 残差块
            self._residual_block(128, 256, stride=2),
            
            # 块4 - 残差块
            self._residual_block(256, 512, stride=2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _residual_block(self, in_channels, out_channels, stride=1):
        """简化的残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_stable_advanced_classifier(input_shape, num_classes):
    """创建稳定的高级分类器"""
    return StableAdvancedClassifier(input_shape, num_classes)

class SimplifiedAdvancedClassifier(nn.Module):
    """简化但有效的分类器"""
    
    def __init__(self, input_shape, num_classes):
        super(SimplifiedAdvancedClassifier, self).__init__()
        
        if len(input_shape) == 3:
            self.input_shape = input_shape
        elif len(input_shape) == 2:
            self.input_shape = (1, input_shape[0], input_shape[1])
        else:
            raise ValueError(f"不支持的输入形状: {input_shape}")
            
        self.num_classes = num_classes
        
        print(f"模型输入形状: {self.input_shape}")
        
        # 增强的CNN主干网络
        self.features = nn.Sequential(
            # 块1
            nn.Conv2d(self.input_shape[0], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 块2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 块3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 块4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_simplified_classifier(input_shape, num_classes):
    """创建简化但有效的分类器"""
    return SimplifiedAdvancedClassifier(input_shape, num_classes)