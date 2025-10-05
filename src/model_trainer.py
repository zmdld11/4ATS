# model_trainer.py (优化版本 - 中文注释)
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from config.config import Config
from src.model_manager import model_manager

class AdvancedModelTrainer:
    """带有改进技术的PyTorch模型训练器"""
    
    def __init__(self, model, preprocessor, device, model_type='simplified'):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model_type = model_type  # 模型类型
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # 标签平滑以获得更好的泛化能力
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """使用混合精度训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if scheduler:
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=Config.EPOCHS, patience=20):
        """使用先进技术训练模型"""
        # 使用AdamW和权重衰减
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 带热重启的余弦退火
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # 第一次重启的迭代次数
            T_mult=2,  # 每次重启后T增加的因素
            eta_min=1e-6  # 最小学习率
        )
        
        print("开始使用先进技术训练模型...")
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            
            # 验证阶段
            val_loss, val_acc = self.validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            print(f'周期 [{epoch+1}/{epochs}] - {self.model_type}模型')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
            print(f'  学习率: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # 早停和模型保存
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model()
                print(f'  ✅ 保存最佳{self.model_type}模型，验证准确率: {val_acc:.4f}')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'早停: 验证准确率连续{patience}个周期未提升')
                break
        
        return self.history
    
    def evaluate(self, test_loader):
        """评估模型"""
        test_loss, test_acc = self.validate(test_loader)
        print(f"{self.model_type}模型最终测试准确率: {test_acc:.4f}")
        return test_loss, test_acc
    
    def save_model(self, model_name=None):
        """保存模型和相关文件"""
        if model_name is None:
            model_name = f"best_{self.model_type}"
        
        # 使用模型管理器保存
        model_path, encoder_path = model_manager.save_model(
            self.model, 
            self.preprocessor, 
            self.model_type,
            history=self.history,
            epoch=len(self.history['train_loss'])
        )
        
        return model_path, encoder_path

class ModelTrainer(AdvancedModelTrainer):
    """向后兼容"""
    pass