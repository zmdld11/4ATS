# model_manager.py (新文件)
import os
import torch
import joblib
from config.config import Config

class ModelManager:
    """模型管理器 - 处理多版本模型的保存和加载"""
    
    def __init__(self):
        self.model_dir = Config.MODEL_DIR
    
    def get_model_filename(self, model_type, suffix="pth"):
        """获取模型文件名"""
        return f"model_{model_type}.{suffix}"
    
    def get_label_encoder_filename(self, model_type):
        """获取标签编码器文件名"""
        return f"model_{model_type}_label_encoder.pkl"
    
    def save_model(self, model, preprocessor, model_type, history=None, epoch=0):
        """保存指定类型的模型"""
        # 模型文件路径
        model_filename = self.get_model_filename(model_type)
        model_path = os.path.join(self.model_dir, model_filename)
        
        # 标签编码器文件路径
        encoder_filename = self.get_label_encoder_filename(model_type)
        encoder_path = os.path.join(self.model_dir, encoder_filename)
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': model.__class__.__name__,
            'model_type': model_type,
            'input_shape': getattr(model, 'input_shape', None),
            'num_classes': getattr(model, 'num_classes', None),
            'history': history or {},
            'epoch': epoch
        }, model_path)
        
        # 保存标签编码器
        joblib.dump(preprocessor.label_encoder, encoder_path)
        
        print(f"✅ {model_type}模型已保存到: {model_path}")
        print(f"✅ {model_type}标签编码器已保存到: {encoder_path}")
        
        return model_path, encoder_path
    
    def load_model(self, model, model_type, device):
        """加载指定类型的模型"""
        model_filename = self.get_model_filename(model_type)
        model_path = os.path.join(self.model_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"⚠️ {model_type}模型文件不存在: {model_path}")
            return False, None
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # 检查模型类型是否匹配
            saved_model_type = checkpoint.get('model_type', 'unknown')
            if saved_model_type != model_type:
                print(f"⚠️ 模型类型不匹配: 保存的是 '{saved_model_type}'，当前请求的是 '{model_type}'")
                return False, checkpoint
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ {model_type}模型加载成功")
            return True, checkpoint
            
        except Exception as e:
            print(f"❌ {model_type}模型加载失败: {e}")
            return False, None
    
    def load_label_encoder(self, model_type):
        """加载标签编码器"""
        encoder_filename = self.get_label_encoder_filename(model_type)
        encoder_path = os.path.join(self.model_dir, encoder_filename)
        
        if not os.path.exists(encoder_path):
            print(f"⚠️ {model_type}标签编码器文件不存在: {encoder_path}")
            return None
        
        try:
            label_encoder = joblib.load(encoder_path)
            print(f"✅ {model_type}标签编码器加载成功")
            return label_encoder
        except Exception as e:
            print(f"❌ {model_type}标签编码器加载失败: {e}")
            return None
    
    def get_available_models(self):
        """获取可用的模型列表"""
        available_models = []
        for model_type in Config.MODEL_VERSIONS:
            model_path = os.path.join(self.model_dir, self.get_model_filename(model_type))
            if os.path.exists(model_path):
                available_models.append(model_type)
        return available_models
    
    def delete_model(self, model_type):
        """删除指定类型的模型"""
        model_path = os.path.join(self.model_dir, self.get_model_filename(model_type))
        encoder_path = os.path.join(self.model_dir, self.get_label_encoder_filename(model_type))
        
        deleted = False
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"✅ 删除模型文件: {model_path}")
            deleted = True
        
        if os.path.exists(encoder_path):
            os.remove(encoder_path)
            print(f"✅ 删除标签编码器文件: {encoder_path}")
            deleted = True
        
        if not deleted:
            print(f"⚠️ 未找到{model_type}模型文件")
        
        return deleted

# 创建全局模型管理器实例
model_manager = ModelManager()