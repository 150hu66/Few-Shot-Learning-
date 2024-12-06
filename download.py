import torch
from torchvision import models
import os

def download_model(save_path='densenet121.pth'):
    """
    下载预训练的DenseNet121模型并保存到指定路径。
    """
    # 初始化预训练的DenseNet121模型
    model = models.densenet121(pretrained=True)
    model.eval()  # 设置为评估模式

    # 获取模型的state_dict（权重参数）
    state_dict = model.state_dict()

    # 保存state_dict到指定路径
    torch.save(state_dict, save_path)
    print(f"DenseNet121模型已保存到: {save_path}")

if __name__ == "__main__":
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, 'densenet121.pth')
    
    # 下载并保存模型
    download_model(model_save_path)
