import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

# 数据路径
data_dir = r'E:\Dashuju\train\train'  # 确保这是你的数据集根目录

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
full_dataset = datasets.ImageFolder(data_dir, transform=transform)

# 获取类别和对应的索引
class_indices = full_dataset.class_to_idx
classes = list(class_indices.keys())
num_classes = len(classes)

# 设定随机种子以确保可重复性
random.seed(42)

def create_task(dataset, num_classes=10, num_train_per_class=5, num_test_samples=20):
    # 随机选择类别
    selected_classes = random.sample(classes, num_classes)
    
    train_indices = []
    
    for cls in selected_classes:
        # 获取当前类别的所有样本索引
        cls_indices = [i for i, (_, label) in enumerate(dataset) if label == class_indices[cls]]
        
        # 随机选择训练样本
        train_samples = random.sample(cls_indices, num_train_per_class)
        train_indices.extend(train_samples)
    
    # 从整个数据集中随机选择20张测试样本
    test_indices = random.sample(range(len(dataset)), num_test_samples)
    
    return Subset(dataset, train_indices), Subset(dataset, test_indices)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# 加载预训练的ResNet18模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)  # 不加载预训练权重
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))  # 替换最后的全连接层
model.load_state_dict(torch.load('resnet18-finetuned1127.pth'))  # 加载自定义的模型权重
model = model.to(device)

num_tasks = 1
accuracies = []

for _ in range(num_tasks):
    train_subset, test_subset = create_task(full_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # 评估模型
    accuracy = evaluate_model(model, test_loader)
    accuracies.append(accuracy)

# 计算最终准确率
final_accuracy = np.mean(accuracies)
print(f'最终准确率: {final_accuracy:.4f}')