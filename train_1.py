# train_finetune_densenet121.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 定义数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并缩放到224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色抖动
    transforms.RandomRotation(10),  # 随机旋转
    #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 随机仿射变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                         std=[0.229, 0.224, 0.225])   # ImageNet标准差
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放到224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, device='cpu'):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()   # 设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅在训练阶段反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 计算epoch损失和准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        # 保存每轮的模型权重
        save_path = os.path.join(os.path.dirname(__file__), f'densenet121-better_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f'模型已保存到: {save_path}')

    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

def main():
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据路径
    data_dir = r'E:\Dashuju\train\train'  # 确保这是你的数据集根目录
    train_dir = data_dir  # 如果没有单独的验证集，直接使用 data_dir
    val_dir = data_dir  # 如果没有单独的验证集，直接使用 data_dir

    # 数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    }

    # 类别数
    num_classes = len(train_dataset.classes)
    print(f'类别数: {num_classes}')

    # 加载预训练的DenseNet121模型
    model = models.densenet121(pretrained=True)  # 使用预训练的DenseNet121

    # 替换最后的全连接层
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)  # 新的全连接层输出维度为类别数

    # 将模型转到设备
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练与微调
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5, device=device)

    # 保存最终模型权重
    final_save_path = os.path.join(os.path.dirname(__file__), 'densenet121-better_final.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f'最终模型已保存到: {final_save_path}')

if __name__ == '__main__':
    main()