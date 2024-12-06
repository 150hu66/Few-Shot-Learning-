import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# 定义数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并缩放到224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放到224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 定义Mixup数据增强函数
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, device='cpu'):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()   # 设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # 计算损失
                    if phase == 'train':
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
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

            # 学习率调度器步进
            if phase == 'train':
                scheduler.step()

        # 在第四轮结束后调整学习率
        if epoch == 4:  # 第四轮（索引从0开始）
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5  # 设置新的学习率

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

    # 数据集加载
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    # 数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    }

    # 类别数
    num_classes = len(train_dataset.classes)
    print(f'类别数: {num_classes}')

    # 加载预训练的ResNet18模型
    model = models.resnet18(weights='IMAGENET1K_V1')  # 使用预训练模型
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # 替换为适合763个类别的全连接层

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 使用Label Smoothing
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 学习率调度器（使用余弦退火学习率调度器，包含学习率预热）
    scheduler = CosineAnnealingLR(optimizer, T_max=6, eta_min=1e-6)

    # 训练与微调
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, device=device)

    # 保存模型权重
    save_path = os.path.join(os.path.dirname(__file__), 'resnet18-finetuned.pth')
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存到: {save_path}')

if __name__ == '__main__':
    main()