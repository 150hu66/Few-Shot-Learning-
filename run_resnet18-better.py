import os
import cv2
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# 定义特征提取模型
class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super(FeatureExtractor, self).__init__()
        # 加载预训练的ResNet18模型
        self.model = models.resnet18()
        # 修改全连接层以匹配权重文件
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 763)  # 将输出层调整为763个类别
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()
        # 去掉最后的全连接层
        self.features = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = x.view(x.size(0), -1)  # 展平
        return x


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = val_transform(image)
    return image.unsqueeze(0)  # 增加batch维度


def main(to_pred_dir, result_save_path):
    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)  # 当前文件夹路径

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载特征提取模型
    model_path = os.path.join(model_dir, 'resnet18-finetuned.pth')
    feature_extractor = FeatureExtractor(model_path)

    dirpath = os.path.abspath(to_pred_dir)
    filepath = os.path.join(dirpath, 'testA')  # 测试集A文件夹路径
    task_lst = os.listdir(filepath)

    res = ['img_name,label']  # 初始化结果文件，定义表头
    for task_name in task_lst:  # 循环task文件夹
        support_path = os.path.join(filepath, task_name, 'support')  # 支持集路径（文件夹名即为标签）
        query_path = os.path.join(filepath, task_name, 'query')  # 查询集路径（无标签，待预测图片）
        class_lst = [d for d in os.listdir(support_path) if os.path.isdir(os.path.join(support_path, d))]  # 过滤非目录

        # 计算每个类别的原型
        prototypes = {}
        for cls in class_lst:
            cls_path = os.path.join(support_path, cls)
            img_names = [name for name in os.listdir(cls_path) if name.endswith(('.png', '.jpg', '.jpeg'))]
            features = []
            for img_name in img_names:
                img_path = os.path.join(cls_path, img_name)
                img = load_image(img_path)
                feat = feature_extractor(img)  # 提取特征
                features.append(feat.numpy())
            # 计算原型（均值）
            prototypes[cls] = np.mean(np.vstack(features), axis=0)

        # 预测
        test_img_lst = [name for name in os.listdir(query_path) if name.endswith('.png')]
        for pathi in test_img_lst:
            name_img = os.path.join(query_path, pathi)
            img = load_image(name_img)
            feat = feature_extractor(img).numpy()  # 提取特征
            # 计算与每个原型的欧氏距离
            distances = {}
            for cls, proto in prototypes.items():
                distance = np.linalg.norm(feat - proto)
                distances[cls] = distance
            # 选择距离最小的类别作为预测
            pred_class = min(distances, key=distances.get)
            res.append(pathi + ',' + pred_class)

    # 将预测结果保存到result_save_path
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(res))


if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = os.path.dirname(os.path.abspath(__file__))  # 当前代码目录
    result_save_path = os.path.join(to_pred_dir, 'result.csv')  # 结果保存路径为当前目录下的 result.csv
    main(to_pred_dir, result_save_path)
'''
if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)'''