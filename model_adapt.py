#1\model 2\data 3\loss 4\eval 
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 设定数据集路径和设备
def main(args):

    data_dir = args.data_dir
    # data_dir = '/data/majc/ImageNet-C/fog/5'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    # 这里使用了标准的ImageNet预处理流程
    transform = transforms.Compose([
        transforms.Resize(256),  # 根据模型要求调整大小
        transforms.CenterCrop(224),  # 中心裁剪以匹配模型输入大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.resnet50(pretrained=True)
    out_num = model.fc.in_features  # 输出维度
    model.fc = torch.nn.Linear(out_num, 200)
    model = model.to(device)
    model.eval()

    # # 加载预训练的分类器
    # # 使用ImageNet1K预训练的ResNet-50
    # model_path = '/data/majc/models/DDA/DDA_ckpt/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
    # model = models.resnet50(pretrained=False)  # 关闭自动加载预训练权重
    # # 加载权重到模型中
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model = model.to(device)
    # model.eval()


    # 计算分类准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing images", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'分类准确率: {accuracy:.3f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--data_dir', default='/data/majc/imagenet-s', type=str)
    args = parser.parse_args()

    main(args)  