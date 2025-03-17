import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from ldm.models.autoencoder import AutoencoderKL
import yaml
from tqdm import tqdm
import csv
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
import joblib


# 加载模型和配置
def load_autoencoder(checkpoint_path, config_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    ddconfig = model_config['params']['first_stage_config']['params']['ddconfig']
    lossconfig = model_config['params']['first_stage_config']['params']['lossconfig']
    embed_dim = model_config['params']['first_stage_config']['params']['embed_dim']

    autoencoder_sd = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("first_stage_model."):
            autoencoder_sd[k[len("first_stage_model."):]] = v

    model = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim)
    model.load_state_dict(autoencoder_sd, strict=True)
    return model.to("cuda")


# 数据集类
class ImagePairDataset(Dataset):
    def __init__(self, folder1, folder2, label, transform=None):
        self.folder1 = folder1
        self.folder2 = folder2
        self.label = label
        self.transform = transform
        self.files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if len(self.files1) != len(self.files2):
            raise ValueError(f"文件数量不一致: {len(self.files1)} vs {len(self.files2)}")

    def __len__(self):
        return len(self.files1)

    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.folder1, self.files1[idx])).convert('RGB')
        img2 = Image.open(os.path.join(self.folder2, self.files2[idx])).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, self.label


# 特征提取
def extract_distances(model, dataset_folder, folder_name, subdir_pairs, batch_size):
    """从指定文件夹提取distance特征"""
    real_folder1 = os.path.join(dataset_folder, folder_name, subdir_pairs['real'][0])
    real_folder2 = os.path.join(dataset_folder, folder_name, subdir_pairs['real'][1])
    fake_folder1 = os.path.join(dataset_folder, folder_name, subdir_pairs['fake'][0])
    fake_folder2 = os.path.join(dataset_folder, folder_name, subdir_pairs['fake'][1])

    dataset_real = ImagePairDataset(real_folder1, real_folder2, 0, transform)
    dataset_fake = ImagePairDataset(fake_folder1, fake_folder2, 1, transform)
    dataset = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    distances, labels = [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for img1, img2, label in tqdm(dataloader, desc=f'Extracting {folder_name}'):
            img1, img2 = img1.to(device), img2.to(device)
            latent1 = model.encode(img1).sample()
            latent2 = model.encode(img2).sample()
            distance = torch.norm(latent1 - latent2, p=2, dim=(1, 2, 3)).cpu().numpy()
            distances.extend(distance)
            labels.extend(label.numpy())

    return np.array(distances), np.array(labels)


# SVM训练
def train_svm(model, args):
    print("训练SVM分类器...")
    subdir_pairs = {
        'real': [args.train_real_subdir1, args.train_real_subdir2],
        'fake': [args.train_fake_subdir1, args.train_fake_subdir2]
    }
    distances, labels = extract_distances(model, args.dataset_folder, args.train_folder,
                                          subdir_pairs, args.batch_size)

    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(distances.reshape(-1, 1), labels)
    joblib.dump(svm, args.svm_save_path)
    print(f"模型保存至 {args.svm_save_path}")
    return svm


# SVM验证
def validate_svm(model, svm, args):
    results = []
    for folder in args.folders:
        try:
            subdir_pairs = {
                'real': [args.val_real_subdir1, args.val_real_subdir2],
                'fake': [args.val_fake_subdir1, args.val_fake_subdir2]
            }
            distances, labels = extract_distances(model, args.dataset_folder, folder,
                                                  subdir_pairs, args.batch_size)

            y_pred = svm.predict(distances.reshape(-1, 1))
            y_scores = svm.predict_proba(distances.reshape(-1, 1))[:, 1]

            acc = accuracy_score(labels, y_pred)
            ap = average_precision_score(labels, y_scores)
            r_acc = accuracy_score(labels[labels == 0], y_pred[labels == 0]) if sum(labels == 0) > 0 else 0
            f_acc = accuracy_score(labels[labels == 1], y_pred[labels == 1]) if sum(labels == 1) > 0 else 0

            results.append([folder, round(ap, 4), round(acc, 4), round(r_acc, 4), round(f_acc, 4)])
            print(f"{folder}: AP={ap:.4f}, ACC={acc:.4f}")
        except Exception as e:
            print(f"处理 {folder} 失败: {str(e)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM图像检测训练验证工具")

    # 必须参数
    parser.add_argument("--dataset_folder", required=True, help="数据集根目录")
    parser.add_argument("--folders", required=True, help="验证文件夹列表，逗号分隔")
    parser.add_argument("--csv_file", required=True, help="结果保存路径")
    parser.add_argument("--svm_save_path", required=True, help="SVM模型保存路径")
    parser.add_argument("--checkpoint_path", required=True, help="模型检查点路径")
    parser.add_argument("--config_path", required=True, help="配置文件路径")

    # 训练参数
    parser.add_argument("--train_folder", default="Midjourney", help="训练用文件夹名称")
    parser.add_argument("--train_real_subdir1", default="0-real", help="真实图片原始目录")
    parser.add_argument("--train_real_subdir2", default="0-real-rd", help="真实图片处理目录")
    parser.add_argument("--train_fake_subdir1", default="1-fake", help="伪造图片原始目录")
    parser.add_argument("--train_fake_subdir2", default="1-fake-rd", help="伪造图片处理目录")

    # 验证参数
    parser.add_argument("--val_real_subdir1", default="0-real", help="验证真实图片目录1")
    parser.add_argument("--val_real_subdir2", default="0-real-rd", help="验证真实图片目录2")
    parser.add_argument("--val_fake_subdir1", default="1-fake", help="验证伪造图片目录1")
    parser.add_argument("--val_fake_subdir2", default="1-fake-rd", help="验证伪造图片目录2")

    # 通用参数
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--force_retrain", action="store_true", help="强制重新训练模型")

    args = parser.parse_args()
    args.folders = args.folders.split(',')

    # 初始化模型
    model = load_autoencoder(args.checkpoint_path, args.config_path)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 训练/加载SVM
    if args.force_retrain or not os.path.exists(args.svm_save_path):
        print("开始训练SVM...")
        svm = train_svm(model, args)
    else:
        print("加载已有SVM模型...")
        svm = joblib.load(args.svm_save_path)

    # 执行验证
    print("开始验证...")
    results = validate_svm(model, svm, args)

    # 保存结果
    with open(args.csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'AP', 'ACC', 'Real_ACC', 'Fake_ACC'])
        writer.writerows(results)

    print("全部处理完成！")
