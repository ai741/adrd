import os
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np


def parse_input_output_pairs(pair_str):
    """将输入输出对字符串解析为元组列表"""
    pairs = []
    for pair in pair_str.split():
        if ',' not in pair:
            raise ValueError(f"无效的输入输出对格式: {pair}，应使用逗号分隔")
        input_dir, output_dir = pair.split(',', 1)
        pairs.append((input_dir.strip(), output_dir.strip()))
    return pairs


def process_images(delta_path, base_dir, folders, input_output_pairs, image_extensions):
    # 加载并验证扰动张量
    delta = torch.load(delta_path).detach()
    assert delta.shape == (1, 3, 256, 256), f"扰动张量形状应为 [1,3,256,256]，当前为 {delta.shape}"

    # 图像预处理流程
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for folder_name in folders:
        for input_subdir, output_subdir in input_output_pairs:
            # 构建路径
            input_folder = os.path.join(base_dir, folder_name, input_subdir)
            output_folder = os.path.join(base_dir, folder_name, output_subdir)

            if not os.path.exists(input_folder):
                print(f"警告: {input_folder} 不存在，已跳过")
                continue

            os.makedirs(output_folder, exist_ok=True)

            # 处理图片
            for filename in os.listdir(input_folder):
                if not any(filename.lower().endswith(ext) for ext in image_extensions):
                    continue

                img_path = os.path.join(input_folder, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = trans(img).unsqueeze(0)

                    # 添加扰动
                    adv_tensor = torch.clamp(img_tensor + delta, -1.0, 1.0)

                    # 转换并保存
                    adv_array = (adv_tensor.squeeze(0).permute(1, 2, 0).numpy() * 127.5 + 127.5)
                    adv_array = np.clip(adv_array, 0, 255).astype(np.uint8)

                    Image.fromarray(adv_array).save(
                        os.path.join(output_folder, filename))

                except Exception as e:
                    print(f"处理 {img_path} 失败: {str(e)}")

            print(f"已完成 {folder_name}/{input_subdir} -> {output_subdir}")


def main():
    parser = argparse.ArgumentParser(description="图像扰动添加工具")

    # 必须参数
    parser.add_argument("--delta_path", required=True,
                        help="扰动张量文件路径（.pt 文件）")
    parser.add_argument("--base_dir", required=True,
                        help="数据集根目录路径")
    parser.add_argument("--folders", required=True,
                        help="要处理的父文件夹列表，逗号分隔")
    parser.add_argument("--input_output_pairs", required=True,
                        help="输入输出对列表，格式：'输入1,输出1 输入2,输出2'")

    # 可选参数
    parser.add_argument("--image_extensions", default=".jpg,.jpeg,.png,.JPEG,.PNG,.JPG",
                        help="支持的图片扩展名，逗号分隔（默认：.jpg,.jpeg,.png,.JPEG,.PNG,.JPG）")

    args = parser.parse_args()

    # 参数转换
    folders = [f.strip() for f in args.folders.split(',')]
    image_extensions = [ext.strip() for ext in args.image_extensions.split(',')]
    input_output_pairs = parse_input_output_pairs(args.input_output_pairs)

    process_images(
        delta_path=args.delta_path,
        base_dir=args.base_dir,
        folders=folders,
        input_output_pairs=input_output_pairs,
        image_extensions=image_extensions
    )


if __name__ == "__main__":
    main()

