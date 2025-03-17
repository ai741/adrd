import os
import random
import cv2
import argparse


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """应用高斯模糊"""
    return cv2.GaussianBlur(image, kernel_size, 0)


def save_compressed(image, output_path, quality=30):
    """保存为压缩的JPEG格式"""
    cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def process_dataset(base_dir, folders, input_subfolders, output_subfolders,
                    n, processing_mode, blur_kernel, jpeg_quality):
    """处理数据集的主函数"""
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPEG', '.PNG', '.JPG']

    for folder in folders:
        for input_subfolder, output_subfolder in zip(input_subfolders, output_subfolders):
            # 输入文件夹路径
            input_folder = os.path.join(base_dir, folder, input_subfolder)

            if not os.path.exists(input_folder):
                print(f"跳过不存在的输入文件夹: {input_folder}")
                continue

            # 输出文件夹路径
            output_folder = os.path.join(base_dir, folder, output_subfolder)
            os.makedirs(output_folder, exist_ok=True)

            # 获取所有图片文件
            all_images = [f for f in os.listdir(input_folder)
                        if os.path.splitext(f)[1] in image_extensions]

            # 处理数量不足的情况
            if len(all_images) < n:
                print(f"警告: {input_folder} 只有 {len(all_images)} 张图片")
                selected = all_images
            else:
                selected = random.sample(all_images, n)

            for filename in selected:
                input_path = os.path.join(input_folder, filename)
                image = cv2.imread(input_path)

                if image is None:
                    print(f"无法读取图片: {input_path}")
                    continue

                processed = image.copy()

                if processing_mode in ['blur', 'both']:
                    processed = apply_gaussian_blur(processed, blur_kernel)

                output_path = os.path.join(output_folder, filename)

                if processing_mode in ['compress', 'both']:
                    save_compressed(processed, output_path, jpeg_quality)
                else:
                    cv2.imwrite(output_path, processed)

                print(f"已处理: {input_path} -> {output_path}")


def main():
    # 配置命令行参数解析
    parser = argparse.ArgumentParser(description='图像处理参数设置')
    parser.add_argument('--base_dir', type=str, required=True,
                      help='数据集根目录路径')
    parser.add_argument('--folders', type=str, required=True,
                      help='要处理的文件夹列表，用逗号分隔')
    parser.add_argument('--input_subfolders', type=str, default='0_real,1_fake',
                      help='输入子文件夹列表，用逗号分隔')
    parser.add_argument('--output_subfolders', type=str, default='3_real-gs5-3,3_fake-gs5-3',
                      help='输出子文件夹列表，用逗号分隔')
    parser.add_argument('--n', type=int, default=1500,
                      help='需要选取的图片数量')
    parser.add_argument('--processing_mode', type=str, default='blur',
                      choices=['blur', 'compress', 'both'],
                      help='处理模式: blur|compress|both')
    parser.add_argument('--blur_kernel', type=int, nargs=2, default=[5, 5],
                      help='高斯模糊核大小，两个整数用空格分隔')
    parser.add_argument('--jpeg_quality', type=int, default=30,
                      help='JPEG压缩质量 (0-100)')

    args = parser.parse_args()

    # 转换参数格式
    folders = args.folders.split(',')
    input_subfolders = args.input_subfolders.split(',')
    output_subfolders = args.output_subfolders.split(',')
    blur_kernel = tuple(args.blur_kernel)

    # 调用处理函数
    process_dataset(
        base_dir=args.base_dir,
        folders=folders,
        input_subfolders=input_subfolders,
        output_subfolders=output_subfolders,
        n=args.n,
        processing_mode=args.processing_mode,
        blur_kernel=blur_kernel,
        jpeg_quality=args.jpeg_quality
    )


if __name__ == '__main__':
    main()