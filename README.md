# 设置

我们的代码基于 stable-diffusion 构建，并且和 stable-diffusion 共享大部分依赖。另外，执行攻击操作需要使用 advertorch 0.2.4 版本。若要完成环境搭建，请运行如下命令：
```bash
conda env create -f environments.yml
conda activate mist
pip install --force-reinstall pillow

```
注意：PyPI 安装的 Pillow 工具包可能不完整，所以需要重新安装。强烈建议在首次激活 adrd 虚拟环境之后，重新安装 Pillow。
官方的 Stable-diffusion-model v1.4 检查点是必要的，你可以在 [huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4.ckpt) 上找到它。目前，可通过运行下面的命令来下载该模型：
```bash
wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mv sd-v1-4.ckpt models/ldm/models/sd-v1-4/sd-v1-4.ckpt
```
# 用法
mist.py文件将你准备好的数据集(扩散模型生成图片)进行寻找通用扰动并将扰动存储下来
示例
```bash
python mist.py --input_dir_path E:\adrd\datasets\train-40-ks --output_dir E:\adrd\datasets\raodong\train-40-ks   --save_path E:\adrd\data\ckpt\train1.pt
```
mist-jiazai.py将你寻找到的通用扰动加入到你的所有图片当中
示例
```bash
python mist-jiazai.py --delta_path E:\adrd\data\ckpt\train1.pt --base_dir E:/adrd/datasets --folders "ADM,DALLE2" --input_output_pairs "0-real,0-real-rd 1-fake,1-fake-rd"
```
## 训练svm分类器
### 训练+验证
```bash
python train-svm.py --dataset_folder "E:\adrd\datasets" --train_folder "DALLE2"--folders "ADM" --csv_file "results.csv" --svm_save_path "svm-train1.pkl" --checkpoint_path "E:\adrd\ldm\models\sd-v1-4\sd-v1-4.ckpt" --config_path "E:\adrd\configs\stable-diffusion\v1-inference.yaml" --force_retrain
```
### 仅验证
```bash
python train-svm.py --dataset_folder "E:\dire-h\datasets" --folders "ADM,DALLE2,Midjourney" --csv_file "results.csv" --svm_save_path "svm_train1.pkl" --checkpoint_path "E:\dire-h\ldm\models\sd-v1-4\sd-v1-4.ckpt" --config_path "E:\dire-h\configs\stable-diffusion\v1-inference.yaml"
```
# 输入转换下的鲁棒性
image transformation.py中提供了为图像添加高斯模糊和压缩质量的脚本，以评估其稳定性
```bash
python script.py --base_dir "E:\adrd\datasets" --folders "ADM DALLE2" --input_subfolders 0-real 1-fake --output_subfolders 0-real-gs7-5 0-fake-gs7-5 --n 1000 --processing_mode both --blur_kernel 7 7 --jpeg_quality 50
```
当然你也可以只加高斯模糊
```bash
python script.py --base_dir "E:\adrd\datasets" --folders "ADM DALLE2" --input_subfolders 0-real 1-fake --output_subfolders 0-real-gs7-5 0-fake-gs7-5 --n 1000 --processing_mode blur --blur_kernel 7 7 
```
