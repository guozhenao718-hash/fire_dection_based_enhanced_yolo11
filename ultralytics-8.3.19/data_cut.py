import os
import random
import shutil


def split_train_val(base_dir, val_ratio=0.2):
    """
    base_dir: D-Fire数据集的根目录
    val_ratio: 划分为验证集的比例，0.2 代表 20% 的数据作为验证集
    """
    # 1. 定义源路径 (train 的图片和标签)
    train_img_dir = os.path.join(base_dir, 'train', 'images')
    train_txt_dir = os.path.join(base_dir, 'train', 'labels')

    # 2. 定义目标路径 (val 的图片和标签)
    val_img_dir = os.path.join(base_dir, 'val', 'images')
    val_txt_dir = os.path.join(base_dir, 'val', 'labels')

    # 自动创建 val 文件夹及其子文件夹
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_txt_dir, exist_ok=True)

    # 3. 读取所有的训练集图片文件名
    if not os.path.exists(train_img_dir):
        print(f"❌ 错误：找不到训练集图片目录 {train_img_dir}，请检查路径！")
        return

    all_images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_count = len(all_images)
    print(f"📂 检查到当前 train 集中共有图片: {total_count} 张")

    if total_count == 0:
        print("❌ 错误：train/images 文件夹中没有图片！")
        return

    # 4. 随机洗牌并计算需要划分到 val 的数量
    random.seed(42)  # 设置随机种子，确保实验可复现
    random.shuffle(all_images)
    val_count = int(total_count * val_ratio)
    val_images_set = all_images[:val_count]
    print(f"📊 按照 {val_ratio * 100}% 的比例，将随机抽取 {val_count} 张图片及其标签移至 val 验证集...")

    # 5. 开始同步移动图片和对应的标签文件
    moved_count = 0
    for img_name in val_images_set:
        # 获取文件名（不带后缀）
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + '.txt'

        # 定义图片源路径和目标路径
        src_img = os.path.join(train_img_dir, img_name)
        dst_img = os.path.join(val_img_dir, img_name)

        # 定义标签源路径和目标路径
        src_txt = os.path.join(train_txt_dir, txt_name)
        dst_txt = os.path.join(val_txt_dir, txt_name)

        # 移动图片
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

            # 同步移动对应的标签 txt 文件（有些图片可能没有目标，没有txt，所以加个判断）
            if os.path.exists(src_txt):
                shutil.move(src_txt, dst_txt)

            moved_count += 1

    print("\n" + "=" * 40)
    print(" 🎉 划分任务顺利完成！")
    print("=" * 40)
    print(f" 🟩 剩余训练集 (train) 图片数: {len(os.listdir(train_img_dir))} 张")
    print(f" 🟦 新生成验证集 (val) 图片数: {len(os.listdir(val_img_dir))} 张")
    print("=" * 40)


if __name__ == "__main__":
    # 根据你的截图，直接锁定你的绝对路径
    DATASET_PATH = "E:/project/fire_detection/ultralytics-8.3.19/D-Fire"
    split_train_val(DATASET_PATH, val_ratio=0.2)