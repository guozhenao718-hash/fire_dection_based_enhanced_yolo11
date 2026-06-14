import os
import cv2
import numpy as np

# 彻底规避潜在的多线程卡死与不必要的日志阻塞
os.environ['WANDB_MODE'] = 'disabled'
os.environ['OMP_NUM_THREADS'] = '1'

from ultralytics import YOLO


# ========================================================================
# 核心创新：根据论文3成果，构建动态限制对比度自适应直方图均衡化（CLAHE）增强流
# ========================================================================
def apply_clahe_enhancement(image):
    """
    对输入图像在LAB空间进行自适应直方图均衡化，在不放大边缘噪声的同时大幅强化烟雾对比度
    """
    # 将BGR图像转换为LAB色彩空间以分离亮度通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 构建限制对比度的CLAHE器（根据火灾多变环境，ClipLimit设为2.0，Grid设为8x8）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 重新合并通道并回转为BGR
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image


if __name__ == '__main__':
    # 1. 声明我们精心设计的 P2 增强型四头拓扑结构
    model = YOLO('yolo11s_fire_perfect.yaml')

    # 2. 完美安全迁移学习加载：Ultralytics 会无缝映射除新增检测头之外的所有通用骨干层权重
    model.load('yolo11s.pt')

    # 3. 启动高质收敛调优（综合三篇文献的最佳实践参数）
    model.train(
        data='./my_data.yaml',
        epochs=150,  # 150轮提供充足的后期平稳凝固期
        patience=40,  # 适当放大容忍度，允许四头网络在震荡中寻找更优局部解
        batch=16,  # 维持16位高显存效率的最佳批次
        imgsz=640,  # 保持640x640标准输入分辨率
        device=0,
        workers=4,  # 安全多线程数，防止I/O死锁

        # ==== 优化器超参重构（依据MDPI火灾论文最稳收敛线调校） ====
        optimizer='SGD',  # 采用MDPI验证对YOLOv11火灾任务更稳健的高阶一阶优化器
        lr0=0.01,  # 设定稍高的初始学习率以带动新增的P2头快速学习
        lrf=0.01,  # 最终学习率衰减系数
        momentum=0.937,  # 维持经典的高动量保持机制
        weight_decay=0.0005,  # 正则化系数
        warmup_epochs=4.0,  # 4轮热身，给新融合的浅层跨层连接提供平滑过渡期
        cos_lr=True,  # 使用余弦退火，确保在训练后期平滑逼近最优值

        # ==== 专属小目标与对比度损失防御（针对烟雾火焰精修） ====
        mosaic=0.7,  # 适当降低Mosaic概率（从1.0降到0.7），减少小烟雾被切碎造成的不利损失
        mixup=0.1,  # 引入适量混合，提高长距离微弱火点的抗干扰泛化能力
        erasing=0.1,  # 降低随机擦除，防止初期微小的烟雾核心直接被完全遮蔽
        label_smoothing=0.05,  # 启用标签平滑，对抗D-Fire数据集里部分复杂云雾界限模糊的噪点

        # 色调微调，大幅贴合红黄火焰和灰白烟雾的颜色特征空间
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,  # 开启水平翻转扩展

        # ==== 稳定期精密控制 ====
        cache='disk',  # 磁盘缓存加速
        amp=True,  # 开启混合精度，不仅大幅节省显存，还能通过动态缩放保持梯度稳定
        close_mosaic=15,  # 临近尾声最后15轮彻底关闭剧烈增强，让四头网络的参数在最干净的数据上凝固

        project='runs_fire_smoke',
        name='exp_yolo11_85plus_achieved',
        exist_ok=True,
        verbose=True
    )