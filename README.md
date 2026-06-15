# 🔥 Fire Detection 项目  
基于深度学习的火灾检测系统，支持图片中的火焰与烟雾识别。  
## 📂 项目结构  
fire_detection_based_enhanced_yolo11/          # 项目根目录  
│  
├── ultralytics-8.3.19/                        # 基于 YOLOv11 的核心代码库  
│   ├── .github/                               # GitHub 相关配置（Issue 模板、CI 等）  
│   ├── .idea/                                 # IDE 配置文件（忽略上传）  
│   │
│   ├── backend/                               # 后端服务模块（Flask/FastAPI）  
│   │   ├── outputs/                           # 检测结果输出目录  
│   │   ├── static/                            # 静态文件（CSS、JS、图片）  
│   │   ├── uploads/                           # 用户上传的待检测图片  
│   │   ├── app.py                             # 后端主程序（启动 API 服务）  
│   │   └── fire_record.db                     # SQLite 数据库（记录检测历史）  
│   │
│   ├── D-Fire/                                # D-Fire 数据集（数据集地址）  
│   ├── test/                                  # 测试集  
│   ├── train/                                 # 训练集  
│   ├── val/                                   # 验证集  
│   │
│   ├── docker/                                # Docker 容器化配置文件  
│   ├── docs/                                  # 项目文档  
│   │  
│   ├── frontend/                              # 前端界面模块    
│   │   ├── css/                               # 样式表文件    
│   │   ├── img/                               # 前端图片资源    
│   │   ├── js/                                # 前端 JavaScript 脚本  
│   │   ├── unpackage/                         # 打包或未压缩的临时文件  
│   │   ├── history.html                       # 检测历史记录页面  
│   │   └── index.html                         # 主页面（实时检测界面）  
│   │  
│   └── ...（其他 YOLO 原始文件如 models, utils 等）  # 未完整列出  
│   
├── yolo11s_fire_perfect.yaml                  # 自定义的 P2 增强型四头拓扑结构模型配置  
├── yolo11s.py                                 # YOLOv11s 预训练权重文件（用于迁移学习）  
├── my_data.yaml                               # 数据集配置文件（路径、类别等）  
│  
├── train.py                                   # 训练入口脚本  
├── requirements.txt                           # Python 依赖清单  
├── environment.yml                            # Conda 完整环境导出  
├── README.md                                  # 项目说明文档  
└── LICENSE                                    # 开源许可证（Apache 2.0）    
  
## 📦 快速开始  

### 1. 克隆项目     
git clone https://github.com/guozhenao718-hash/fire_dection_based_enhanced_yolo11.git      
cd fire_dection_based_enhanced_yolo11  
2. 创建虚拟环境并安装依赖    
conda create -n fire-env python=3.8 -y    
conda activate fire-env    
pip install -r requirements.txt  
 3. 下载并划分数据集  
数据集存放在 D-Fire/ 目录下，请自行下载并放置到对应位置。  
python data_cut.py  
4. 运行推理  
python infer_fire_smoke.py  
5. 训练模型  
python train_perfect.py  
