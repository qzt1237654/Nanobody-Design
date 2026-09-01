# Nanobody 设计 - Germline-Absorbing 扩散模型

基于 SEDD 框架实现的 germline-absorbing discrete diffusion 模型，用于纳米抗体（VHH）序列生成。

---

## 1. 环境安装

### 方法一：使用 environment.yaml（推荐）

```bash
conda env create -f environment.yaml
conda activate sedd_vhh
```

### 方法二：手动安装

```bash
# 第一步：创建环境
conda create -n sedd_vhh python=3.10 -y
conda activate sedd_vhh

# 第二步：安装基础包
conda install -c conda-forge numpy scipy pandas pyyaml tqdm -y

# 第三步：安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 第四步：安装配置和训练工具
pip install hydra-core==1.3.2 omegaconf==2.3.0
pip install submitit==1.5.4 hydra-submitit-launcher==1.2.0
pip install wandb==0.29.0 accelerate==1.14.0

# 第五步：安装模型工具
pip install einops==0.8.2 fancy-einsum==0.0.3
pip install jaxtyping==0.3.7 typeguard==4.6.0 rich==15.0.0

# 第六步：安装 Transformers 生态
pip install transformers==5.16.1 tokenizers==0.23.1
pip install safetensors==0.8.0 datasets==5.0.1 pyarrow==25.0.1

# 验证安装
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 2. 超算训练

### 修改路径

编辑 `submit_train.sh`：

```bash
# 修改项目路径
PROJECT_DIR="/你的路径/Score-Entropy-Discrete-Diffusion-main"

# 修改数据路径
DATA_FILE="/你的路径/VHHCorpus-2M_top1_pairs_clean.tsv"

# 修改 conda 环境名（如果不是 sedd_vhh）
conda activate 你的环境名
```

编辑 `configs/config.yaml`：

```yaml
data:
  tsv_path: /你的路径/VHHCorpus-2M_top1_pairs_clean.tsv
```

### 提交训练任务

```bash
# 提交任务
sbatch submit_train.sh

# 查看任务状态
squeue -u $USER

# 查看日志
tail -f logs/train_<job_id>.out
```

### 本地训练

```bash
python train.py
```

---

## 3. 可修改参数

编辑 `configs/config.yaml`：

### 训练参数

```yaml
training:
  batch_size: 64        # 每卡每次训练样本数（显存不够就改小）
  n_iters: 500000       # 总训练步数
  log_freq: 50          # 每多少步打印日志
  snapshot_freq: 10000  # 每多少步保存检查点
```

### 模型参数

```yaml
model:
  hidden_size: 512      # 模型隐藏层维度（256/512/768/1024）
  n_blocks: 4           # Transformer 层数（4/6/8）
  n_heads: 8            # 注意力头数（4/8/12/16）
```

**常用模型规模**：
- 小模型：`hidden_size=256, n_blocks=4, n_heads=4`
- 中模型：`hidden_size=512, n_blocks=4, n_heads=8`（默认）
- 大模型：`hidden_size=768, n_blocks=6, n_heads=12`

### 优化器参数

```yaml
optim:
  lr: 1e-4              # 学习率
  warmup: 2000          # 预热步数
  grad_clip: 1.0        # 梯度裁剪
```

### 数据参数

```yaml
data:
  max_length: 128       # 序列最大长度
  train_ratio: 0.95     # 训练集比例
  num_workers: 4        # 数据加载进程数
```

### 采样参数

```yaml
sampling:
  steps: 128            # 采样步数（越多越慢但质量越好）
  batch_size: 8         # 每次生成样本数
```

### 命令行覆盖参数

```bash
# 不修改配置文件，直接命令行指定
python train.py \
  training.batch_size=32 \
  model.hidden_size=768 \
  optim.lr=5e-5
```

---

## 数据格式

TSV 文件需要包含两列：

| 列名 | 说明 |
|------|------|
| `mature` | 成熟抗体序列 |
| `germline` | 对应的种系序列（必须已对齐） |

**要求**：mature 和 germline 必须等长且预先对齐。

---

## 输出结果

训练输出在 `exp_local/germline_vhh/日期/时间/` 目录：

```
├── checkpoints/          # 模型检查点
│   ├── checkpoint_10000.pth
│   └── checkpoint_20000.pth
├── samples/              # 生成的序列样本
│   ├── iter_10000/
│   └── iter_20000/
└── logs/                 # 训练日志
    └── log.txt
```

查看训练进度：
```bash
tail -f exp_local/.../logs/log.txt
```
