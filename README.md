# Nanobody 设计 - Germline-Absorbing 扩散模型

基于 SEDD 框架实现的 germline-absorbing discrete diffusion 模型，用于纳米抗体（VHH）序列生成。

## 📋 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [配置参数详解](#配置参数详解)
- [训练命令](#训练命令)
- [超算提交](#超算提交)
- [结果查看](#结果查看)

---

## 项目简介

本项目实现了一个新颖的扩散模型，用于条件生成纳米抗体序列：

- **核心思想**：从成熟抗体序列逐渐扩散到对应的种系（germline）序列，训练模型学习反向过程
- **应用场景**：基于种系序列生成功能性成熟抗体，用于抗体设计和优化
- **技术特点**：每个序列使用其自身的种系序列作为吸收状态，而非单一的MASK token

---

## 环境配置

### 方法一：使用 environment.yaml（推荐，快速安装）

项目提供了完整的环境配置文件，可以一键安装所有依赖：

```bash
# 从 yaml 文件创建环境
conda env create -f environment.yaml

# 激活环境
conda activate sedd_vhh

# 验证安装
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 方法二：手动分步安装（详细步骤）

如果你需要自定义安装或 environment.yaml 出现问题，可以按照以下步骤手动安装：

#### 第一步：创建基础 Python 环境

```bash
# 创建名为 sedd_vhh 的 Python 3.10 环境
conda create -n sedd_vhh python=3.10 -y

# 激活环境
conda activate sedd_vhh
```

#### 第二步：安装核心科学计算包

```bash
# 安装 numpy、scipy 等基础包
conda install -c conda-forge numpy scipy pandas pyyaml tqdm -y
```

#### 第三步：安装 PyTorch（根据你的 CUDA 版本选择）

**如果有 CUDA 11.8：**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**如果有 CUDA 12.1：**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**如果只有 CPU（不推荐，训练会很慢）：**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**查看你的 CUDA 版本：**
```bash
nvidia-smi  # 查看右上角的 CUDA Version
```

#### 第四步：安装配置管理工具

```bash
# Hydra 用于管理配置文件
pip install hydra-core==1.3.2 omegaconf==2.3.0
```

#### 第五步：安装超算调度工具（超算必需）

```bash
# Submitit 用于在 SLURM 集群上提交任务
pip install submitit==1.5.4 hydra-submitit-launcher==1.2.0
```

#### 第六步：安装训练监控工具

```bash
# W&B 用于实验跟踪（可选）
pip install wandb==0.29.0

# Accelerate 用于分布式训练
pip install accelerate==1.14.0
```

#### 第七步：安装模型工具包

```bash
# Einops 用于张量操作
pip install einops==0.8.2 fancy-einsum==0.0.3

# 类型检查和调试工具
pip install jaxtyping==0.3.7 typeguard==4.6.0 rich==15.0.0
```

#### 第八步：安装 Transformers 生态（数据处理和 tokenizer）

```bash
# Hugging Face transformers 和相关工具
pip install transformers==5.16.1
pip install tokenizers==0.23.1
pip install safetensors==0.8.0
pip install datasets==5.0.1
pip install pyarrow==25.0.1
```

#### 第九步：验证安装

```bash
# 验证 Python 和环境
python --version  # 应该显示 Python 3.10.x

# 验证 PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__)"

# 验证 CUDA（如果使用 GPU）
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# 验证关键依赖
python -c "import hydra; print('Hydra:', hydra.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import einops; print('Einops OK')"
python -c "import submitit; print('Submitit OK')"

# 验证项目代码能否导入
python -c "import graph_lib_germline; print('Project imports OK')"
```

#### 常见问题排查

**1. CUDA 版本不匹配**
```bash
# 卸载当前 PyTorch
pip uninstall torch torchvision torchaudio -y

# 重新安装匹配的版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**2. 依赖冲突**
```bash
# 创建全新环境
conda deactivate
conda env remove -n sedd_vhh
# 然后重新按步骤安装
```

**3. 网络问题（国内用户）**
```bash
# 使用清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

### 完整依赖列表

安装完成后，你的环境应包含以下主要包：

| 包名 | 版本 | 用途 |
|------|------|------|
| python | 3.10 | 基础 Python 环境 |
| pytorch | 最新 | 深度学习框架 |
| hydra-core | 1.3.2 | 配置管理 |
| transformers | 5.16.1 | Tokenizer 和数据处理 |
| einops | 0.8.2 | 张量操作 |
| wandb | 0.29.0 | 实验追踪 |
| submitit | 1.5.4 | SLURM 任务提交 |
| datasets | 5.0.1 | 数据集加载 |
| accelerate | 1.14.0 | 分布式训练 |

### 环境导出（方便在其他机器复现）

```bash
# 导出完整环境
conda env export > my_environment.yaml

# 导出 pip 依赖
pip freeze > requirements.txt

# 在新机器上复现
conda env create -f my_environment.yaml
```

---

## 数据准备

### 数据格式要求

训练数据需要是 **TSV 文件**，包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `mature` | 成熟抗体序列 | `QVQLQESGGG...` |
| `germline` | 对应的种系序列 | `QVQLQESGPG...` |

**重要要求**：
- mature 和 germline 序列必须**预先对齐**
- 两者长度必须相同
- 只使用 20 种标准氨基酸（A-Y，不包括 B, J, O, U, X, Z）

### 数据路径配置

在配置文件中设置数据路径：

```yaml
data:
  tsv_path: /gpfs/work/bio/zhengtaoqi24/germline/VHHCorpus-2M_top1_pairs_clean.tsv
```

或通过命令行覆盖：

```bash
python train.py data.tsv_path=/path/to/your/data.tsv
```

---

## 配置参数详解

配置文件位于：`configs/config.yaml`

### 基础配置

```yaml
# GPU 数量（当前实现支持单卡训练）
ngpus: 1

# 词汇表大小（20种标准氨基酸）
tokens: 20
```

### 训练参数 (`training`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 每个 GPU 每次加载的序列对数量 |
| `accum` | 1 | 梯度累积步数（实际batch = batch_size × accum） |
| `n_iters` | 500000 | 总训练步数（optimizer updates） |
| `log_freq` | 50 | 每隔多少步打印训练损失 |
| `eval_freq` | 500 | 每隔多少步在验证集上评估 |
| `snapshot_freq` | 10000 | 每隔多少步保存模型检查点 |
| `snapshot_freq_for_preemption` | 5000 | 每隔多少步保存中断恢复检查点 |
| `snapshot_sampling` | true | 是否在每个检查点生成样本序列 |
| `ema` | 0.9999 | 指数移动平均衰减率，用于稳定模型 |

**示例**：
- `batch_size=64, accum=2` → 实际梯度基于 128 个样本计算
- `n_iters=500000, batch_size=64` → 总共训练约 3200 万个样本

### 数据参数 (`data`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tsv_path` | 必填 | TSV 数据文件的绝对路径 |
| `max_length` | 128 | 序列的最大长度（padding 到此长度） |
| `train_ratio` | 0.95 | 训练集占比（剩余为验证集） |
| `seed` | 42 | 数据划分的随机种子 |
| `num_workers` | 4 | 数据加载的并行进程数 |

**注意**：
- `max_length` 必须与 `model.length` 保持一致
- 较长序列会占用更多显存，根据 GPU 内存调整

### 图结构参数 (`graph`)

```yaml
graph:
  type: germline_absorb  # 固定使用 germline-absorbing 图
```

**不要修改此参数**，除非使用其他扩散类型。

### 噪声参数 (`noise`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `type` | loglinear | 噪声调度类型（对数线性） |
| `eps` | 0.001 | 数值稳定性参数，避免完全吸收 |

**噪声调度公式**：
```
σ(t) = -log(1 - (1-eps)t)
移动概率 = (1-eps)t
```

- `eps` 越小，扩散越完全（但可能数值不稳定）
- 推荐保持默认值 0.001

### 采样参数 (`sampling`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `predictor` | euler | 反向过程求解器（Euler 方法） |
| `steps` | 128 | 反向采样的步数 |
| `noise_removal` | true | 最后一步是否去除噪声 |
| `batch_size` | 8 | 每次生成的样本数量 |

**采样步数说明**：
- 更多步数 → 更高质量，但更慢
- 推荐范围：64-256

### 优化器参数 (`optim`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `optimizer` | AdamW | 优化器类型 |
| `lr` | 1e-4 | 学习率 |
| `weight_decay` | 0.01 | 权重衰减（L2 正则化） |
| `beta1` | 0.9 | Adam 的一阶矩估计衰减率 |
| `beta2` | 0.999 | Adam 的二阶矩估计衰减率 |
| `eps` | 1e-8 | Adam 的数值稳定性参数 |
| `warmup` | 2000 | 学习率预热步数 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

**学习率调度**：
- 前 `warmup` 步：从 0 线性增加到 `lr`
- 之后保持恒定（可自定义其他策略）

### 模型参数 (`model`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 512 | Transformer 隐藏层维度 |
| `cond_dim` | 512 | 条件编码维度（种系序列编码） |
| `length` | 128 | 序列长度（必须等于 data.max_length） |
| `n_blocks` | 4 | Transformer 层数 |
| `n_heads` | 8 | 注意力头数 |
| `dropout` | 0.0 | Dropout 比率 |

**模型规模调整**：

| 规模 | hidden_size | n_blocks | n_heads | 参数量（约） |
|------|-------------|----------|---------|-------------|
| 小 | 256 | 4 | 4 | ~10M |
| 中 | 512 | 4 | 8 | ~30M |
| 大 | 768 | 6 | 12 | ~60M |
| 超大 | 1024 | 8 | 16 | ~120M |

### 输出路径 (`hydra`)

```yaml
hydra:
  run:
    dir: exp_local/germline_vhh/${now:%Y.%m.%d}/${now:%H%M%S}
```

自动创建按日期和时间命名的实验目录，例如：
```
exp_local/germline_vhh/2026.09.01/143022/
├── checkpoints/          # 模型检查点
├── samples/              # 生成的样本序列
├── logs/                 # 训练日志
└── checkpoints-meta/     # 中断恢复检查点
```

---

## 训练命令

### 本地训练（单卡）

```bash
# 使用默认配置
python train.py

# 覆盖特定参数
python train.py \
  data.tsv_path=/path/to/data.tsv \
  training.batch_size=32 \
  training.n_iters=100000 \
  model.hidden_size=768

# 使用自定义配置文件
python train.py --config-name=my_config
```

### 调试模式（小规模测试）

```bash
python train.py \
  training.n_iters=1000 \
  training.batch_size=16 \
  training.snapshot_freq=500 \
  model.hidden_size=256 \
  model.n_blocks=2
```

### 从检查点恢复训练

训练会自动从最新的检查点恢复。如果需要从特定检查点恢复：

```bash
# 将检查点复制到 checkpoints-meta/checkpoint.pth
cp exp_local/.../checkpoints/checkpoint_50000.pth \
   exp_local/.../checkpoints-meta/checkpoint.pth

# 重新启动训练
python train.py
```

---

## 超算提交

### SLURM 提交脚本

使用提供的 `submit_train.sh` 脚本提交任务：

```bash
sbatch submit_train.sh
```

### 脚本参数说明

编辑 `submit_train.sh` 以调整资源配置：

```bash
#SBATCH --job-name=vhh_sedd_train      # 任务名称
#SBATCH --nodes=1                       # 节点数
#SBATCH --ntasks-per-node=1             # 每节点任务数
#SBATCH --cpus-per-task=8               # 每任务CPU核心数
#SBATCH --partition=gpu3090             # 分区名称
#SBATCH --qos=4gpus                     # QoS设置
#SBATCH --gres=gpu:1                    # GPU数量
#SBATCH --mem=64G                       # 内存大小
```

**重要路径配置**：

```bash
PROJECT_DIR="/gpfs/work/bio/zhengtaoqi24/Score-Entropy-Discrete-Diffusion-main"
DATA_FILE="/gpfs/work/bio/zhengtaoqi24/germline/VHHCorpus-2M_top1_pairs_clean.tsv"
CONFIG_NAME="config"  # configs/config.yaml 的名称（不含.yaml后缀）
```

### 查看任务状态

```bash
# 查看队列中的任务
squeue -u $USER

# 查看特定任务
squeue -j <job_id>

# 取消任务
scancel <job_id>
```

### 查看实时日志

```bash
# 查看标准输出
tail -f logs/train_<job_id>.out

# 查看错误输出
tail -f logs/train_<job_id>.err
```

### 多卡训练（实验性）

如果需要使用多卡：

```bash
# 修改配置
#SBATCH --gres=gpu:4

# 修改配置文件
ngpus: 4
training:
  batch_size: 32  # 每卡batch size，总batch = 32 × 4 = 128
```

---

## 结果查看

### 目录结构

训练完成后，输出目录包含：

```
exp_local/germline_vhh/2026.09.01/143022/
├── checkpoints/
│   ├── checkpoint_10000.pth    # 每 10k 步的检查点
│   ├── checkpoint_20000.pth
│   └── ...
├── checkpoints-meta/
│   └── checkpoint.pth          # 最新的中断恢复检查点
├── samples/
│   ├── iter_10000/
│   │   └── sample_0.txt        # 生成的序列样本
│   ├── iter_20000/
│   └── ...
├── logs/
│   └── log.txt                 # 完整训练日志
├── .hydra/
│   └── config.yaml             # 本次运行的完整配置
└── .submitit/                  # SLURM 提交信息（如果使用）
```

### 监控训练进度

```bash
# 实时查看日志
tail -f exp_local/germline_vhh/2026.09.01/143022/logs/log.txt

# 提取损失曲线
grep "training_loss" exp_local/.../logs/log.txt
grep "evaluation_loss" exp_local/.../logs/log.txt
```

### 查看生成的序列

```bash
# 查看某个检查点生成的样本
cat exp_local/.../samples/iter_50000/sample_0.txt
```

示例输出：
```
QVQLQESGGGSVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAITSGGSTYYADSVRGRFTISRDNSKNTLYLQMNSLRPEDTAVYYCAAXXXXX
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK...
...
```

### 评估指标

你可以编写脚本评估生成质量：

```python
# 示例：计算与种系序列的相似度
import numpy as np

def evaluate_samples(sample_file, germline_file):
    samples = open(sample_file).readlines()
    germlines = open(germline_file).readlines()
    
    identities = []
    for s, g in zip(samples, germlines):
        identity = sum(a == b for a, b in zip(s, g)) / len(s)
        identities.append(identity)
    
    print(f"平均相似度: {np.mean(identities):.2%}")
    print(f"标准差: {np.std(identities):.2%}")
```

---

## 常见问题

### 1. CUDA out of memory

**解决方法**：
```yaml
training:
  batch_size: 32  # 减小 batch size
model:
  hidden_size: 256  # 减小模型尺寸
  n_blocks: 3
```

### 2. 数据加载慢

**解决方法**：
```yaml
data:
  num_workers: 8  # 增加数据加载进程
```

### 3. 检查点文件过大

检查点包含完整模型、优化器状态、EMA 等，较大是正常的。如需减小：
- 仅保留关键检查点
- 删除中间检查点：
```bash
cd exp_local/.../checkpoints
ls checkpoint_*.pth | grep -v "00000.pth$" | xargs rm
```

### 4. 训练不收敛

**调整建议**：
```yaml
optim:
  lr: 5e-5          # 降低学习率
  warmup: 5000      # 增加预热步数
  grad_clip: 0.5    # 更严格的梯度裁剪

training:
  ema: 0.995        # 降低 EMA 衰减（更快适应）
```

---

## 技术原理

### Germline-Absorbing 扩散过程

**前向过程**（加噪）：
```
成熟序列 x₀ → ... → 种系序列 x_T
```

每个位置 i 在时间 t 的转移概率：
```
p(x_t[i] | x₀[i], germline[i]) = 
    exp(-σ(t)) · δ(x_t[i] = x₀[i]) +           # 保持成熟序列
    [1 - exp(-σ(t))] · δ(x_t[i] = germline[i]) # 变为种系序列
```

**反向过程**（去噪）：
```
种系序列 x_T → ... → 成熟序列 x₀
```

模型学习在每一步预测：
```
p(x_{t-Δt} | x_t, germline, t)
```

### 与标准扩散的区别

| 特性 | 标准 Absorbing | Germline-Absorbing |
|------|----------------|-------------------|
| 吸收状态 | 全局 MASK token | 每个序列的种系序列 |
| 条件信息 | 无/外部条件 | 种系序列作为条件 |
| 生成控制 | 无条件生成 | 基于种系的条件生成 |
| 应用场景 | 通用序列生成 | 抗体成熟度模拟 |

---

## 引用

如果使用本代码，请引用：

```bibtex
@article{mochidiff2024,
  title={Conditional generation of antibody sequences with classifier-guided germline-absorbing discrete diffusion},
  journal={arXiv preprint arXiv:2605.06720},
  year={2024}
}

@inproceedings{sedd2024,
  title={Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  booktitle={ICML},
  year={2024}
}
```

---

## 许可

请查阅原始 SEDD 仓库的许可证。

---

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。
