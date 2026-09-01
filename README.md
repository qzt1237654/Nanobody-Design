# Germline-Absorbing Discrete Diffusion

基于 SEDD 框架实现的 germline-absorbing discrete diffusion (MochiDiff)

## 项目状态

✅ **核心实现完成**  
✅ **所有测试通过 (7/7)**  
⏳ **等待实际 antibody 数据**

## 快速开始

### 1. 验证实现
```bash
python verify_implementation.py
python test_germline_absorbing.py
```

### 2. 阅读文档
**推荐阅读顺序：**
1. `中文实现总结.md` ⭐ - 完整中文概述（最推荐）
2. `README_GERMLINE.txt` - 快速入门
3. `IMPLEMENTATION_REPORT.md` - 详细技术报告

### 3. 准备数据
参考 `data_germline.py` 准备 aligned mature/germline 序列对

### 4. 训练
参考 `example_germline_training.py` 修改训练代码，然后：
```bash
python train.py graph.type=germline_absorb
```

## 项目结构

```
.
├── 核心框架 (SEDD 原始代码)
│   ├── graph_lib.py          # Graph 抽象 (已修改支持 germline)
│   ├── noise_lib.py          # Noise schedules
│   ├── losses.py             # Loss 计算 (已修改支持 germline)
│   ├── sampling.py           # 采样算法 (已修改支持 germline)
│   ├── utils.py              # 工具函数
│   └── catsample.py          # Categorical sampling
│
├── Germline 实现 (新增)
│   ├── graph_lib_germline.py        # GermlineAbsorbing 类
│   ├── data_germline.py             # 数据加载框架
│   ├── test_germline_absorbing.py   # 单元测试 (7/7 通过)
│   ├── example_germline_training.py # 训练示例
│   └── verify_implementation.py     # 自动验证脚本
│
├── 模型
│   └── model/
│       ├── transformer.py    # DiT 模型
│       ├── utils.py          # 模型工具
│       ├── ema.py            # EMA
│       └── ...
│
├── 训练
│   ├── train.py              # 训练主程序
│   ├── run_train.py          # 训练入口
│   └── load_model.py         # 模型加载
│
├── 配置
│   └── configs/
│       ├── config.yaml       # 主配置
│       └── model/
│           ├── small.yaml
│           └── medium.yaml
│
└── 文档
    ├── 中文实现总结.md              # 完整中文总结 ⭐
    ├── README_GERMLINE.txt         # 快速入门
    ├── IMPLEMENTATION_REPORT.md    # 技术报告
    ├── FINAL_SUMMARY.md            # 最终总结
    ├── FILE_INDEX_AND_CHECKLIST.md # 文件索引
    └── QUICK_REFERENCE.py          # 快速参考
```

## 核心概念

### Standard Absorbing vs Germline-Absorbing

| 特性 | Standard Absorbing | Germline-Absorbing |
|------|-------------------|-------------------|
| 吸收状态 | 单一 MASK token | 每个位置的 germline 残基 |
| x_T | [MASK, MASK, ...] | germline 序列 |
| Forward | mature → all MASK | mature → germline |
| 数据需求 | 只需 mature | 需要 aligned pairs |

### 数学公式

```
Forward: p(x_t | x_0, g) = exp(-σ) · δ(x_t = x_0) + [1-exp(-σ)] · δ(x_t = g)
Loss: Score entropy (保留 SEDD 框架，非简单 CE)
```

## 测试结果

```bash
$ python test_germline_absorbing.py

✓ TEST 1: Forward Corruption
✓ TEST 2: Absorbing Property
✓ TEST 3: Sample Independence
✓ TEST 4: Score Entropy Shape
✓ TEST 5: Sample Limit
✓ TEST 6: Positions at Germline
✓ TEST 7: Conceptual Comparison

ALL TESTS PASSED (7/7)
```

## 数据格式

训练数据需要：
```python
{
    'mature': Tensor [B, L],      # 成熟 VHH token IDs
    'germline': Tensor [B, L],    # 对应 germline token IDs (已对齐!)
    'attention_mask': Tensor [B, L]  # 1=valid, 0=padding
}
```

**关键要求：**
- mature 和 germline 必须预先对齐
- mature.shape == germline.shape
- 使用相同 tokenizer

## 下一步

1. ✅ 核心实现完成
2. ✅ 测试验证通过
3. ⏳ 准备 aligned antibody 数据
4. ⏳ 修改 run_train.py (见 example_germline_training.py)
5. ⏳ 运行训练
6. ⏳ 评估生成质量

## 论文引用

- SEDD: Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution", ICML 2024
- MochiDiff: "Conditional generation of antibody sequences with classifier-guided germline-absorbing discrete diffusion", arXiv:2605.06720

## 实现统计

- 新增代码: ~1,365 行
- 文档: ~2,183 行
- 修改: ~85 行
- 测试覆盖: 7/7 通过

**状态**: Ready for real data 🚀
