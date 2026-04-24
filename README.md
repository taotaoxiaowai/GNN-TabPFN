# GCN-TabPFN

基于 PyTorch Geometric 的图节点表征学习实验项目，支持多种图编码器（GCN/GAT/GraphSAGE）、多种编码器融合策略（none/sum/weighted）、多种推理头（linear/TabPFN/LimiX），并支持 TabPFN bagging 与冻结 TabPFN 前向训练模式。

## 1. 环境准备

### 1.1 使用现有环境

```powershell
conda activate gcn_tabpfn
pip install -e .[dev]
```

### 1.2 使用项目提供的 Python 3.11 环境

```powershell
conda env create -f environment.gcn_tabpfn-py311.yml
conda activate gcn_tabpfn_py311
pip install -e .[dev]
```

### 1.3 关键依赖

- `torch`
- `torch-geometric`
- `scikit-learn`
- `tabpfn`（仅 TabPFN 推理/训练链路需要）
- `limix`（仅 LimiX 推理链路需要，可通过 `pip install -e .[limix]`）

## 2. 入口脚本

### 2.1 主实验脚本

```powershell
python examples/train_gcn_planetoid.py --dataset Cora
```

### 2.2 支持数据集

- `Cora`
- `CiteSeer`
- `PubMed`
- `Computers`
- `Photo`
### 2.3 数据集信息查看

```powershell
python examples/load_planetoid_datasets.py
```

## 3. 运行流程（分支版）

`examples/train_gcn_planetoid.py` 的流程可概括为：

1. 解析参数、设随机种子、加载 Planetoid 数据。
2. 生成 train/test mask（`mask` 或 `random`）。
3. 根据 `--encoder-fusion` 进入分支：
   - `none`：仅训练一个基模型（由 `--base-model` 指定）。
   - `sum`：分别训练 `GCN/GAT/SAGE`，embedding 相加并按训练集统计量标准化。
   - `weighted`：`GCN/GAT/SAGE` 加权融合，权重 `w` 可训练。
4. 当 `--encoder-fusion weighted` 时，根据 `--weighted-fusion-train-strategy`：
   - `two-stage`：先训三路 encoder，再训融合权重 `w`。
   - `joint`：encoder 参数 + 融合权重 `w` + 临时线性头联合训练。
5. 编码器预训练目标由 `--encoder-pretrain-head` 决定：
   - `linear`：标准线性头监督预训练。
   - `tabpfn-frozen-forward`：冻结 TabPFN 适配器参数，仅更新 encoder；前向经过 TabPFN 适配器。
     - 该模式下适配器内部初始化 TabPFN 时会强制 `ensemble_members=1`，以降低训练阶段显存占用。
     - 真实推理阶段仍使用 `--prediction-head tabpfn` 的默认 TabPFN 设置。
6. 得到最终 embedding 后，进入 `--prediction-head` 推理分支：
   - `linear`
   - `tabpfn`
   - `limix`
7. 当 `--prediction-head tabpfn` 且 `--tabpfn-bagging` 开启时：
   - 对训练集做 m 次 bootstrap 子上下文采样
   - 每个子上下文独立推理（可并行）
   - 通过 `average` 或 `vote` 聚合结果
8. 当 `--prediction-head tabpfn-ensemble-selection` 时：
   - 对训练集进行 fit/val 划分
   - 为每个源表生成多个候选 TabPFN 模型（行采样 + 列采样）
   - 使用贪心策略逐步选择能提升验证准确率的候选模型
   - 最终用完整训练集重新推理选中的候选模型，取平均概率作为预测
9. 当 `--prediction-head tabpfn-ensemble-average` 时：
   - 为每个源表生成多个候选 TabPFN 模型（行采样 + 列采样）
   - 直接平均所有候选模型的测试概率作为最终预测
10. 输出 accuracy/report，并按 `--save-prefix` 保存结果文件。

## 4. 全参数说明（train_gcn_planetoid.py）

命令格式：

```powershell
python examples/train_gcn_planetoid.py [参数]
```

### 4.1 数据与划分参数

- `--dataset`：数据集名。可选 `Cora/CiteSeer/PubMed`，默认 `Cora`
- `--root`：数据根目录，默认 `./data`
- `--split-source`：划分方式。可选 `mask/random`，默认 `random`
- `--test-size`：随机划分测试比例参数，默认 `0.2`
- `--train-test-ratio`：随机划分 train:test 比值，默认 `5.0`
- `--seed`：随机种子，默认 `42`

### 4.2 编码器结构参数

- `--base-model`：单编码器模式下的基模型。`gcn/gat/sage`，默认 `gcn`
- `--encoder-fusion`：编码器融合方式。`none/sum/weighted`，默认 `none`
- `--device`：全局设备。`cpu/cuda`，默认 `cuda`。使用 `cpu` 可强制整个数据和编码器都在 CPU 上运行。
- `--hidden-dim`：隐藏维度，默认 `64`
- `--embedding-dim`：输出 embedding 维度，默认 `64`
- `--num-layers`：图聚合层数，默认 `2`
- `--dropout`：dropout 概率，默认 `0.5`
- `--gat-heads`：GAT 多头数，默认 `8`
- `--inspect-layer`：中间层检查索引（从 0 开始），默认 `0`

### 4.3 编码器预训练参数

- `--pretrain-epochs`：编码器预训练轮数，默认 `200`
- `--pretrain-lr`：编码器预训练学习率，默认 `1e-2`
- `--pretrain-weight-decay`：编码器预训练权重衰减，默认 `5e-4`
- `--pretrain-log-every`：编码器预训练日志间隔，默认 `20`
- `--encoder-pretrain-head`：编码器预训练头。`linear/tabpfn-frozen-forward`，默认 `linear`
- `--tabpfn-frozen-adapter`：冻结 TabPFN 适配器工厂（格式：`module.path:factory`）
- `--tabpfn-frozen-model-path`：传给适配器工厂的可选模型路径
- `--tabpfn-frozen-config-path`：传给适配器工厂的可选配置路径

说明：当 `encoder-pretrain-head=tabpfn-frozen-forward` 时，必须提供可微的 TabPFN 适配器工厂；其返回值需为 `torch.nn.Module`，并实现调用签名：

```python
logits = module(x_context, y_context, x_query)
```

### 4.4 融合参数

- `--fusion-epochs`：融合阶段轮数，默认 `200`
- `--fusion-lr`：融合阶段学习率，默认 `1e-2`
- `--fusion-weight-decay`：融合阶段权重衰减，默认 `0.0`
- `--fusion-log-every`：融合阶段日志间隔，默认 `20`
- `--weighted-fusion-train-strategy`：仅 `encoder-fusion=weighted` 生效。`two-stage/joint`，默认 `two-stage`

### 4.5 推理头参数

- `--prediction-head`：推理头类型。`tabpfn/linear/limix/tabpfn-ensemble-selection/tabpfn-ensemble-average`，默认 `tabpfn`
- `--feature-normalization`：进入表格推理头前的特征处理。`none/standardize`，默认 `standardize`

### 4.6 TabPFN 参数

- `--tabpfn-device`：TabPFN 设备。`cpu/cuda`，默认 `cpu`
- `--tabpfn-bagging`：是否启用 TabPFN bagging（布尔开关）
- `--tabpfn-bagging-m`：bag 数量，默认 `8`
- `--tabpfn-bagging-context-size`：每个 bag 的 bootstrap 样本量，默认 `0`（表示全量训练集大小）
- `--tabpfn-frozen-context-ratio`：冻结 TabPFN 编码器预训练时的 context/query 比例，默认 `0.75`，即 3:1 上下文:查询。
- `--tabpfn-bagging-aggregation`：聚合方式。`average/vote`，默认 `average`
- `--tabpfn-bagging-n-jobs`：bagging 并行 worker 数，默认 `1`

### 4.7 LimiX 参数

- `--limix-model-path`：LimiX 模型/检查点路径（可选）
- `--limix-config-path`：LimiX 推理配置路径（可选）

### 4.8 输出参数

- `--save-prefix`：输出前缀。为空时不保存

## 5. 输出文件

设置 `--save-prefix outputs/exp1` 时，可能保存：

- `outputs/exp1_final_embedding.pt`
- `outputs/exp1_inspect_layer_embedding.pt`
- `outputs/exp1_<prediction_head>_pred.pt`（如 `linear/tabpfn/limix`）
- `outputs/exp1_fusion_weights.pt`（仅融合模式）

## 6. 常用运行命令（按支线）

### 6.1 单编码器 + 线性头

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --base-model sage --encoder-fusion none --prediction-head linear --save-prefix outputs/pubmed_sage_linear
```

### 6.2 单编码器 + TabPFN

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --base-model gat --prediction-head tabpfn --tabpfn-device cpu --save-prefix outputs/pubmed_gat_tabpfn
```

### 6.3 TabPFN + Bagging

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --prediction-head tabpfn --tabpfn-bagging --tabpfn-bagging-m 16 --tabpfn-bagging-context-size 512 --tabpfn-bagging-aggregation average --tabpfn-bagging-n-jobs 4 --save-prefix outputs/pubmed_tabpfn_bagging
```

### 6.4 三编码器 `sum` 融合 + LimiX

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --encoder-fusion sum --prediction-head limix --limix-model-path <model_path> --limix-config-path <config_path> --save-prefix outputs/pubmed_sum_limix
```

### 6.5 三编码器 `weighted` 融合（两阶段）+ TabPFN

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --encoder-fusion weighted --weighted-fusion-train-strategy two-stage --prediction-head tabpfn --save-prefix outputs/pubmed_weighted_twostage_tabpfn
```

### 6.6 三编码器 `weighted` 融合（联合训练）+ 线性头

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --encoder-fusion weighted --weighted-fusion-train-strategy joint --prediction-head linear --save-prefix outputs/pubmed_weighted_joint_linear
```

### 6.7 冻结 TabPFN 前向训练 encoder（需可微适配器）

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --encoder-pretrain-head tabpfn-frozen-forward --tabpfn-frozen-adapter my_project.tabpfn_adapter:build_frozen_tabpfn --tabpfn-frozen-model-path <model_path> --tabpfn-frozen-config-path <config_path> --prediction-head tabpfn --save-prefix outputs/pubmed_frozen_tabpfn_forward
```

### 6.8 TabPFN 集成选择（贪心策略）

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --encoder-fusion sum --prediction-head tabpfn-ensemble-selection --tabpfn-device cuda --tabpfn-ens-sources all-backbones --tabpfn-ens-val-size 0.2 --tabpfn-ens-candidates-per-table 8 --tabpfn-ens-context-size 0 --tabpfn-ens-colsample-min-rate 0.4 --tabpfn-ens-colsample-max-rate 1.0 --tabpfn-ens-max-selected 32 --tabpfn-ens-n-jobs 1 --save-prefix outputs/pubmed_tabpfn_ensemble_selection
```

### 6.9 TabPFN 集成平均（直接平均所有候选）

```powershell
python examples/train_gcn_planetoid.py --dataset PubMed --encoder-fusion sum --prediction-head tabpfn-ensemble-average --tabpfn-device cuda --tabpfn-ens-sources all-backbones --tabpfn-ens-candidates-per-table 8 --tabpfn-ens-context-size 0 --tabpfn-ens-colsample-min-rate 0.4 --tabpfn-ens-colsample-max-rate 1.0 --tabpfn-ens-n-jobs 1 --save-prefix outputs/pubmed_tabpfn_ensemble_average
```

### TabPFN 冻结预训练
python examples/train_gcn_planetoid.py --dataset Cora --split-source random --train-test-ratio 5.0 --encoder-pretrain-head tabpfn-frozen-forward --tabpfn-frozen-adapter examples.tabpfn_adapter:build_frozen_tabpfn --tabpfn-frozen-subset-size 128 --tabpfn-bagging --tabpfn-bagging-m 2 --device cpu --tabpfn-device cpu --pretrain-epochs 20 --tabpfn-bagging-context-size

### 最新命令
python examples\train_gcn_planetoid.py --dataset squirrel --base-model gcn  --split-source random --train-test-ratio 4 --val-size 0.25 --encoder-pretrain-head linear --prediction-head tabpfn --device cuda --pretrain-epochs 200 --tabpfn-device cuda --feature-normalization none -num-layers 1


## 7. 主要代码结构

- `examples/train_gcn_planetoid.py`：主流程编排
- `src/dl_research_kit/data/planetoid.py`：数据加载与划分
- `src/dl_research_kit/models/gcn.py`：GCN 编码器
- `src/dl_research_kit/models/gat.py`：GAT 编码器
- `src/dl_research_kit/models/sage.py`：GraphSAGE 编码器
- `src/dl_research_kit/training/gcn_pretrain.py`：基础编码器预训练
- `src/dl_research_kit/training/fusion.py`：融合训练（sum/weighted/two-stage/joint）
- `src/dl_research_kit/training/frozen_tabpfn.py`：冻结 TabPFN 前向训练
- `src/dl_research_kit/inference/tabpfn.py`：TabPFN 推理与 bagging
- `src/dl_research_kit/inference/limix.py`：LimiX 推理
- `src/dl_research_kit/inference/linear_head.py`：线性头推理
