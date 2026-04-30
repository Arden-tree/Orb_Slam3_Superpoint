# SuperPoint Fine-tune 训练与 FMQL30TAI 三核异构部署方案

## 1. 项目概述

将 ORB-SLAM3 中的 ORB 特征提取替换为 SuperPoint，通过 Fine-tune 提升 SLAM 场景下的特征质量，部署到复旦微 FMQL30TAI 异构芯片（CPU + NPU + FPGA PL）上实现端侧实时推理。

### 1.1 目标硬件

| 参数 | FMQL30TAI |
|------|-----------|
| 芯片类型 | FPAI（FPGA + AI）异构融合芯片 |
| CPU | 4 核 Cortex-A（PS 端） |
| NPU | 8 TOPS (INT8) "诸葛"架构 |
| FPGA PL | 125K 逻辑单元 |
| 功耗 | ~8W |

### 1.2 工具链

| 工具 | 定位 | 类比 |
|------|------|------|
| **ProciseAI** | FPGA 硬件开发 IDE（PS/PL 配置、RTL 综合、bitstream 生成） | 类似 Vivado |
| **iCraft** | AI 模型量化 + 编译 + NPU 部署（ONNX → INT8 → NPU 可执行文件） | 类似 TensorRT |
| **iCraft Runtime** | 板端 NPU 推理驱动引擎 | 类似 TensorRT Runtime |

---

## 2. 训练方法选择

### 2.1 方案对比

| | Fine-tune（选定） | Homographic Adaptation 自监督 |
|---|---|---|
| 训练数据 | SLAM 图像 + Homographic 变换生成匹配对 | 只需要 SLAM 图像 |
| 标注需求 | 需要生成匹配对 | 完全不需要标注 |
| 训练难度 | 中等 | 较高（多阶段） |
| 训练时间 | 1-2 天 | 3-5 天 |
| GPU 需求 | 8GB+ | 8GB+ |
| 效果 | 在目标场景提升明显 | 更通用，泛化更好 |
| NPU 兼容性 | 不改变网络结构，权重兼容 | 可能产生量化不友好的权重分布 |

### 2.2 选择 Fine-tune 的理由

1. **NPU 推理只关心卷积部分**：Fine-tune 不改变网络结构，只调权重，导出 ONNX 后 NPU 兼容性不变
2. **当前瓶颈是特征区分度不够**：SearchByBoW 只匹配 2-14 个点，说明描述子区分度不足，Fine-tune 直接优化描述子质量
3. **训练成本低、迭代快**：1-2 天收敛，部署到悟坚板前可快速验证
4. **保留预训练知识**：HA 本质是从头训练，会丢弃预训练模型已学到的通用特征

---

## 3. 三核异构架构设计

### 3.1 任务硬件适配分析

| 任务 | 计算特征 | CPU | NPU | FPGA PL | 推荐 |
|------|---------|-----|-----|---------|------|
| Conv 编码器 | 密集矩阵乘 | 慢 | **最佳** | 能做但不如NPU | **NPU** |
| Softmax + Reshape | 逐元素 | 一般 | 不友好 | 能做 | CPU |
| NMS (dilate) | 滑窗比较 | 慢 | 不支持 | **最佳** | **FPGA** |
| 描述子双线性插值 | 256维×N点并行 | 很慢 | 不适合 | **最佳** | **FPGA** |
| L2 归一化 | 256维平方和+除法 | 一般 | 不适合 | 好 | **FPGA** |
| L2 距离匹配 | 256维点积×千级并行 | **瓶颈** | 不适合 | **最佳** | **FPGA** |
| 图像预处理 | 逐像素 | 一般 | 不适合 | 好 | CPU/FPGA |
| SLAM 调度 (BA/优化) | 复杂逻辑 | **CPU** | 不适合 | 不适合 | **CPU** |

### 3.2 三核分工架构图

```
+----------------------------------------------------------+
|                  FMQL30TAI 三核异构                        |
|                                                           |
|  CPU (4 x A53)          NPU (8 TOPS)        FPGA PL (125K)|
|  +------------+        +------------+       +--------------+|
|  | 调度主循环  |        | Conv编码器  |       | 1) DMA 引擎   ||
|  | 图像预处理  |  --->  | Score head  | -->  |    CPU<->NPU  ||
|  | Softmax    |        | Desc head   |       |    零拷贝传输  ||
|  | SLAM跟踪   |        +------------+       |              ||
|  | BA优化     |                              | 2) NMS 加速   ||
|  | 回环检测   |                              |   滑窗+阈值   ||
|  +------------+                              |              ||
|       ^                                      | 3) 描述子插值  ||
|       |                                      |   256维xN并行 ||
|       |                                      |              ||
|       +--------- 匹配结果 <----------------- | 4) L2 匹配    ||
|                                              |   256维点积   ||
|                                              |   千级并行    ||
|                                              |              ||
|                                              | 5) L2 归一化  ||
|                                              +--------------+|
+----------------------------------------------------------+

数据流：
Image -> CPU预处理 -> [DMA] -> NPU推理 -> [DMA] -> FPGA后处理 -> CPU SLAM
```

### 3.3 FPGA 加速模块预估收益

| # | 模块 | 为什么适合 FPGA | CPU 耗时 | FPGA 预估 |
|---|------|----------------|---------|----------|
| 1 | DMA 引擎 | CPU↔NPU 零拷贝，消除 memcpy | 1-2ms 拷贝 | ~0ms |
| 2 | NMS 加速 | dilate 是固定窗口滑窗比较，FPGA 天然流式 | ~5ms | ~0.1ms |
| 3 | 描述子插值 | N×256 双线性插值，每个描述子独立，大规模并行 | ~15ms | ~0.5ms |
| 4 | L2 距离匹配 | 256 维点积是乘加阵列，FPGA 最擅长 | ~20ms | ~1ms |
| 5 | L2 归一化 | 256 维平方和归约+除法，流式处理 | ~2ms | ~0.1ms |

**最大收益是 4) L2 匹配**：ORB-SLAM3 跟踪阶段 SearchByProjection/SearchByBoW 每帧要做上千次 256 维点积比较，FPGA 可做成 256 并行乘加树，一个时钟周期完成一个距离计算。

---

## 4. 完整部署流程（7 步）

### 4.1 工具使用总览

| 步骤 | ProciseAI | iCraft | PyTorch |
|------|-----------|--------|---------|
| 1. Fine-tune 训练 | | | **Yes** |
| 2. 导出 ONNX | | | **Yes** |
| 3. 量化编译 | | **Yes** | |
| 4. PL 硬件设计 | **Yes** | | |
| 5. PS 配置 | **Yes** | | |
| 6. 交叉编译 C++ | | | |
| 7. 板端部署运行 | 固件加载 | **Runtime** | |

### 4.2 第 1 步：Fine-tune 训练（GPU 服务器）

**工具**：PyTorch

**数据准备**：
- 输入：TUM 数据集图像（fr1_desk, fr1_room, fr1_xyz, fr2_desk 等）
- 用 Homographic 变换从每张图像生成匹配对（旋转、缩放、透视变换）
- 不需要人工标注

**训练流程**：
```
TUM 图像
    |
    v
随机 Homographic 变换生成图像对 (I, I')
    |
    v
加载预训练 SuperPoint 权重 (superpoint_v1.pth)
    |
    v
前向推理得到两幅图的关键点 + 描述子
    |
    v
计算 Loss：
  - 检测 Loss：关键点检测交叉熵（65 通道 softmax）
  - 描述子 Loss：正样本对拉近 + 负样本对推远
    |
    v
反向传播，小学习率微调（lr=1e-4）
    |
    v
输出：superpoint_finetuned.pth
```

**关键参数**：
- 学习率：1e-4（编码器 1e-5，head 1e-4 差异化学习率）
- Batch size：8（取决于 GPU 显存）
- 训练轮次：50-100 epoch
- 优化器：AdamW
- 验证集：留出 10% TUM 图像

**训练环境需求**：
- GPU：NVIDIA 8GB+（如 RTX 3060 及以上）
- PyTorch 2.x + CUDA 11.x
- 训练时间：1-2 天

### 4.3 第 2 步：导出 NPU 友好 ONNX（GPU 服务器）

**工具**：PyTorch `torch.onnx.export`

**关键要求**：
- **固定输入 shape**：`(1, 1, 480, 640)`，去掉 dynamic_axes（NPU 要求固定尺寸）
- **只导出卷积部分**：不做 softmax、reshape、L2 归一化（这些移到 CPU/FPGA）
- **输出两个 tensor**：
  - `semi`: `(1, 65, H/8, W/8)` 原始 logits
  - `desc`: `(1, 256, H/8, W/8)` 原始描述子

**导出脚本要点**：
```python
torch.onnx.export(
    model,
    dummy_input,                    # (1, 1, 480, 640) float32
    "superpoint_npu.onnx",
    opset_version=18,
    input_names=['input'],
    output_names=['semi', 'desc'],
    # 注意：不要加 dynamic_axes，固定 shape
    do_constant_folding=True,
    export_params=True,
)
```

**ONNX 模型内算子分布**：
- NPU 友好（占 95% 计算量）：Conv x12, Relu x10, MaxPool x3
- 已移除（原模型中的后处理）：Softmax, Reshape x6, Transpose x2, Slice, Squeeze, Expand

### 4.4 第 3 步：iCraft 量化 + 编译（开发机）

**工具**：iCraft

**流程**：
```
superpoint_npu.onnx
       |
       v
  iCraft 导入 ONNX 模型
       |
       v
  配置量化策略：
    - 编码器 (conv1a ~ conv4b)      -> INT8
    - Score head (convPa, convPb)   -> INT8
    - Desc head (convDa, convDb)    -> FP16  <-- 保留精度
       |
       v
  准备校准数据集：
    - 从 TUM 数据集选 100-200 张图像
    - 转为 (1, 1, 480, 640) float32 格式
       |
       v
  iCraft 量化校准（FP32 -> INT8/FP16）
       |
       v
  iCraft 编译优化（算子融合、内存规划）
       |
       v
  iCraft 仿真验证：
    - 对比 ONNX 原始输出 vs 量化后输出
    - 检查描述子余弦相似度误差 < 1%
    - 检查关键点检测 mAP 下降 < 2%
       |
       v
  输出：npu_model.bin（NPU 可执行固件）
```

**量化配置要点**：
- `convDa`、`convDb`（描述子头）必须保持 FP16，INT8 会严重损失匹配精度
- 编码器和 score head 可安全使用 INT8（卷积对量化鲁棒）
- 校准数据必须覆盖典型 SLAM 场景（室内、弱光、运动模糊）

### 4.5 第 4 步：ProciseAI PL 端设计（开发机）

**工具**：ProciseAI

**设计 5 个 FPGA 加速模块**：

```
ProciseAI 项目
       |
       v
  RTL 设计（Verilog/VHDL）：
    |
    +-- 模块 1: DMA 引擎
    |     - PS <-> NPU 零拷贝数据传输
    |     - AXI DMA IP 核配置
    |     - 消除 CPU memcpy 开销
    |
    +-- 模块 2: NMS 加速器
    |     - 输入：热力图 (H, W) float32
    |     - 滑窗比较 + 阈值过滤
    |     - 流式处理，一行一行扫描
    |     - 输出：关键点坐标列表 + 置信度
    |
    +-- 模块 3: 描述子插值器
    |     - 输入：N 个关键点坐标 + desc 特征图 (256, H/8, W/8)
    |     - 每个关键点独立执行 256 维双线性插值
    |     - 256 路并行，单个描述子 1 时钟周期
    |     - 输出：N x 256 描述子矩阵
    |
    +-- 模块 4: L2 距离匹配器
    |     - 输入：query 描述子 (M x 256) + reference 描述子 (K x 256)
    |     - 256 维点积乘加树
    |     - M x K 并行比较
    |     - 输出：最优匹配索引 + 距离
    |     - 核心：ORB-SLAM3 SearchByProjection/SearchByBoW 的计算瓶颈
    |
    +-- 模块 5: L2 归一化器
    |     - 输入：N x 256 描述子矩阵
    |     - 256 维平方和归约 + sqrt + 除法
    |     - 流式处理
    |     - 输出：归一化后的描述子
    |
       v
  ProciseAI 综合 + 布局布线
       |
       v
  资源评估（125K 逻辑单元预算）：
    - DMA 引擎：~5K LUT
    - NMS 加速器：~8K LUT
    - 描述子插值器：~20K LUT
    - L2 匹配器：~40K LUT（最大模块）
    - L2 归一化器：~10K LUT
    - AXI 互联 + 其他：~10K LUT
    - 总计：~93K / 125K LUT（74% 利用率）
       |
       v
  生成 bitstream：pl_bitstream.bin
```

### 4.6 第 5 步：ProciseAI PS 端配置（开发机）

**工具**：ProciseAI

**配置内容**：
- PS（ARM Cortex-A）外设配置：
  - DDR 内存控制器
  - UART（调试串口）
  - 以太网（数据传输）
  - NPU 接口（PS↔NPU 通信）
- PS↔PL AXI 总线配置：
  - CPU 通过 AXI Lite 读写 PL 寄存器（触发加速操作）
  - AXI Master 用于 PL 直接访问 DDR（DMA 数据搬运）
- 生成 FSBL（First Stage Boot Loader）+ Linux 设备树

### 4.7 第 6 步：交叉编译 C++ 应用（开发机）

**工具**：`arm-linux-gnueabihf` 交叉编译器

**代码修改**：

`SuperPointExtractor.cc` 需要修改：

| 原实现 | 替换为 |
|--------|--------|
| ONNX Runtime `session->Run()` | iCraft Runtime `icraft_infer()` |
| `cv::dilate` NMS | PL 硬件加速（AXI 写寄存器触发） |
| CPU 循环描述子插值 | PL 描述子插值器 |
| CPU `DescriptorDistance` L2 循环 | PL L2 匹配器 |
| CPU L2 归一化循环 | PL L2 归一化器 |

**编译流程**：
```
交叉编译 ORB-SLAM3 库
  + iCraft Runtime SDK
  + PL 驱动层（AXI 寄存器访问封装）
  + OpenCV（ARM 版本）
       |
       v
  输出：mono_tum（ARM 可执行文件）
```

### 4.8 第 7 步：板端部署运行（FMQL30TAI 板卡）

**工具**：iCraft Runtime + ProciseAI 生成的固件

**板卡启动流程**：
```
FMQL30TAI 上电
  |
  v
FSBL 加载（PS 初始化）
  |
  v
加载 PL bitstream（FPGA 配置）
  |
  v
加载 NPU 模型（iCraft Runtime 初始化）
  |
  v
Linux 系统启动
  |
  v
运行 ORB-SLAM3 应用
  |
  +-- CPU: 图像读取 + 灰度转换 + 归一化到 [0,1]
  +-- CPU: 将图像写入 DMA buffer
  +-- NPU: SuperPoint 卷积推理（iCraft Runtime，预估 5-10ms）
  +-- PL:  NMS + 描述子插值 + L2 匹配 + L2 归一化
  +-- CPU: SLAM 跟踪/建图调度（TrackLocalMap, BA 等）
```

**预期性能**：

| 指标 | CPU ONNX（当前） | NPU + FPGA（目标） |
|------|-----------------|-------------------|
| 特征提取 | ~420ms | ~10-15ms |
| NMS | ~5ms | ~0.1ms |
| 描述子插值 | ~15ms | ~0.5ms |
| L2 匹配 | ~20ms | ~1ms |
| **总前端耗时** | **~460ms** | **~12-17ms** |
| 实时性 | 不可用（2fps） | **可用（60-80fps）** |

---

## 5. 当前状态与下一步

### 已完成

- [x] SuperPoint ONNX 模型导出（opset 18）
- [x] ORB-SLAM3 C++ 前端替换（CPU ONNX Runtime 验证通过）
- [x] TUM fr1_desk 数据集验证（RMSE 0.229m）
- [x] SP 词袋训练（SPvoc_v3.txt, 27458 words）

### 待完成

- [ ] Fine-tune 训练框架搭建（需要 GPU 服务器）
- [ ] NPU 友好 ONNX 导出（固定 shape + 去后处理）
- [ ] iCraft 量化编译
- [ ] ProciseAI PL 端 RTL 设计
- [ ] ProciseAI PS 端配置
- [ ] 交叉编译 + 板端部署

---

## 参考资源

- [手把手搭建复旦微FMQL100TAI900 AI推理原型](https://blog.csdn.net/weixin_28432777/article/details/159750120)
- [FMQL开发环境搭建全流程](https://blog.csdn.net/weixin_29193259/article/details/159194878)
- [Docker搭建FMQL30TAI交叉编译环境](https://wenku.csdn.net/doc/4iq73v8tvjp6)
- [从PC到嵌入式：AI模型移植全流程](https://wenku.csdn.net/column/1gy8jcpka0)
- [国产异构融合FPAI-FMQL30TAI芯片介绍](https://aihardware.csdn.net/69bcff100a2f6a37c598ec80.html)
- 复旦微电子官网：[fmsh.com](https://www.fmsh.com/)
