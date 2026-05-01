# SP-FPGA IP 核架构设计文档

## 1. 概述

本文档描述 SuperPoint 后处理 FPGA 加速 IP 核的架构设计，包括：
- 当前片上 BRAM 版本（功能验证通过）
- AXI 接口版本（面向 FMQL30TAI 部署）

### 1.1 数据流总览

```
NPU 输出               FPGA 后处理                    输出
─────────              ──────────                    ─────

semi logits ──► softmax_reshape ──► heatmap ──► NMS ──► keypoints
(65×60×80)     (softmax+reshape)   (H×W)     (top-k)   (N×3)

desc feature ──────────────────────────────► interp ──► L2 norm ──► descriptors
(256×60×80)                                (bilinear)  (normalize)  (N×256)
```

### 1.2 模块清单

| 模块 | 文件 | 功能 | 接口 |
|------|------|------|------|
| softmax_reshape | `softmax_reshape.v` | 65ch softmax + 8×8 reshape → 热力图 | AXI-S 输入, BRAM 写输出 |
| nms_topk | `nms_topk.v` | 滑动窗口 NMS + Top-K 选择 | BRAM 读输入, 寄存器输出 |
| desc_interp | `desc_interp.v` | 关键点坐标描述子双线性插值 | BRAM 随机读, AXI-S 输出 |
| l2_normalize | `l2_normalize.v` | 256 维向量 L2 归一化 | AXI-S 输入/输出 |
| sp_postprocess_top | `sp_postprocess_top.v` | 顶层控制状态机 | AXI-S 外部接口 |

### 1.3 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| IMG_W | 640 | 输入图像宽度 |
| IMG_H | 480 | 输入图像高度 |
| FEAT_W | 80 | 特征图宽度 (IMG_W/8) |
| FEAT_H | 60 | 特征图高度 (IMG_H/8) |
| DESC_DIM | 256 | 描述子维度 |
| MAX_KPTS | 2048 | 最大关键点数 |
| CONF_THRESH | 60 (Q12) | NMS 置信度阈值 |

---

## 2. 当前架构：片上 BRAM 版本

### 2.1 结构图

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  sp_postprocess_top                      │
                    │                                                         │
  DDR ──AXI-S──►  semi_tdata ──► ┌──────────────┐      ┌──────────┐          │
                              │  │softmax_reshape│─────►│heatmap   │◄────┐    │
                              │  │ (65ch→H×W)   │write │_bram     │     │    │
                              │  └──────────────┘      │307200×16b│     │    │
                              │                        └──────────┘     │    │
                              │                              │read     │    │
                              │                              ▼         │    │
                              │  控制                    ┌──────────┐  │    │
                              │  状态机                  │ nms_topk │  │    │
                              │  ┌───┐                   │(滑动窗口 │  │    │
                              │  │IDLE→SOFT→NMS→         │ +Top-K)  │  │    │
                              │  │  OUTPUT→INTERP→       └────┬─────┘  │    │
                              │  │  L2→DONE                  │        │    │
                              │  └───┘              kpt_x/y/score     │    │
                              │                        │    │         │    │
  DDR ──AXI-S──►  desc_tdata ──► ┌──────────┐         │    │         │    │
                              │  │desc_bram │◄────┐   │    │         │    │
                              │  │1228800×32│     │   │    │         │    │
                              │  └──────────┘     │   │    ▼         │    │
                              │       │read       │  ┌────────────┐  │    │
                              │       ▼           │  │desc_interp │  │    │
                              │  ┌──────────┐     │  │(双线性插值) │◄─┘    │
                              │  │l2_norm   │◄────┤  └────────────┘       │
                              │  │(L2归一化)│     │       │               │
                              │  └──────────┘     │       │               │
                              │       │           │       │               │
  DDR ◄──AXI-S──  kpt_tdata   │  ┌──┘     ┌─────┘       │               │
  DDR ◄──AXI-S──  desc_out    │  │        └─────────────┘               │
                              │  ▼                                        │
                              │  输出关键点坐标 → AXI-S → DDR             │
                              └──────────────────────────────────────────┘
```

### 2.2 控制状态机

```
IDLE ──start──► PHASE_SOFTMAX ──softmax_done──► PHASE_NMS ──nms_done──►
  │                                                             │
  │         ┌───nms_kpt_count==0──► DONE_ST                     │
  │         │                                                   │
  │         └───nms_kpt_count>0──► PHASE_OUTPUT ──► PHASE_INTERP ──+
  │                                                               │
  │         interp_start + l2_start (同时启动，L2 流水接收)        │
  │                                                               │
  ◄─── interp_done ──► PHASE_L2 ──l2_done──► DONE_ST ◄──────────┘
```

状态机特点：
- **PHASE_OUTPUT → PHASE_INTERP 时同时启动 L2**，实现 desc_interp → l2_normalize 的流水线数据传输
- **nms_kpt_count == 0 时直接跳到 DONE_ST**，避免 interp/L2 空转死锁
- **DONE_ST 持续拉高 done**，由外部拉低 start 复位

### 2.3 各模块详细设计

#### 2.3.1 softmax_reshape

```
输入: 串行 float32, 每 65 个为一组 (64 通道 + 1 dustbin)
输出: 写入 heatmap BRAM, Q12 定点

处理流程 (每个 cell):
  1. ACCUMULATE: 接收 65 个通道值
     - float → 定点: input_fixed = {1'b0, exp[3:0], mant[22:15]}
     - 查表: exp_val = exp_lut[input_fixed[15:8]]
     - 累加: exp_sum += exp_val
     - 缓存: exp_vals[ch] <= exp_val

  2. NORMALIZE: 输出 64 个通道
     - 概率: prob = (exp_vals[ch] << 12) / exp_sum
     - reshape: ch[i] → heatmap[cell_y*8 + i/8][cell_x*8 + i%8]
     - 写 BRAM: bram_addr, bram_wdata, bram_wen

资源: 256 entry exp_lut (ROM), 64 entry exp_vals (寄存器)
```

#### 2.3.2 nms_topk

```
输入: heatmap BRAM (H×W, Q12 定点)
输出: top-k 关键点坐标 + 分数

处理流程:
  1. SCAN: 逐像素扫描, 填充 9 行 line_buf
     - pipeline 写入: 地址当拍发, 数据下拍写 (wr_x/wr_y 延迟寄存器)
     - scan_y >= NMS_R*2 时转入 SCAN_FLUSH

  2. SCAN_FLUSH: 刷新最后 1 拍 pipeline 数据

  3. NMS_CHECK: 对有效区域像素做 NMS
     - 9×9 邻域求最大值 local_max
     - 边界像素 (x < BORDER 或 y < BORDER 等) 直接跳过

  4. HEAP_INSERT: 插入最小堆
     - center_val == local_max && center_val > CONF_THRESH → 关键点
     - heap_size < MAX_KPTS: 直接插入
     - heap_size == MAX_KPTS && center_val > min_heap_score: 替换最小值

  5. DONE_ST: 拷贝 heap → 输出寄存器, 拉高 done

资源: 9×640 line_buf (90Kb), 3×2048 heap (98Kb), 2048 比较器 (for 循环展开)
```

#### 2.3.3 desc_interp

```
输入: 关键点坐标 + desc BRAM
输出: AXI-S float32 描述子

处理流程 (每个关键点的每个通道):
  1. COMPUTE_COORD: 计算特征图坐标 x0=x/8, y0=y/8, 钳位到合法范围
  2. READ_4: 读取 4 个相邻点的 desc 值 (4 拍, 顺序地址)
  3. INTERPOLATE: 双线性插值 (当前简化为最近邻 val_00)
  4. OUTPUT: AXI-S 输出 interp_val
  5. NEXT_CH / NEXT_KPT: 循环直到所有关键点所有通道处理完
```

#### 2.3.4 l2_normalize

```
输入: AXI-S float32 (每 VEC_DIM 个为一组)
输出: AXI-S float32 (归一化后)

处理流程 (循环, 支持多向量):
  1. ACCUMULATE: 缓存 VEC_DIM 个值, 累加平方和 sum_sq
  2. DIVIDE: 逐个输出 buffer[idx]
     - 最后一个向量 (s_tlast 标记): 输出完 → IDLE + done
     - 非最后向量: 输出完 → 回到 ACCUMULATE 继续下一组

注意: 当前为直通模式 (未实现真正的归一化), TODO: 实现 buffer[idx] * inv_norm
```

### 2.4 已知限制

| 问题 | 影响 | 优先级 |
|------|------|--------|
| heatmap_bram = 4.7 Mbit | 超出 FMQL30TAI BRAM 容量 | **高** |
| desc_bram = 37.5 Mbit | 远超任何 FPGA 片上 BRAM | **高** |
| NMS for 循环展开 2048 | 综合面积巨大,时序可能不收敛 | 中 |
| softmax float-to-fixed 只寻址 LUT 0-15 | exp 值全零,heatmap 输出为 0 | 中 |
| desc_interp 最近邻替代双线性 | 描述子精度损失 | 低 |
| l2_normalize 直通未归一化 | 描述子未归一化 | 低 |

### 2.5 测试状态

| 测试 | 文件 | 参数 | 结果 |
|------|------|------|------|
| softmax_reshape | `tb_softmax_reshape.v` | 4×3 cell → 768 写入 | PASS |
| nms_topk | `tb_nms_topk.v` | 32×24, MAX_KPTS=16, 找到 16 点 | PASS |
| desc_interp | `tb_desc_interp.v` | 2 kpts × 4D → 8 输出 | PASS |
| l2_normalize | `tb_l2_normalize.v` | 4 值 → 4 值 | PASS |
| 顶层集成 | `tb_top_integration.v` | 32×24 全流水线, 8 kpts | PASS |

---

## 3. 目标架构：AXI 接口版本

### 3.1 设计原则

1. **大数组留 DDR，FPGA 只做计算** — heatmap/desc 数据在 DDR，按需 DMA 搬运
2. **片上 BRAM 仅用于行缓存和堆** — 从 42.4 Mbit 降至 ~0.19 Mbit
3. **标准 AXI 接口** — 兼容 Procise IP 集成器
4. **PS 配置，PL 计算** — ARM 配参数启停，FPGA 做加速

### 3.2 整体结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FPGA (PL)                                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 sp_postprocess_top (AXI 版)                  │   │
│  │                                                             │   │
│  │  ┌─────────┐                                                │   │
│  │  │AXI-Lite │◄── PS 配置 (start/addr/thresh/done/nkpts)      │   │
│  │  │Slave    │                                                │   │
│  │  └────┬────┘                                                │   │
│  │       │ 寄存器                                              │   │
│  │       ▼                                                     │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │                   控制状态机                           │   │   │
│  │  └──┬─────────┬──────────┬──────────┬──────────────┬───┘   │   │
│  │     │         │          │          │              │        │   │
│  │     ▼         ▼          ▼          ▼              ▼        │   │
│  │  ┌──────┐ ┌──────┐  ┌──────┐  ┌──────┐     ┌──────┐      │   │
│  │  │DMA R │ │DMA R │  │DMA R │  │DMA W │     │DMA W │      │   │
│  │  │semi  │ │heat  │  │desc  │  │kpt   │     │desc  │      │   │
│  │  └──┬───┘ └──┬───┘  └──┬───┘  └──┬───┘     └──┬───┘      │   │
│  │     │        │         │         │             │           │   │
│  │     ▼        ▼         ▼         ▼             ▼           │   │
│  │  ┌──────┐ ┌──────┐  ┌──────┐  ┌──────┐   ┌──────┐        │   │
│  │  │softm.│ │NMS   │  │interp│  │kpt   │   │L2    │        │   │
│  │  │ax.s  │ │+topk │  │      │  │out   │   │norm  │        │   │
│  │  └──────┘ └──────┘  └──────┘  └──────┘   └──────┘        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │ AXI4 Master                        │
│                              ▼                                     │
│                     ┌──────────────────┐                           │
│                     │ AXI Interconnect  │                           │
│                     └────────┬─────────┘                           │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   DDR (256MB~1GB)    │
                    │                      │
                    │ 0x3000_0000 semi     │  NPU → DDR (65×60×80 f32)
                    │ 0x3500_0000 desc     │  NPU → DDR (256×60×80 f32)
                    │ 0x3800_0000 heatmap  │  FPGA 中间结果 (480×640 Q16)
                    │ 0x4000_0000 kpts     │  FPGA 输出 (N×48bit)
                    │ 0x4100_0000 desc_out │  FPGA 输出 (N×256 f32)
                    └─────────────────────┘
```

### 3.3 接口定义

```verilog
module sp_postprocess_top_axi (
    // 时钟复位
    input  logic        clk,
    input  logic        rst_n,

    // AXI-Lite Slave: PS 配置寄存器
    axi_lite_slave_if   s_axi_lite,

    // AXI4 Master: PL 读写 DDR
    axi4_master_if      m_axi4,

    // 中断: 处理完成
    output logic        irq
);
```

### 3.4 寄存器映射 (AXI-Lite)

| 偏移 | 名称 | R/W | 说明 |
|------|------|-----|------|
| 0x00 | REG_CTRL | W | [0]=start, [1]=reset |
| 0x04 | REG_STATUS | R | [0]=done, [1]=busy, [7:2]=reserved |
| 0x08 | REG_SEMI_ADDR | R/W | DDR semi 起始地址 |
| 0x0C | REG_DESC_ADDR | R/W | DDR desc 起始地址 |
| 0x10 | REG_HEATMAP_ADDR | R/W | DDR heatmap 起始地址 (中间结果) |
| 0x14 | REG_KPT_ADDR | R/W | DDR 关键点输出地址 |
| 0x18 | REG_DESC_OUT_ADDR | R/W | DDR 描述子输出地址 |
| 0x1C | REG_NUM_KPTS | R | 检测到的关键点数量 |
| 0x20 | REG_CONF_THRESH | R/W | NMS 置信度阈值 |
| 0x24 | REG_MAX_KPTS | R/W | 最大关键点数 |

### 3.5 AXI 版片上 BRAM 用量

| 模块 | BRAM 用途 | 大小 |
|------|----------|------|
| softmax | 乒乓写 buffer (1 行特征) | ~1.3 Kb |
| NMS | line_buf (9 行 × IMG_W) | 90 Kb |
| NMS | heap (MAX_KPTS × 48bit) | 98 Kb |
| **合计** | | **~190 Kb = 0.19 Mbit** |

对比当前版本的 42.4 Mbit，降低 **99.6%**，完全可放入 FMQL30TAI 的 ~5.3 Mbit BRAM。

### 3.6 各子模块改动分析

#### softmax_reshape — 改动最小

```
输入: 不变 (AXI-S 收 semi 数据, 由 DMA 从 DDR 搬入)
输出: 改为 AXI4 Master 写 heatmap 到 DDR

选项 A: softmax 输出直接写 DDR (每算完一个 cell 就写)
选项 B: softmax 输出写片上小 BRAM, 完成后 DMA 一次性搬 DDR

推荐选项 A: 按行写, 减少片上 BRAM
```

#### nms_topk — 改动最大

```
核心变化: heatmap 读接口从片上 BRAM 组合读 → AXI4 Master DDR 读

新增: 行 DMA 引擎
  - 预取: DMA 提前读取接下来 N 行的 heatmap 数据到 line_buf
  - 流水: DMA 填充第 N+1 行时, NMS 处理第 N 行
  - 地址计算: base_addr + row * IMG_W * 2 (Q16, 每像素 2 字节)

NMS 处理流程:
  SCAN 阶段不再需要 (DMA 按行搬运)
  直接从 line_buf 做 NMS_CHECK + HEAP_INSERT
```

#### desc_interp — 改动中等

```
核心变化: desc_bram 随机读 → AXI4 Master DDR 随机读

挑战: AXI 读有延迟 (非组合读)
  - 原版: desc_bram_rdata = desc_bram[addr] (组合, 0 延迟)
  - AXI 版: 发读请求 → 等 ARREADY → 等 RVALID → 收数据 (3-10 拍延迟)

解决方案: 读请求 FIFO + 数据缓冲
  - READ_4 改为: 先发 4 个 AXI 读请求, 再等 4 个响应
  - 或: 使用 Procise 内置 AXI DMA 做 scatter-read
```

#### l2_normalize — 改动最小

```
输入/输出已经是 AXI-S, 不需要改
直接挂在 desc_interp 输出和 DMA 写 DDR 之间
```

---

## 4. Procise 集成方案

### 4.1 FMQL30TAI 平台特性

```
FMQL30TAI = PS(4×Cortex-A) + NPU(8TOPS INT8) + FPGA PL(125K LC)

PS 端:
  - Linux 操作系统
  - iCraft Runtime: 加载 NPU 模型, 执行推理
  - /dev/mem mmap: 直接访问 DDR 物理地址
  - UIO / devmem: 配置 FPGA 寄存器

PL 端:
  - BRAM: ~5.3 Mbit
  - DSP: 乘法器 (可做定点/浮点运算)
  - AXI 互联: PS↔PL, PL↔DDR
```

### 4.2 Procise Block Design 连线

```
┌──────────┐  AXI-Lite   ┌──────────────────────────┐
│  PS7     │──────────────│ s_axi_lite (寄存器配置)    │
│  (ARM)   │              │                          │
│          │  AXI         │ sp_postprocess_top_axi    │
│          │◄════════════►│                          │
│          │              │ m_axi4 (读写 DDR)         │──► DDR
│          │  中断        │                          │
│          │◄─────────────│ irq                      │
└──────────┘              └──────────────────────────┘
```

### 4.3 PS 端软件流程 (C 代码)

```c
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>

#define FPGA_BASE       0x40000000  // PL 寄存器基地址 (根据实际分配)
#define REG_CTRL        0x00
#define REG_STATUS      0x04
#define REG_SEMI_ADDR   0x08
#define REG_DESC_ADDR   0x0C
#define REG_HEAT_ADDR   0x10
#define REG_KPT_ADDR    0x14
#define REG_DESC_OUT    0x18
#define REG_NUM_KPTS    0x1C
#define REG_THRESH      0x20

// 1. NPU 推理: semi + desc → DDR
void run_npu(icraft_ctx *ctx, float *image,
             uint32_t semi_ddr, uint32_t desc_ddr)
{
    icart_run(ctx, image, semi_ddr, desc_ddr);
}

// 2. 配置 FPGA 寄存器并启动
void start_fpga(volatile uint32_t *fpga,
                uint32_t semi_addr,  uint32_t desc_addr,
                uint32_t heat_addr,  uint32_t kpt_addr,
                uint32_t desc_out_addr, uint32_t thresh)
{
    fpga[REG_SEMI_ADDR  >> 2] = semi_addr;
    fpga[REG_DESC_ADDR  >> 2] = desc_addr;
    fpga[REG_HEAT_ADDR  >> 2] = heat_addr;
    fpga[REG_KPT_ADDR   >> 2] = kpt_addr;
    fpga[REG_DESC_OUT   >> 2] = desc_out_addr;
    fpga[REG_THRESH     >> 2] = thresh;
    fpga[REG_CTRL       >> 2] = 0x1;  // start
}

// 3. 等待完成并读取结果
int wait_and_read(volatile uint32_t *fpga,
                  uint32_t kpt_addr, uint32_t desc_out_addr)
{
    // 轮询 done (也可用中断)
    while (!(fpga[REG_STATUS >> 2] & 0x1));

    int nkpts = fpga[REG_NUM_KPTS >> 2];
    printf("Detected %d keypoints\n", nkpts);

    // 从 DDR 读取结果
    // keypoint 格式: {x[15:0], y[15:0]} per word, nkpts words
    // desc 格式: 256 × float32 per keypoint
    return nkpts;
}

// 主函数
int main()
{
    int mem_fd = open("/dev/mem", O_RDWR);
    volatile uint32_t *fpga = mmap(NULL, 4096,
                          PROT_READ|PROT_WRITE, MAP_SHARED,
                          mem_fd, FPGA_BASE);

    // DDR 地址规划
    uint32_t semi_ddr    = 0x30000000;
    uint32_t desc_ddr    = 0x35000000;
    uint32_t heat_ddr    = 0x38000000;
    uint32_t kpt_ddr     = 0x40000000;
    uint32_t desc_out_ddr = 0x41000000;

    // Step 1: NPU 推理
    run_npu(ctx, image, semi_ddr, desc_ddr);

    // Step 2: FPGA 后处理
    start_fpga(fpga, semi_ddr, desc_ddr, heat_ddr,
               kpt_ddr, desc_out_ddr, 60);

    // Step 3: 等待并读结果
    int nkpts = wait_and_read(fpga, kpt_ddr, desc_out_ddr);

    close(mem_fd);
    return 0;
}
```

### 4.4 DDR 地址规划

```
0x3000_0000 ┌─────────────────┐
            │  semi logits     │  65 × 60 × 80 × 4 = 1,248,000 字节 (~1.2MB)
            │  (NPU 输出)      │
0x3500_0000 ├─────────────────┤
            │  desc features   │  256 × 60 × 80 × 4 = 4,915,200 字节 (~4.7MB)
            │  (NPU 输出)      │
0x3800_0000 ├─────────────────┤
            │  heatmap         │  480 × 640 × 2 = 614,400 字节 (~0.6MB)
            │  (FPGA 中间结果)  │
0x4000_0000 ├─────────────────┤
            │  keypoints       │  N × 6 字节 (x:u16, y:u16, score:u16)
            │  (FPGA 输出)     │  最大 2048 × 6 = 12,288 字节
0x4100_0000 ├─────────────────┤
            │  descriptors     │  N × 256 × 4 字节
            │  (FPGA 输出)     │  最大 2048 × 1024 = 2,097,152 字节 (~2MB)
            └─────────────────┘
```

---

## 5. 两个版本对比

| 维度 | 当前 BRAM 版 | AXI 版 |
|------|-------------|--------|
| **片上 BRAM** | 42.4 Mbit (爆) | 0.19 Mbit (放得下) |
| **可综合性** | 不能 (BRAM 不够) | 可以 |
| **Procise 集成** | 不行 | 标准做法 |
| **数据来源** | AXI-S 直灌 | AXI DMA 读 DDR |
| **NPU 对接** | 需要外部搬运 | NPU 输出直接在 DDR |
| **CPU 可控性** | 仅 start/done | 完整寄存器配置 |
| **开发状态** | 功能验证通过 | 待开发 |

---

## 6. 实施建议

### Phase 1: 小模块独立验证 (当前可做)

将 softmax_reshape、l2_normalize、desc_interp 各自封装为 AXI-S 接口 IP 核，在 Procise 中独立验证：
- 这三个模块内部没有大数组，可直接综合
- 用 Procise 内置 AXI-S 验证 IP (如 Data Generator + Data Checker) 测试

### Phase 2: NMS 重构 (核心工作)

- 将 nms_topk 的 heatmap 读取改为 AXI4 Master
- 添加行 DMA 预取引擎
- 优化 for 循环: 用流水线比较或 BRAM-based 堆替代 2048 展开比较器

### Phase 3: 顶层 AXI 集成

- 实现 AXI-Lite 寄存器模块
- 实现 AXI4 Master 读写引擎 (或使用 Procise 内置 DMA IP)
- 集成测试: PS 端 C 代码 + PL 端 Verilog

### Phase 4: NPU 联调

- iCraft 部署 SuperPoint ONNX 模型到 NPU
- NPU 输出 semi/desc 到 DDR 指定地址
- FPGA 从 DDR 读取并后处理
- 端到端验证关键点检测精度
