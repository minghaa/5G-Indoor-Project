# 3GPP Indoor Hotspot Simulation - PPO Power Control


## Tóm Tắt Bài Toán

**Môi trường**: 3GPP Indoor Hotspot (12 Small Cells, 120 UEs, 120m×50m)

**Vấn đề cần giải quyết**:
- Phương pháp On/Off trạm gây hiệu ứng ping-pong và vi phạm QoS
- Cần học chính sách điều khiển công suất "mềm" và liên tục

**Giải pháp**: Sử dụng **PPO (Actor-Critic)** để học điều chỉnh công suất phát liên tục

---

## Kết Quả Đạt Được

### So Sánh Các Policy

| Policy | Reward | Drop Rate | Latency | Energy Saved | QoS Violations |
|--------|--------|-----------|---------|--------------|----------------|
| Full Power (Baseline) | 0.00 | 0.51% | 22ms | 0% | - |
| Half Power | 7.45 | 0.51% | 25ms | 25% | - |
| Random | -1169.5 | 0.20% | 51ms | 31% | Many |
| **PPO Agent** | **14.25** | **0.00%** | **35ms** | **53%** | **0** ✅ |

### Kết Luận
- ✅ **Drop Rate = 0%** (thấp nhất, đạt ngưỡng ≤5%)
- ✅ **Latency = 35ms** (đạt ngưỡng ≤50ms)
- ✅ **Tiết kiệm 53% năng lượng** (gấp đôi Half Power)
- ✅ **0 vi phạm QoS** trong toàn bộ episode
- ✅ **Không dừng episode khi violation** - agent học được cách recovery

---

## Kiến Trúc Môi Trường

### State Space (173 chiều)
Theo đúng DataDescription.xlsx:
- **Simulation Features (17)**: totalCells, simTime, minTxPower, maxTxPower, thresholds...
- **Network Features (12)**: totalEnergy, avgDropRate, avgLatency, connectedUEs...
- **Cell Features (12×12=144)**: cpuUsage, prbUsage, txPower, avgSINR mỗi cell...

### Action Space
- **Continuous Box(12,)** ∈ [0, 1]
- **Mapping**: `TxPower = ratio × (maxTxPower - minTxPower) + minTxPower`
- **< 0.1**: Cell vào chế độ Sleep

### Reward Function
```
R = R_energy - R_qos_drop - R_qos_latency - R_stability
```
**QoS Priority**: Trọng số penalty QoS lớn hơn nhiều so với reward energy

---

## Files Nộp

| File | Mô tả |
|------|-------|
| `cells_layout.json` | Tọa độ 12 cells (2×6 grid, ISD=20m) |
| `ue_mobility_trace.npy` | Pre-generated mobility (300, 120, 3) |
| `simulator_v2.py` | Simulator với 173-dim state |
| `multi_cell_env_v2.py` | Gymnasium Environment |
| `train_ppo_v2.py` | Script training PPO |
| `ppo_multicell_v2.zip` | Model đã train |
| `training_results_v2.png` | Visualization kết quả |

---

## Hướng Dẫn Chạy

```bash
# Activate environment
cd "/Users/tominhhaxinhdep/Downloads/5G Project/MultiCell"
source ../rl_env/bin/activate

# Test environment
python simulator_v2.py
python multi_cell_env_v2.py

# Train model
python train_ppo_v2.py

# Load trained model
from stable_baselines3 import PPO
model = PPO.load("ppo_multicell_v2")
```

---

## Training Curve

![Training Results](training_results_v2.png)

---

## Ghi Chú Kỹ Thuật

1. **Episode không dừng khi QoS violation**: Agent học được cách recovery (theo yêu cầu mentor)

2. **QoS Priority**: 
   - Drop penalty weight: 100.0
   - Latency penalty weight: 20.0
   - Energy reward weight: 0.1 (nhỏ hơn nhiều)

3. **Latency Model**: Base 3ms + queue delay (tối đa 35ms) + power penalty khi công suất < 70%

4. **Mobility**: Pre-generated Random Walk (1m/step), reproducible với seed
