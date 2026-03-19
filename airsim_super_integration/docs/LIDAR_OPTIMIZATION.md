# LiDAR点云优化指南

## 🎯 问题：点云稀疏、距离远

### 原因分析

1. **AirSim LiDAR参数配置不当**
2. **RViz显示设置问题**
3. **点云过滤导致点数减少**

---

## 🔧 解决方案

### 方案1：高密度配置（推荐）⭐

适合密集障碍物环境，如ModularPark：

```json
"Lidar1": {
  "SensorType": 6,
  "Enabled": true,
  "NumberOfChannels": 64,        // 64线LiDAR（高端配置）
  "RotationsPerSecond": 20,      // 20Hz更新
  "PointsPerSecond": 300000,     // 30万点/秒
  "MaxDistance": 5000,           // 50米（适合城市环境）
  "VerticalFOVUpper": 30,        // 上30度
  "VerticalFOVLower": -30,       // 下30度（共60度）
  "HorizontalFOVStart": 0,
  "HorizontalFOVEnd": 359,
  "DrawDebugPoints": true
}
```

**适用场景**：
- ✅ 城市环境（楼宇、街道）
- ✅ 室内环境
- ✅ 密集障碍物

---

### 方案2：超高密度配置（性能强）

适合高性能电脑：

```json
"Lidar1": {
  "NumberOfChannels": 128,       // 128线（超高密度）
  "RotationsPerSecond": 20,
  "PointsPerSecond": 500000,     // 50万点/秒
  "MaxDistance": 3000,           // 30米（更集中）
  "VerticalFOVUpper": 45,        // 上45度
  "VerticalFOVLower": -45        // 下45度（共90度）
}
```

**优点**：超密集点云
**缺点**：CPU占用高

---

### 方案3：平衡配置

适合性能一般的电脑：

```json
"Lidar1": {
  "NumberOfChannels": 32,        // 32线（中等密度）
  "RotationsPerSecond": 10,
  "PointsPerSecond": 200000,     // 20万点/秒
  "MaxDistance": 5000,           // 50米
  "VerticalFOVUpper": 20,
  "VerticalFOVLower": -20        // 40度视场
}
```

---

## 🎨 RViz显示优化

### 调整点云大小

**位置**：RViz左侧 → AirSim点云(实时) → Size (m)

| 场景 | 推荐大小 | 效果 |
|------|---------|------|
| 远距离观察 | 0.03-0.05 | 清晰可见 |
| 中等距离 | 0.05-0.08 | 较为密集 |
| 近距离 | 0.08-0.15 | 非常密集 |

### 调整颜色方案

**推荐配置**：

1. **Intensity**（强度）- 默认，适合看细节
2. **AxisColor**（轴颜色）- 适合看方向
   - X=红，Y=绿，Z=蓝
3. **FlatColor**（单色）- 简洁
   - 白色：最清晰
   - 黄色：醒目
   - 青色：护眼

### QoS设置

确保设置为：
- **Reliability**: Best Effort
- **History**: Keep Last (10-20)
- **Durability**: Volatile

---

## 📊 参数详解

### NumberOfChannels（扫描线数）

| 线数 | 类型 | 密度 | 性能 |
|------|------|------|------|
| 16 | 低端 | 稀疏 | 低 |
| 32 | 中端 | 适中 | 中 |
| 64 | 高端 | 密集 | 高 |
| 128 | 顶级 | 超密集 | 很高 |

**推荐**：64线（性能与效果平衡）

### PointsPerSecond（点云密度）

| 点数/秒 | 密度 | CPU占用 |
|---------|------|---------|
| 100k | 低 | 低 |
| 200k | 中 | 中 |
| 300k | 高 | 高 ⭐ |
| 500k+ | 超高 | 很高 |

**推荐**：300k（密集且流畅）

### MaxDistance（最大距离）

| 距离 | 说明 | 适用场景 |
|------|------|---------|
| 3000 (30m) | 近 | 室内、密集障碍物 |
| 5000 (50m) | 中 | 城市环境 ⭐ |
| 10000 (100m) | 远 | 开阔地带 |

**注意**：距离越远，点云越稀疏！

### VerticalFOV（垂直视场角）

| 范围 | 总角度 | 适用 |
|------|--------|------|
| ±15° | 30° | 狭窄（看前方） |
| ±20° | 40° | 适中 |
| ±30° | 60° | 宽广（看上下）⭐ |
| ±45° | 90° | 超宽（全方位） |

**推荐**：±30°（覆盖上下障碍物）

---

## 🔍 故障排查

### 问题1：点云仍然稀疏

**检查**：
```bash
ros2 topic hz /cloud_registered
```

**解决**：
- ✅ 确认频率在15-20 Hz
- ✅ 增加 PointsPerSecond
- ✅ 增加 NumberOfChannels
- ✅ 减小 MaxDistance

### 问题2：点云有但看不见

**检查**：
```bash
ros2 topic echo /cloud_registered --no-arr | head -30
```

**解决**：
- ✅ 增大RViz中的Size
- ✅ 改变Color Transformer
- ✅ 检查Fixed Frame是否为"world"

### 问题3：性能下降/卡顿

**解决**：
- ⬇️ 减少 PointsPerSecond
- ⬇️ 减少 NumberOfChannels
- ⬇️ 减小 VerticalFOV
- ⬇️ 降低 RotationsPerSecond

---

## 🚀 快速配置模板

### 推荐配置（已应用）

当前系统使用：
- 64线
- 300k点/秒
- 50米范围
- 60度视场

**效果**：密集、清晰、性能好 ✅

---

## 📝 应用配置的步骤

1. 修改 `/home/wuyou/Documents/AirSim/settings.json`
2. 重启AirSim环境
3. 重启桥接节点
4. 在RViz中调整显示参数
5. 观察效果

---

## 💡 高级技巧

### 动态调整点云大小

根据无人机高度自动调整：
- 低空（<5m）：大点（0.08-0.15）
- 中空（5-15m）：中点（0.05-0.08）
- 高空（>15m）：小点（0.03-0.05）

### 多LiDAR配置

如需更密集的点云，可以配置多个LiDAR：

```json
"Sensors": {
  "Lidar1": {
    "Z": -0.5,
    "Pitch": 0
  },
  "Lidar2": {
    "Z": -0.5,
    "Pitch": -30  // 向下倾斜
  }
}
```

---

**优化后点云应该更密集、更清晰！** 🎉


