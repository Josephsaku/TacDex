# 离线数据契约 v2

本仓库处理采集完成后的文件。SDK 消息解码、MuJoCo 碰撞求解、RBF 和逐点深度的生成均由上游负责。接入时先提取单片 ID3，统一坐标和单位，再导出一个 episode 对应的 NPZ。严禁把其他手指的合力合并进 ID3。

## 必需数组

| 字段 | 形状 | 含义 |
|---|---|---|
| `metadata` | 标量 JSON 字符串 | 下述来源与配对信息 |
| `geometry` | `[N,6]` | taxel 的 xyz(m) 和单位法向，保持相同顺序 |
| `time_s` | `[T]` | 严格递增的共同时间轴，秒 |
| `sim_depth_m` | `[T,N]` | 逐 taxel 非负接触深度，米 |
| `sim_force_N` | `[T,3]` | MuJoCo 直接输出、作用在 ID3 上的接触合力，N |
| `sim_mask` | `[N]` bool | 有效深度 taxel |
| `frame_valid` | `[T]` bool | 当前帧是否可用于构造历史窗口 |
| `stimulus_id` | `[T]` 字符串 | 同一物理刺激的标识，不等同于任意帧序号 |

配对训练和真机测试另外要求：

| 字段 | 形状 | 含义 |
|---|---|---|
| `real_taxel_N` | `[T,N,3]` | 真机实际测得的每个 taxel 三轴力 |
| `real_force_N` | `[T,3]` | SDK 实际输出的 ID3 三轴合力 |
| `real_mask` | `[N]` bool | 有效真机 taxel |
| `raw_event_id` | `[T]` 整数 | 独立触觉包编号，不能重复 |
| `pair_error_s` | `[T]` | 选用仿真帧相对真机触觉帧的带符号时间差 |

`downstream_train` 和 `downstream_val` 允许完全没有真机字段，仍必须有 `sim_force_N`。缺失三轴合力不能用零向量替代；真实无接触时实际测得的零向量则是合法数据。

## 元数据示例

```json
{
  "schema_version": 2,
  "episode_id": "trial_001",
  "group_id": "trajectory_001",
  "location_id": "probe_location_001",
  "geometry_id": "id3_calibration_v1",
  "sensor_id": 3,
  "units": "m,N,s",
  "force_frame": "sensor_local",
  "depth_method": "taxel_normal_raycast",
  "sim_force_source": "mujoco_contact_solver",
  "synthetic": false,
  "pairing_verified": true,
  "axis_calibrated": true,
  "pair_tolerance_s": 0.02,
  "max_contact_mismatch_fraction": 0.1
}
```

`pairing_verified` 和 `axis_calibrated` 必须由实际标定与验证支持，不能为了通过校验直接填 true。程序验证数值和契约，不能仅凭一个来源标签证明物理采集正确。建议额外保存模型与原始文件哈希、MuJoCo 版本、solver 参数、传感器 geom 名单、taxel 排列标定版本及 SDK 单位换算。

深度来源支持：

- `taxel_normal_raycast`：上游在每个 taxel 表面法向采样得到的几何深度。
- `taxel_collision_depth`：上游建立了逐 taxel 碰撞几何及明确映射后保存的接触深度。

MuJoCo 的 `contact.dist` 属于碰撞接触点，不能无映射地解释为全部 taxel 深度。此仓库不把接触点距离扩散成深度图。

正式仿真合力要求 `sim_force_source=mujoco_contact_solver`。它应是 MuJoCo 求解得到的 ID3 外部接触力向量和：先从每个接触坐标系变换到共同坐标系，确认“作用在传感器上”的正负号，再变换到 ID3 局部坐标。不要使用各接触力模长之和、机械臂总力、含其他手指的整体力，或未经核对的 site 内力代替。

已有 replay 若通过 `module[idx] += fs` 直接累加求解器接触力，同时另行 RBF 分配 taxel 力，则前者可以使用。离线导出常见选择为 `module_force_sensor_N[:, module_ids.index(3), :]`；必须依据保存的模块 ID 顺序定位，不能无条件假设数组第 3 项就是 ID3。

## 配对、历史与数据划分

100 Hz 机械臂行可能重复引用 10 Hz 触觉包。先按原始触觉包去重，再建立配对；10 帧历史意味着 10 个独立触觉包。窗口只读取当前及过去帧，不跨 episode、无效帧或过大的时间间隔；默认最大间隔 0.25 s，可调整。

只有时间对齐不够：压头形状、相对位置、接触深度、运动状态也要对应。真机 ID3 有接触而仿真仅其他指尖接触的记录不能当作正样本。接触一致性阈值是数据诊断，不是声明两域力值必然相等。

manifest 示例：

```json
{
  "schema_version": 2,
  "episodes": [
    {"path": "align_001.npz", "split": "align_train"},
    {"path": "align_002.npz", "split": "align_val"},
    {"path": "sim_003.npz", "split": "downstream_train"},
    {"path": "sim_004.npz", "split": "downstream_val"},
    {"path": "paired_005.npz", "split": "test_id"}
  ]
}
```

也可定义 `test_ood_size`、`test_ood_shape`。轨迹、位置分组和物理刺激不得跨分区重复。OOD 的形状／尺寸定义需要采集者实际保证；程序不会从标签名称推断是否真正 OOD。归一化只拟合 `align_train`，下游和测试复用 checkpoint 中的尺度。

`synthetic` 命令生成的文件必须保持 `synthetic=true`，对应 `analytic_synthetic`／`synthetic_fixture`，仅用于程序测试，不能用于报告实测迁移性能。
