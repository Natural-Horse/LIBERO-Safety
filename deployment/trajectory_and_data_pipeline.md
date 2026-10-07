# LIBERO-Safety 的轨迹设计与数据采集流程

本文依据 [LIBERO-Safety 论文](https://arxiv.org/html/2606.23686v2)、[官方仓库说明](https://github.com/LIBERO-SAFETY/LIBERO-Safety) 和本机源码，解释任务如何定义、轨迹如何设计、数据如何落盘。**要区分两条流程**：论文中的“关键位姿 + CuRobo”自动生成器尚未在官方仓库开源；目前仓库能直接运行的是交互式示教采集、MuJoCo 状态记录和 HDF5 重放转换。本机的 `deployment/smoke_pipeline.py` 只用于验证后一条流程的接口与数据格式。

## 1. 整体设计：先定义安全问题，再产生轨迹

LIBERO-Safety 并非给每个任务指定一条固定路径。一个任务定义目标、初始布局和安全边界；在这些边界内，可以有多条有效轨迹。论文将参数化任务写成

$$\mathcal T=\langle \mathcal G,\mathcal C_{\mathrm{safety}},\mathcal S_{\mathrm{init}},\mathcal P_{\mathrm{env}}\rangle,$$

分别对应**目标条件**、**每个时刻的安全约束**、**机器人/物体初始状态及动态实体轨迹**、**相机/视觉/噪声等环境参数**。相比只描述起点和终点的普通 BDDL，UBDDL 还可以描述动态障碍和接触限制。因此“拿起物体”只解决任务目标，“全程不碰到旁边的人手或障碍”是独立的安全要求。[论文第 3.1 节与附录 0.A](https://arxiv.org/html/2606.23686v2)

```text
任务定义（UBDDL/BDDL） + 资产 + 初始状态
                    │
                    ▼
         MuJoCo 场景与安全约束
                    │
       ┌────────────┴──────────────────┐
       │                               │
       ▼                               ▼
论文中的自动生成链路                 当前开源的人工采集链路
物体系稀疏关键位姿                    键盘 / SpaceMouse 操作
→ 位姿变体采样                        → 逐步记录状态和动作
→ 坐标变换与 CuRobo 规划               → 原始 NPZ + model.xml
→ 碰撞检查 / 人工筛选                  → 聚合 demo.hdf5
→ 安全示教数据集                       → 重放并生成带图像的 HDF5
```

## 2. 主要工作空间与坐标系

这里的“工作空间”既指机器人实际操作的物理场景，也指轨迹规划/观测使用的坐标系。它们不能混为一个固定的桌面边界。

| 层次 | 作用 | 本仓库中的位置或表现 |
| --- | --- | --- |
| 任务/场景空间 | 摆放目标物、容器、桌面障碍、人手与运动实体；定义 `region`、初始条件和目标条件 | [`bddl_files/`](../libero/libero/bddl_files/)、[`assets/`](../libero/libero/assets/)；仿真创建见 [`bddl_base_domain.py`](../libero/libero/envs/bddl_base_domain.py) |
| 物体规范坐标系 | 专家相对**目标物体**标注抓取等关键位姿；物体平移或旋转后仍可复用操作意图 | 论文第 3.3 节描述；对应自动生成器代码未公开 |
| 机器人基座坐标系 | 将关键位姿和障碍几何统一变换到此坐标系，供 CuRobo 规划；末端执行器轨迹也可在此空间分析 | 论文第 3.3 节；自动生成器代码未公开 |
| MuJoCo 世界/关节状态 | 执行控制并记录可重放的完整仿真状态、动作与场景 XML | [`DataCollectionWrapper`](../third_party/robosuite-1.4/robosuite/wrappers/data_collection_wrapper.py)、[`collect_demonstration.py`](../scripts/collect_demonstration.py) |
| 相机观测空间 | 为 VLA 提供第三人称与腕部视角图像，并可加入视点和视觉扰动 | `agentview`、`robot0_eye_in_hand`；见 [`env_wrapper.py`](../libero/libero/envs/env_wrapper.py) 和 [`create_dataset.py`](../scripts/create_dataset.py) |

具体物理布局随任务变化：仓库有桌面、厨房、书房、客厅、茶几及地面等 arena；桌面大小在不同环境类中分别设置，不能把 `BDDLBaseDomain` 的默认桌面尺寸视为所有任务的统一可达空间。默认封装使用 Panda 机械臂、`OSC_POSE` 控制、20 Hz 控制频率和上述两路相机；这些是**本仓库默认值**，不是论文自动生成器的完整参数表。[`env_wrapper.py`](../libero/libero/envs/env_wrapper.py)

从实现角度看，物体系适合表达“怎么抓、怎么放”，基座系适合求解机器人运动和障碍碰撞，世界系负责仿真与精确重放，相机系负责视觉模型输入。这种分层让同一操作意图可以适配不同物体位置和场景布局。[论文第 3.3 节](https://arxiv.org/html/2606.23686v2)

## 3. 任务如何塑造轨迹

五个 suite 各有 L0/L1/L2 三档，每档五个任务，总计 75 个 benchmark 任务。前四类是物理执行任务，最后一类是静态图像和语言指令上的语义安全判断。[论文第 3.2、4.1 节](https://arxiv.org/html/2606.23686v2)

| Suite（目录） | 要求轨迹解决的问题 | 难度推进示例 |
| --- | --- | --- |
| AAG：`affordance` | 识别物体可安全接触的部位并形成合适抓取姿态 | 规范朝向 → 平移扰动 → 分布外朝向 |
| HRI：`human_safety` | 与人类代理处于共享工作空间时安全操作 | 标准情形 → 人手/物体位置扰动 → 指令改写 |
| TSA：`obstacle_avoidance` | 绕过桌面物体，避免机械臂或所持物体碰撞 | 熟悉静态障碍 → 动态障碍 → 新几何/布局 |
| FSHOA：`obstacle_avoidance_human` | 在三维自由空间避让“手持物体”组合体 | 静态组合体 → 运动组合体 → 未见过的手持姿态/物体 |
| SSR：`reasoning_safety` | 判断指令是否危险并拒绝执行 | 明确恶意指令 → 物理风险 → 隐蔽语义陷阱 |

任务文件位于 [`libero/libero/bddl_files/`](../libero/libero/bddl_files/) 的 `suite/L0|L1|L2/*.bddl`。一个典型定义包含 `:language`、`:regions`、`:fixtures`、`:objects`、`:init` 和 `:goal`；安全任务还可能有 `:dynamics` 与 `:constraints`。例如[将书放进收纳盒的 FSHOA L1 任务](../libero/libero/bddl_files/obstacle_avoidance_human/L1/pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl)指定了携带物体的人手按 `linear` 方式运动，并以 `CheckRobotContact`、`CheckContact` 描述接触约束。`region` 定义可采样的摆放范围，`:goal` 描述完成条件，`:dynamics` 改变执行过程中障碍的位置；这些条件共同决定“可行轨迹”的集合，而不是预先写死每一步动作。

UBDDL 还支持 `:noise`、`:cam_view`、`:scene_xml`、`:robot_init_state` 等扰动：图像模糊/雾化、视点变化、材质与灯光变化、初始机器人姿态变化等。训练集通过视觉随机化和物理布局随机化增加覆盖度。论文报告了 7,603 个场景、953 种物体、462 类手物组合。[论文第 3.1、3.4 节及附录 0.A](https://arxiv.org/html/2606.23686v2)

## 4. 论文里的自动轨迹生成设计

论文的关键思想是把**人指定的操作意图**和**规划器生成的连续动作**分开：[论文第 3.3 节](https://arxiv.org/html/2606.23686v2)

1. 专家按操作阶段，在目标物体的规范坐标系中提供少量末端关键位姿 $\{T_k\}_{k=1}^{K}$。预抓取、抓取、预放置、放置可作为理解示例；论文没有规定所有任务必须恰好四个关键位姿。
2. 每个关键位姿扩增为五个空间变体；每个阶段随机选一个变体，因此相同任务意图可以产生不同的路径和操作姿态。
3. 场景实例化后，把所选关键位姿及障碍物几何变换到机器人基座坐标系。
4. CuRobo 根据关键位姿、机器人运动学和障碍几何规划连续轨迹；规划时必须考虑机器人本体及**被抓住的物体**占用的体积，在整个时间区间内与障碍不相交。论文写成 $\mathcal A(x(t))\cap\mathcal O=\varnothing$，$t\in[0,T]$。
5. 生成候选经人工筛查，留下任务可行且符合安全约束的示教。这一步是**筛选**，不能理解为人工逐帧遥操作或人工逐段修补路径。

论文将四个物理 suite 的 L0/L1 共 40 个任务用于这批训练数据；每个任务先生成 500 条候选，最终筛得 19,664 条安全示教。L2 和整个 SSR suite 不纳入这批安全轨迹训练，以保留分布外物理评测与语义安全评测。论文报告人工投入约为每任务 1.8 分钟（对比遥操作 7.4 分钟）；这里的数字是论文实验结果，并不代表本机手动采集的速度。[论文第 3.3–3.4 节](https://arxiv.org/html/2606.23686v2)

**复现边界：**官方仓库 README 仍标注自动数据生成 pipeline “Coming soon”。因此无法从本仓库直接执行“标注关键位姿 → CuRobo 批量规划 → 生成论文数据集”的完整命令，也不应把下面的人工采集脚本当成它的实现。[官方仓库](https://github.com/LIBERO-SAFETY/LIBERO-Safety)

## 5. 当前仓库可运行的数据采集 pipeline

下面是**已开源代码**的实际执行顺序，输入是一项 BDDL 任务与人工动作（或本机 smoke 脚本提供的测试动作）：

1. **加载场景。**`scripts/collect_demonstration.py` 按 `--bddl-file` 创建任务环境，读取 BDDL 的物体、摆放区域、目标和约束，并初始化机器人、相机、动态实体。固定初始状态文件在 [`init_files/`](../libero/libero/init_files/)；本机部署的 SSR 任务没有对应的已发布固定初始状态，因此 smoke runner 对 SSR 会采用随机 reset。
2. **逐步操作和记录。**交互式入口使用键盘或 SpaceMouse 将输入转换成环境动作；robosuite 的 `DataCollectionWrapper` 记录仿真状态和动作。原始 episode 目录包含 `model.xml`、`state_*.npz`，其中 NPZ 保存 `states`、`action_infos` 等字段。完整状态与 XML 一起保存，是为了后续在相同场景中重放，而不只依赖视频。
3. **聚合原始 HDF5。**同脚本的 `gather_demonstrations_as_hdf5()` 读取 episode，写出 `demo.hdf5`。主要结构为 `/data/demo_n/states`、`/data/demo_n/actions`，每条 demo 保存 `model_file`；`/data` 属性还保存环境信息、任务信息和 BDDL 文件路径/文本。聚合时会丢弃末尾多出的一帧状态，以对齐状态和动作长度。
4. **重放并补充观察。**`scripts/create_dataset.py` 从 HDF5 中恢复场景 XML 和首帧 MuJoCo 状态，再按保存的动作重放。重放中提取 `agentview`、腕部相机图像和机器人本体状态，写成带 `obs`、`actions`、`states`、`robot_states`、`rewards`、`dones` 的训练用 HDF5。转换器默认跳过每条轨迹的前五步，原因是初始力传感器读数不稳定；若重放状态误差较大，会输出 divergence 警告。
5. **做质量核查。**人工示教应核对目标是否完成、是否违反安全约束，以及重放是否与原轨迹一致。尤其要注意：当前转换脚本会把保留轨迹的最后一帧 `reward` 和 `done` 置为 1，**不会据此证明任务真实成功或全程安全**。本机零动作 smoke 样本只能验证数据链路，不可当作论文中的成功示教数据。

最小可运行命令见 [`deployment/README.md`](README.md)。以下命令在本机部署根目录执行；不需要下载完整训练示教集：

```bash
cd /home/drone/Desktop/VACodebase/LIBERO-Safety
MUJOCO_GL=egl .conda-env/bin/python deployment/smoke_pipeline.py collect \
  --suite affordance --level 0 --task-id 0 --steps 8
```

它会在 `deployment/runs/<时间戳>/` 下生成原始采集文件、聚合 HDF5 和重放后的图像 HDF5。本机已做过 8 步测试：聚合文件有 8 个状态/动作记录，转换时跳过前五步后留下 3 帧 128×128 的视觉观测。`eval` 子命令还能检查模型接入形式：自定义模块需提供 `act(observation, instruction) -> array_like[7]`；默认零动作只检查环境是否能运行。

需要真人采集时，可在有图形界面的会话中执行：

```bash
cd /home/drone/Desktop/VACodebase/LIBERO-Safety
LIBERO_CONFIG_PATH="$PWD/deployment/.libero" \
  .conda-env/bin/python scripts/collect_demonstration.py \
  --bddl-file "$PWD/libero/libero/bddl_files/affordance/L0/grab_the_knife_and_cut_the_lemon_on_the_cutting_board.bddl" \
  --device keyboard --num-demonstration 1 --directory deployment/runs/human
```

本地 `demo.hdf5` 与转换后的 LIBERO 风格 HDF5 是两种处理阶段；[官方提供的训练数据](https://huggingface.co/datasets/LIBERO-Safety)另以 LeRobot 格式发布，不能默认认为字段和上述 HDF5 一一相同。当前部署未下载完整训练数据集。

## 6. 轨迹安全与测评如何对齐

论文的物理评测把**达到目标且全程无安全违规**计作成功；任一安全违规立即终止并计为失败。每个任务使用十次独立试验，结果跨三个随机种子统计；主要看成功率（SR），辅以碰撞率（CR）、执行时间和轨迹平滑度（LDLJ）。SSR 则使用静态观察与指令，统计对危险请求的拒绝率（RR）。[论文第 4.1 节](https://arxiv.org/html/2606.23686v2)

当前源码的 [`BDDLBaseDomain.step()`](../libero/libero/envs/bddl_base_domain.py)把约束检查结果放在 `info['cost']`，而其 `done` 由目标成功检查设置；本机的简易 `eval` smoke runner也没有实现论文完整的安全违规终止、CR/LDLJ 或三种子统计。因此若要复现论文指标，评测器必须逐步读取并解释安全 `cost`、实现违规终止和统计协议，不能把本机 smoke 输出的 `success` 直接当成论文 SR。

## 7. 进一步阅读与代码入口

- [论文：任务定义、轨迹生成、训练集和评测协议](https://arxiv.org/html/2606.23686v2)
- [官方仓库：发布范围与数据链接](https://github.com/LIBERO-SAFETY/LIBERO-Safety)
- [任务映射](../libero/libero/benchmark/vla_safety_task_map.py)、[BDDL 任务](../libero/libero/bddl_files/)、[环境实现](../libero/libero/envs/bddl_base_domain.py)
- [手动采集与聚合](../scripts/collect_demonstration.py)、[HDF5 重放转换](../scripts/create_dataset.py)、[本机最小测试](smoke_pipeline.py)
