# 实验 2、3：模块归因与到达反馈闭环

日期：2026-11-17。状态：**实验设计已完成，尚未实现新增开关、尚未启动训练**。

本方案针对 2026-11-16 审阅中第 2 项“隔离模块作用”和第 3 项“集中验证反馈闭环”，同时覆盖纯无线分配、双陈旧度、LPC 对照。实验以原生 FedSONATA 控制路径为对象。

## 先锁定实验底稿

统一使用 **Edge-factor + projection-LPC + LEF + CORA**，沿用 `runs/2026-11-05-fedsonata-audit/run_edge_factor_main.sh` 的训练协议。该脚本对应的归档位于 `artifacts/edge_factor_main_20261106`。不能将旧 value-based selector、train-LPC 或 wire-replay 诊断结果拼进新的 Full 行。

当前本地最接近的训练入口快照与归档主实验 `main.py` 哈希不同。因此执行前需获取准确源码、增加独立干预开关，并通过下述检查。`manifest.jsonl` 中的干预字段是**待实现的配置规格，不是已有命令行参数**。

本次论文修正以用户提供的 `(3).pdf` 对应源稿为底稿，仅补式 (13) 的加号及两处 Fixed-A 差值。代码权重函数本来就是两项相加，其 SHA-256 与 11 月 6 日主实验记录一致。另一个本地稿中的新主实验及纯无线行属于不同版本，本次不把其结果替换进用户稿。

## 共同协议与指标

| 项目 | 主协议 |
|---|---|
| 数据与模型 | CIFAR-100，ImageNet 预训练 ViT-B/16；准确记录数据、预训练权重及 tokenizer/图像预处理版本 |
| 客户端 | 20 个客户端，每轮候选 20 个；资源异构 6:3:1，allocation ratio=0.5 |
| 训练 | 固定 20 轮；data fraction=0.2；Dirichlet alpha=0.1；train/val/holdout=0.8/0.1/0.1 |
| LoRA | rank=8，alpha=16，dropout=0；batch=16；local epochs=1 |
| 学习率 | LoRA=3e-4，classifier=1e-3；其他优化器参数、client_lr=0.01 沿用归档并逐项存档 |
| 因子传输 | edge_random_rift；rho=0.35；quantum=4096 B；可靠性 floor=0.25；深度 tau=0.22、floor=0.25 |
| 校正与残差 | 主实验 projection-LPC、ridge=1e-6；b_train_every=4；LEF beta=0.5；所有臂使用相同触发实现 |
| 无线 | correlated channel，相关系数=0.9，平均增益=1e-10；总带宽=5 MHz，功率上限=23 dBm |
| 可用性与中断 | 初始坏状态概率=0.2；GE good→bad=0.1、bad→good=0.4；hazard=0.1/s；时间索引按轮固定 |
| CORA | D∈[0.5,10] s，grid=24；lambda_E=0.1、lambda_U=1；无线外层年龄系数固定为 1 |
| 状态 | reliability EWMA beta=0.8，r_init=1；effective age 的 delivery 权重=1；原始计数始终维护 |
| 随机种子 | 42、2025、2026，逐种子配对；若扩大种子数，所有主要对比同时扩至 5 个，不能按显著性选择扩跑 |
| 分类头和评估 | client_local；主指标为第 20 轮 personalized client-holdout；global accuracy 单列 |

固定 20 轮，不因某臂早停节省成本。第 20 轮是主终点，另保存按验证集选取的 checkpoint；选取结果与最终轮结果分列，holdout 不参与选择。旧 `restore_best_validation` 输出不能直接充当第 20 轮结果。

系统指标：累计 attempted UL（含元数据）、DL 每轮 server pool broadcast、received UL、compute/radio/correction modeled energy、累计 modeled MEC time、on-time、deadline/dropout/interruption 各原因、fallback 比例、predictor/corrector 实测耗时及额外状态。

主终点同时报告固定 20 轮的精度和资源成本，避免凭单一加权分数宣称全面占优。另预设个性化**验证**精度目标 85%、88%，报告首次达到目标时的通信、能耗和 modeled time；20 轮内未达到者保留为未达标，不剔除。holdout 只评估最终轮和选定 checkpoint。

## A. 纯无线分配消融

问题：相同学习、因子预算和状态反馈规则下，实际载荷的无线分配器是否改善精度—成本权衡？

| arm | 块选择 P1 | 因子选择 P2 | 实际载荷无线 corrector | 其他 |
|---|---|---|---|---|
| FULL | 原 FedSONATA | Edge-factor+原 byte cap | CORA | 主对照 |
| RADIO_EQUAL | 同 FULL | 同 FULL | Equal-Fixed | 只替换 corrector |
| RADIO_BAPFT | 同 FULL | 同 FULL | BAPFT-style radio-only | 不引入 prototype tuning |

主比较保留同一 predictor 策略，并对三臂都计入 predictor 开销；只在最终 actual-payload radio-plan dispatch 处替换 corrector。Equal-Fixed 固定带宽均分、功率=23 dBm、deadline=10 s。BAPFT 使用预先冻结的 radio-only 适配实现，并限制在同一带宽、功率及 deadline 约束内；适配后不可行则明确计为 fallback。它是 BAPFT-style 无线模块对照，不声称复现完整 BAPFT。

“相同 P1/P2”指相同规则和参数。第一轮在相同输入下，三个臂的 pre-radio 块计划、因子掩码和消息字节必须相同；不同无线计划造成的到达差异，会使之后的状态、载荷和模型自然分化。**不能要求整个训练的实际到达或后续载荷相同。**

两个层次的证据分别报告：

1. 完整训练：三臂各 3 个种子，测终点精度和资源成本。这是主因果比较。
2. 无训练条件 replay：从 FULL 的训练后快照读取相同 `(payload bytes, compute time, channel, latent outage/interruption)`，交给三个分配器；测条件能耗、时延和成功率，不宣称产生新的训练精度。若从不同 donor 采样快照，分层报告，不把它们当独立训练种子。

Edge 分支在已核对快照中不读取 predictor rates，但 predictor 还有可行性检查。因此 predictor-off 先做可行输入上的动作/载荷/corrector 计划等价性和耗时诊断；不可行情形单列。它不应与无线 corrector 消融混在一起，也不能预期会提升精度。

## B. 双陈旧度的 2×2 干预

问题：未分配历史和训练后未到达历史，各自及共同进入控制，是否改变长期结果？

始终按真实事件更新原始 `tau_alloc`、`tau_delivery`；仅改变给消费者读取的视图：

`tau_visible = m_a * tau_alloc + omega_delivery * m_n * tau_delivery`。

| arm | m_a | m_n | reliability | 说明 |
|---|---:|---:|---|---|
| AGE_OFF | 0 | 0 | observed | 关闭所提出的有效陈旧度信号 |
| ALLOC_ONLY | 1 | 0 | observed | 只保留 allocation 历史 |
| DELIVERY_ONLY | 0 | 1 | observed | 只保留 delivery 历史 |
| FULL | 1 | 1 | observed | 两种历史共同使用 |

视图必须传到所有实际读取该有效年龄的地方：P1 score/coverage、LPC 触发、predictor/corrector 的年龄项，以及当前策略确实使用的 selector 输入。无线分配外层年龄权重保持原值。不能用现有 `--mec_omega_net 0` 代替此干预，因为它同时改变多个含义不同的系数。

`AGE_OFF` 中投影模块仍存在，但年龄触发数可能为零；这是“该状态信号进入现有控制系统的总效应”。因此报告每组校正次数、计算量和触发块；若要单独归因调度器，再做只遮蔽 P1 年龄、保持其他消费者读取真实年龄的次实验。

旧 server aggregation age 与本方案的 dual counters 不同：所有组保持其原策略，并在表注中明确“off”只关闭提出的 dual signal；真实到达的改变可使其值自然变化。

主比较为 FULL 对 ALLOC_ONLY、DELIVERY_ONLY、AGE_OFF 的逐种子差值。交互量为 `I=Y_11-Y_10-Y_01+Y_00`，按独立种子估计区间；报告两种 raw counter 的非零覆盖率、分布和恢复情况。若 delivery counter 始终为零，该条件不具备检验 delivery 信息价值的能力。

**表示等价性检查。** 当前两个计数器同在 selected+arrived 时清零。在 omega_delivery=1 且消费者只读两者之和时，它与一个“未分配或已训练未到达则 +1、成功则清零”的合并计数器完全等价。更一般地，固定权重的和也能由按事件分别加 1 或加 omega 的单状态维护。先用合成事件轨迹验证此等价性；不能预设双计数器的存储形式会优于等价单计数器。四格实验检验的是事件信息是否有用，而非多存一个变量本身更优。

## C. 历史到达信息进入控制的反馈实验

问题：无线到达历史进入之后的调度/预算/资源选择，是否在不改变真实聚合与 ACK 残差语义的情况下改善结果？

用 reliability-history 开关 R 和 delivery-age 开关 N 构成第二个 2×2。allocation 年龄一直保留；和 B 组共享能复用的臂。

| arm | 控制器 reliability | delivery 年龄可见 | 复用 |
|---|---|---:|---|
| FULL | 真实 observed EWMA | 是 | A/B/C 共同 Full |
| ALLOC_ONLY | 真实 observed EWMA | 否 | B 组同名臂 |
| NO_RELIABILITY | 固定 r_init | 是 | 新臂 |
| DELIVERY_CONTROL_OPEN | 固定 r_init | 否 | 新臂 |

所有组的 raw reliability、raw counters 都继续记录真实更新；“固定”仅指消费者读取固定视图。r_visible 要贯穿 P1、byte cap 和 CORA 权重，不能只关闭 cap 缩放。叶子消费者不要再直接绕过视图读取 raw 状态。

真实 arrival 始终决定哪些更新可聚合、哪些残差可清零。LEF、availability、interruption 和 deadline 计算均正常运行。此实验精确定义为“历史到达信息进入控制的反馈”，不称关闭全部 ACK 或关闭整个学习闭环。

除主网络条件外，对 FULL 与 DELIVERY_CONTROL_OPEN 两臂增加：

- **IDEAL（校验条件）**：availability=1、hazard=0、r_init=1；使用同一 channel/资源预算，并检查全过程无 deadline miss。若检查通过，delivery counter 永远为零、r 永远为 1，两臂应逐轮等价。出现 deadline miss 时本次校验失败，不能宣称理想条件或忽略失败；先在合成可行输入上完成等价测试。
- **SHIFT（环境变化）**：预先固定轮 1-10 为主协议；轮 11-15 将 GE good→bad 从 0.1 改为 0.2、bad→good 从 0.4 改为 0.3，hazard 从 0.1 改为 0.2/s；轮 16-20 恢复主参数。阶段边界继续维护 GE 状态，不能重新抽初态。phase 内使用固定转移矩阵，时间戳为轮号；只说明它是预设网络压力条件。

SHIFT 报告各阶段 accuracy/validation 曲线、reliability 跟踪、到达率、attempted/received bytes、能耗和 fallback。恢复期除阈值恢复轮数外，报告到轮 20 的恢复缺口，未恢复者保留未恢复；不按最好结果重新选择 shift。

## D. LPC 补充实验（独立、可后续执行）

若保留“闭式 LPC 减少额外训练开销并维持精度”的主张，补 FULL projection、LPC_OFF、LPC_TRAIN 三臂。FULL 复用，新增两臂各 3 种子，共 6 次。

三臂计算相同的 trigger-eligibility 规则并记录候选事件；OFF 只跳过校正操作，其他训练相同；TRAIN 仅在相同规则触发时执行归档的 B-only 校正，固定训练步数/epoch 和训练样本，关闭 A 与任务头更新。新到达/模型状态可使后续触发自然不同，不能强迫 trace 等同后称完整控制效应。

先用相同单次 state 快照比较三种校正，记录 `||B_new A_current - B_old A_old||_F`、额外矩阵状态、校正时间及梯度步数；再做完整训练比较。TRAIN 的额外计算必须计费；projection 也计入矩阵乘法、求解及保存旧基的开销。仅“均值接近”或置信区间跨零不能宣称统计等效。

## 随机性、统计与因果边界

共享外生输入，允许控制产生的内生量不同。至少独立命名以下流：`data_split/model_init/client_sampling/minibatch/channel/GE_transition/interruption/factor_sampling`。流按 `(seed, round, client, event)` 定位；policy RNG 不能消耗 network RNG。

中断使用相同潜在 `U∈[0,1)`，按现有代码的约定计算 `T_fail=-log1p(-U)/hazard`，与事件判定 `U < 1-exp(-hazard*T_upload)` 保持一致；不能在重放时改用互补随机数。hazard=0 直接走“无中断事件”分支，日志记 `null`，不能除以零。用每臂真实上传时长判断是否中断，不重放最终 arrival 布尔值。候选客户端相同；本主协议每轮全部候选。partial selection 的扩展另立协议。

每个独立种子先形成 paired difference，再报告均值、sample SD、95% t 区间。3 种子区间自由度为 2，应完整报告且避免过强统计结论；客户端、轮次、客户端×题目不是新增独立种子。主对比预先指定，不按 p 值筛选：

1. A：FULL 对 RADIO_EQUAL；RADIO_BAPFT 为次对比。
2. B：FULL 对 ALLOC_ONLY；对 DELIVERY_ONLY/AGE_OFF 和交互量为机制分解。
3. C：SHIFT 中 FULL 对 DELIVERY_CONTROL_OPEN 的终点精度和资源差值，结合阶段轨迹；固定网络中同对比为主协议支持，IDEAL 为负对照。

## 执行规模与顺序

主网络 8 个唯一设置：FULL、RADIO_EQUAL、RADIO_BAPFT、AGE_OFF、ALLOC_ONLY、DELIVERY_ONLY、NO_RELIABILITY、DELIVERY_CONTROL_OPEN。3 种子共 **24 次**。

IDEAL/SHIFT 各只增加 FULL 与 DELIVERY_CONTROL_OPEN，两种条件共 **12 次**。核心方案合计 **36 次完整训练**。A/B/C 表格中的 FULL 按 run ID 共享，不重复训练。LPC 两个新增臂作为可选 **6 次**，不混入 36 次。

顺序：准确源码/接口检查 → 无训练合成和 replay → 2 轮小数据 smoke（只验证程序，不作精度证据）→ 主网络的 8 臂×3 种子 → IDEAL/SHIFT 的 12 次 → 可选 LPC。所有正式臂使用新输出目录。主论文 90.99 行、旧 replay 行或归档 91.11 行仅可作回归参考；不自动复用为新配对 Full。

## 完成与交付检查

- 固化 source/config/environment/model/data split/preprocessing/潜在网络 trace 哈希；新增开关实际到达消费者。
- 逐轮记录 raw/visible state、P1、factor mask、wire bytes、predictor/reference source、corrector 决策、arrival/failure、LPC 触发/工作量、LEF residual。
- 满足带宽和功率约束；不可行和 fallback 不被隐藏；未达目标不被删除。
- 保存每轮验证、final-round 与 validation-selected checkpoint/holdout，以及全部 attempted costs。
- 20 轮每轮 20 候选时，客户端决策预期 400 条（若分层记录则验证等价组合键集合），每轮评价 20 条；额外调试/评估行分类型计数，不靠总行数猜测完整性。
- 配对外生哈希一致，真实内生动作/arrival 可不同；无 NaN/Inf、Traceback/OOM、FAILED；有 DONE 和预期全部轮次后才标记完成。

实现映射与源码风险见 `implementation_audit.md`。机器可读设计清单见 `manifest.jsonl`、`manifest.csv`、`design.json`；所有行状态为 planned，清单生成不代表训练已启动。
