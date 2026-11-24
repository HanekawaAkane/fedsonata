"""
Fed-HARP 联邦学习模拟主程序。
负责初始化环境、加载数据、构建模型以及执行联邦训练循环。
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import sys
import time
import types
from typing import Dict, List, Optional, Set, Tuple

# FedHARP-MEC independent modules (compute/communication/dropout/staleness).
from mec import (
    build_default_profiles,
    TraceGenerator,
    EffectiveStaleness,
    update_effective_staleness,
    lpc_trigger_mask,
    MecMetrics,
    make_round_summary,
    model_client,
    solve_round,
    solve_convex_round,
    REFERENCE_RATE_SOURCE,
    solve_rift_cora_preallocation,
    reliability_aware_selection,
    StalenessState,
    SchedulerConfig,
    est_success_default,
    compute_trace_hash,
    set_compute_costs,
    dbm_to_watt,
    shannon_rate,
    FactorResidualState,
)
from mec.scheduler import update_success_ewma
from mec.staleness import ReliabilityView
from mec.factor_selector import RiftSelection, oldest_guard_layers, select_rift_updates
from mec.loss_aware_factor_selector import select_loss_aware_updates
from mec.symmetric_lookahead_selector import select_symmetric_lookahead_updates
from wire_codec import WireController
from mec.fedsonata_schedule import resolve_round_policy
from mec.early_stopping import ValidationEarlyStopper
from mec.fedhello import empirical_fim_scores, should_refresh_fim
from factor_projection import project_factor_dicts, project_factor_dicts_A
from mec.compression import compress_dict as compress_updates_dict


def parse_args(argv=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FedHARP: personalized federated LoRA with sensitivity-aware allocation")
    
    # 数据集参数
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="CIFAR-10 数据集目录")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "chestxray8"],
                        help="数据集名称 (cifar10/cifar100/chestxray8，默认: cifar10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Non-IID Dirichlet 分布参数 (默认: 0.5)")
    parser.add_argument("--data_partition", type=str, default="dirichlet", choices=["dirichlet", "uniform"],
                        help="客户端数据划分方式: dirichlet=Non-IID, uniform=类别均衡 IID (默认: dirichlet)")
    parser.add_argument("--client_splits", type=str, default="0.9,0.1",
                        help="每个客户端本地 train/test 划分比例，例如 '0.9,0.1'")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="每个客户端使用的训练数据比例 (0-1, 如 0.01 表示 1%% 数据；默认: 1.0)")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="ViT 模型名称 (默认: vit_base_patch16_224)")
    parser.add_argument("--pretrained", action="store_true",
                        help="是否使用预训练权重")
    parser.add_argument("--pretrained_source", type=str, default="local", choices=["modelscope", "local"],
                        help="预训练权重来源 (默认: modelscope)")
    parser.add_argument("--pretrained_path", type=str, default="vit",
                        help="本地 pretrained checkpoint 路径 (pretrained_source=local)")
    parser.add_argument("--ms_model_id", type=str, default=None,
                        help="ModelScope 模型 ID (pretrained_source=modelscope)")
    parser.add_argument("--ms_revision", type=str, default=None,
                        help="ModelScope revision/tag/commit (可选)")
    parser.add_argument("--ms_cache_dir", type=str, default=None,
                        help="ModelScope 缓存目录 (可选)")
    parser.add_argument("--ms_checkpoint_file", type=str, default=None,
                        help="ModelScope snapshot 内 checkpoint 相对路径 (可选)")
    parser.add_argument("--hf_endpoint", type=str, default=None,
                        help="HuggingFace Hub 端点/镜像 (如 https://hf-mirror.com)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA 秩 (默认: 8)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha 缩放因子 (默认: 16.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,  
                        help="LoRA dropout 率 (默认: 0.0)")
    parser.add_argument("--flora_rank_policy", type=str, default="capacity", choices=["capacity", "uniform"],
                        help="FLoRA 客户端秩策略: capacity=按资源比例映射, uniform=全部使用 lora_rank")
    parser.add_argument("--flora_min_rank", type=int, default=1,
                        help="capacity 策略下允许的最小客户端秩 (默认: 1)")
    parser.add_argument("--flora_local_ranks", type=str, default=None,
                        help="可选的 FLoRA 显式客户端秩列表，例如 '4,4,6,8'; 提供后覆盖 rank_policy")
    parser.add_argument("--hetero_rank_policy", type=str, default=None, choices=["capacity", "uniform"],
                        help="异构 rank baseline 的客户端秩策略；未设置时兼容 --flora_rank_policy")
    parser.add_argument("--hetero_min_rank", type=int, default=None,
                        help="异构 rank baseline 的最小客户端秩；未设置时兼容 --flora_min_rank")
    parser.add_argument("--hetero_local_ranks", type=str, default=None,
                        help="异构 rank baseline 的显式客户端秩列表；未设置时兼容 --flora_local_ranks")
    
    # 联邦学习参数
    parser.add_argument("--method", type=str, default="fedharp",
                        choices=[
                            "fedharp",
                            "fedharp_r",
                            "fedharp_g",
                            "fedharp_l",
                            "fedharp_a",
                            "fedanon",
                            "fedhello",
                            "fedra",
                            "flora",
                            "fedsalora",
                            "fedsa-lora",
                            "fedsa_lora",
                            "FedSA-LoRA",
                            "ffa-lora",
                            "ffalora",
                            "ffa_lora",
                            "flexlora",
                            "flex-lora",
                            "flex_lora",
                            "fedit",
                            "fed-it",
                            "fed_it",
                        ],
                        help="对比方法 (默认: fedharp)")
    parser.add_argument("--wire_codec", choices=["none", "rift", "random_rift", "edge_random_rift", "lossaware_rift", "lookahead_rift", "balanced_lookahead_rift", "counterfactual_rift", "topk", "stc", "qsgd"], default="none")
    parser.add_argument("--wire_replay", default=None)
    parser.add_argument("--num_clients", type=int, default=10,
                        help="客户端总数 (默认: 10)")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="联邦训练轮数 (默认: 1)")
    parser.add_argument("--clients_per_round", type=int, default=None,
                        help="每轮选中的客户端数量 (默认: 全部)")
    parser.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"],
                        help="客户端资源异构类型: 'uniform'(全部相同) 或 '6-3-1'(模拟高中低资源分布)")
    parser.add_argument("--allocation_ratio", type=float, default=0.5,
                        help="基础分配比例 (在 uniform 模式下使用，或作为低资源基准)")
    parser.add_argument("--aggregation_lr", type=float, default=1.0,
                        help="服务器聚合学习率 eta (默认: 1.0)")
    parser.add_argument("--struct_to_data_rounds", type=int, default=20,
                        help="全局先验到客户端敏感度反馈的过渡轮数 (默认: 20)")
    parser.add_argument("--fedhello_rgd_rounds", type=int, default=2)
    parser.add_argument("--fedhello_fim_every", type=int, default=2)
    parser.add_argument("--fedhello_fim_proxy_size", type=int, default=50)
    parser.add_argument("--fedhello_geometry", type=str, default="bottleneck",
                        choices=["triangle", "inverted_triangle", "bottleneck", "uniform"])
    parser.add_argument("--fedhello_proxy_source", type=str, default="official_cifar_test",
                        choices=["official_cifar_test", "train_slice"],
                        help="Source of the Fed-HeLLo FIM proxy samples. 'train_slice' "
                             "draws a disjoint CIFAR-train slice reserved before "
                             "partitioning (requires --reserve_proxy_slice > 0); "
                             "'official_cifar_test' is the legacy behaviour.")
    parser.add_argument("--reserve_proxy_slice", type=int, default=0,
                        help="Number of CIFAR *train* examples reserved as an "
                             "independent server-side proxy pool, excluded from the "
                             "client partition. 0 disables the reservation.")
    parser.add_argument("--reserve_proxy_slice_from", type=str, default="pre_partition",
                        choices=["pre_partition", "unused_train"],
                        help="How the reserved proxy slice is obtained. "
                             "'pre_partition' removes the indices before the "
                             "Dirichlet split (D047 behaviour; changes the client "
                             "partition). 'unused_train' keeps the partition "
                             "identical to an unreserved run and carves the slice "
                             "from client train examples that data_fraction never "
                             "consumes, so the proxy is disjoint from everything "
                             "used for training, validation and reporting.")
    parser.add_argument("--allocation_temperature", type=float, default=1.0,
                        help="DSA softmax 温度，越大越平滑 (默认: 1.0)")
    parser.add_argument("--coverage_prior_weight", type=float, default=0.15,
                        help="DSA 覆盖先验权重，用于降低过度重复选择同一层 (默认: 0.15)")
    parser.add_argument("--sample_weight_power", type=float, default=0.5,
                        help="FedHARP 聚合中的有效样本权重指数 n_k^gamma (默认: 0.5)")
    parser.add_argument("--fedharp_matrix_mode", type=str, default="a", choices=["a", "b", "ab", "adaptive"],
                        help="FedHARP 家族共享/聚合的 LoRA 矩阵模式: a/b/ab/adaptive (默认: a)")
    parser.add_argument("--adaptive_matrix_threshold", type=float, default=1.25,
                        help="adaptive 模式下 A/B delta norm 的选择阈值 (默认: 1.25)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小 (默认: 128)")
    parser.add_argument("--client_lr", type=float, default=0.01,
                        help="客户端本地学习率 (默认: 0.01)")
    parser.add_argument("--lora_lr", type=float, default=None,
                        help="LoRA factor learning rate; defaults to --client_lr")
    parser.add_argument("--classifier_lr", type=float, default=None,
                        help="classifier/head learning rate; defaults to --lora_lr")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="每轮本地训练的 Epoch 数 (默认: 1)")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="B 矩阵预热对齐的步数 (默认: 10)")
    parser.add_argument("--warmup_lr", type=float, default=0.0001,
                        help="预热阶段的学习率 (默认: 0.0001)")
    parser.add_argument("--b_train_every", type=int, default=4,
                        help="每隔多少轮通信后触发一次 B-only 训练 (默认: 5)")
    parser.add_argument("--b_num_epochs", type=int, default=1,
                        help="B-only 阶段本地训练的 Epoch 数 (默认: 1)")
    parser.add_argument("--lpc_mode", type=str, default="train",
                        choices=["train", "projection"],
                        help="LPC 机制：train=遗留 B-only/A-only 本地修正；"
                             "projection=FedHARP 闭式 Local Projection Corrector 输运 "
                             "(默认: train)")
    parser.add_argument("--projection_ridge", type=float, default=1e-6,
                        help="LPC projection 的 Tikhonov ridge (默认: 1e-6)")
    
    # 评估参数
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每多少轮评估一次 (默认: 1)")
    parser.add_argument("--test_batch_size", type=int, default=100,
                        help="测试批次大小 (默认: 100)")
    parser.add_argument("--validation_patience", type=int, default=0,
                        help="validation-only patience; 0 disables truncation")
    parser.add_argument("--validation_min_delta", type=float, default=0.0,
                        help="minimum validation accuracy improvement in percentage points")
    parser.add_argument("--min_rounds_before_stop", type=int, default=1,
                        help="minimum completed rounds before patience stopping")
    parser.add_argument("--restore_best_validation", action="store_true",
                        help="restore the best validation checkpoint before final test evaluation")
    parser.add_argument("--classifier_policy", type=str, default="client_local",
                        choices=["client_local", "server_global"],
                        help="whether classifier remains personalized or is server-aggregated")
    parser.add_argument("--evaluation_scope", type=str, default="personalized_client_holdout",
                        choices=["personalized_client_holdout", "server_global_test"],
                        help="selection/reporting scope; test is never used for stopping")
    
    # 系统参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--strict_determinism", action="store_true",
                        help="启用严格 CUDA/CPU 确定性协议；不改变信道 trace")
    parser.add_argument("--device", type=str, default="cuda",
                        help="使用的设备 (cuda/cpu, 默认: 自动检测)")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="是否保存模型检查点")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="检查点保存目录")

    # ---- FedHARP-MEC 联合优化参数 ----
    parser.add_argument("--mec", action="store_true",
                        help="启用 MEC 计算/通信/掉点/block-staleness 闭环；"
                             "不设置则走 original FedHARP 退化路径（arrival 恒真、"
                             "不调用 solver、mec_latency_s=0、energy 记为 model-based 0）")
    parser.add_argument("--mec_method", type=str, default=None,
                        choices=[
                            "fedharp_equal_fixed", "fedharp_bapft_ba",
                            "fedhello_equal_fixed",
                            "fedra_equal_fixed", "fedra_bapft_ba",
                            "fedsalora_equal_fixed", "ffa_lora_equal_fixed",
                            "flora_equal_fixed", "full_block_lora_equal_fixed",
                            "proposed_joint_cora",
                            "proposed_joint_rift_cora",
                            "proposed_joint",
                            "proposed_joint_adaptive_ab",
                            "proposed_joint_rift",
                            "proposed_joint_rift_ef",
                            "proposed_joint_rift_ef_cora",
                            "proposed_joint_rift_ef_cora_noduration",
                            "proposed_joint_rift_ef_equal_fixed",
                            "proposed_joint_rift_ef_cora_no_reliability",
                            "proposed_joint_fedsonata",
                            # ablations
                            "proposed_no_reliability", "proposed_random_blocks",
                            "proposed_equal_bandwidth", "proposed_fixed_deadline",
                            "proposed_no_network_staleness", "proposed_no_lpc",
                            "proposed_no_power_optimization",
                        ],
                        help="因子化 baseline 组合 [学习策略_网络策略]；"
                             "隐含设置 --method 与网络策略")
    parser.add_argument("--mec_channel", type=str, default="rayleigh",
                        choices=["constant", "rayleigh", "correlated"],
                        help="信道模型 (默认: rayleigh)")
    parser.add_argument("--mec_dropout_model", type=str, default="bernoulli",
                        choices=["none", "bernoulli", "gilbert_elliott"],
                        help="外生掉线模型 (默认: bernoulli)")
    parser.add_argument("--mec_dropout_rate", type=float, default=0.4,
                        help="平均外生掉线率 (默认: 0.4，与 experiment_matrix 一致)")
    parser.add_argument("--mec_channel_correlation", type=float, default=0.8,
                        help="相关移动信道的时间相关系数 (默认: 0.8)")
    parser.add_argument("--mec_channel_gain_mean", type=float, default=1.0,
                        help="Rayleigh/相关信道平均功率增益；现实链路可设 1e-10 等")
    parser.add_argument("--mec_ge_good_to_bad", type=float, default=0.1,
                        help="Gilbert-Elliott good->bad 转移概率 (默认: 0.1)")
    parser.add_argument("--mec_duration_hazard_rate", type=float, default=0.0,
                        help="上传中断 hazard (1/s); P(fail|T)=1-exp(-hazard*T)")
    parser.add_argument("--mec_total_bw_mhz", type=float, default=20.0,
                        help="总上行带宽 MHz (默认: 20)")
    parser.add_argument("--mec_power_max_dbm", type=float, default=23.0,
                        help="最大发射功率 dBm (默认: 23)")
    parser.add_argument("--mec_d_min", type=float, default=0.5,
                        help="最小 round deadline 秒 (默认: 0.5)")
    parser.add_argument("--mec_d_max", type=float, default=20.0,
                        help="最大 round deadline 秒 (默认: 20)")
    parser.add_argument("--mec_lambda_e", type=float, default=0.0,
                        help="资源求解器能量权重 lambda_E (默认: 0)")
    parser.add_argument("--mec_lambda_u", type=float, default=1.0,
                        help="资源求解器可靠学习效用权重 lambda_U (默认: 1)")
    parser.add_argument("--mec_deadline_grid_size", type=int, default=24,
                        help="CORA deadline 搜索网格点数 (默认: 24)")
    parser.add_argument("--mec_deadline_mode", type=str, default="fill",
                        choices=["fill", "natural"],
                        help="'fill' 保持旧的等窗传输语义; 'natural' 把 deadline 作为上界, "
                             "使用能量最优的真实传输时长 (T5 修复)")
    parser.add_argument("--mec_partial_arrival", action="store_true", default=False,
                        help="开启后 solver 允许部分客户端错过 deadline (不再要求全体可行), "
                             "使 deadline 选择对效用权重敏感 (T5 修复)")
    parser.add_argument("--compression", type=str, default="none",
                        choices=["none", "topk", "stc", "qsgd"],
                        help="D053 压缩基线: none/topk/stc/qsgd")
    parser.add_argument("--compression_ratio", type=float, default=0.1,
                        help="topk/stc 保留比例")
    parser.add_argument("--compression_levels", type=int, default=4,
                        help="qsgd 量化级数")
    parser.add_argument("--sched_mu_c", type=float, default=1.0,
                        help="P1 block score: compute-cost penalty mu_c (默认: 1.0)")
    parser.add_argument("--sched_mu_s", type=float, default=1.0,
                        help="P1 block score: payload-cost penalty mu_s (默认: 1.0)")
    parser.add_argument("--sched_mu_f", type=float, default=0.1,
                        help="P1 block score: coverage term mu_f (默认: 0.1)")
    parser.add_argument("--sched_gamma", type=float, default=1.0,
                        help="P1 block score: staleness exponent gamma (默认: 1.0)")
    parser.add_argument("--sched_lambda_cov", type=float, default=0.05,
                        help="P1 block score: coverage weight lambda_cov (默认: 0.05)")
    parser.add_argument("--mec_omega_net", type=float, default=1.0,
                        help="effective staleness 中网络项权重 omega_net (默认: 1)")
    parser.add_argument("--mec_reliability_beta", type=float, default=0.8,
                        help="到达可靠性 EWMA 系数 beta，越大越平滑 (默认: 0.8)")
    # ---- FS23 intervention switches (2026-11-17). These are independent
    # ---- overrides; legacy aliases are rejected by argparse choices.
    parser.add_argument("--radio_policy", type=str, default="auto",
                        choices=["auto", "cora", "equal_fixed", "bapft_style"],
                        help="最终 actual-payload 无线计划的 corrector 分派覆盖；"
                             "auto=沿用 mec_method 派生策略 (默认: auto)")
    parser.add_argument("--age_view", type=str, default="dual",
                        choices=["dual", "allocation_only", "delivery_only",
                                 "zero", "merged"],
                        help="消费者可见有效陈旧度视图；raw 计数器始终按真实事件更新 (默认: dual)")
    parser.add_argument("--reliability_view", type=str, default="observed",
                        choices=["observed", "constant_initial"],
                        help="消费者可见可靠性视图；raw EWMA 始终真实更新 (默认: observed)")
    parser.add_argument("--age_delivery_weight", type=float, default=None,
                        help="有效陈旧度合成中 delivery 项权重；未设置时沿用 --mec_omega_net")
    parser.add_argument("--radio_outer_age_weight", type=float, default=None,
                        help="无线规划外层年龄系数；未设置时沿用 --mec_omega_net")
    parser.add_argument("--network_scenario", type=str, default="base",
                        choices=["base", "ideal", "shift"],
                        help="预设网络条件：base=主协议；ideal=无中断校验；"
                             "shift=轮11-15 GE/hazard 压力段 (默认: base)")
    # ---- S3 idea switches (2026-11-18). Defaults = baseline behaviour.
    parser.add_argument("--lpc_trigger", type=str, default="age",
                        choices=["age", "predictive", "value_budget"],
                        help="LPC 触发语义：age=基线等值触发；predictive=I1 期望边际收益；"
                             "value_budget=I2 预算内按分数取前 K (默认: age)")
    parser.add_argument("--lpc_score_threshold", type=float, default=0.0,
                        help="I1 分数阈值（delta 范数 × 分配频率）(默认: 0)")
    parser.add_argument("--lpc_round_budget", type=int, default=0,
                        help="I2 每客户端每轮校正预算 K；0 表示不限 (默认: 0)")
    parser.add_argument("--cap_policy", type=str, default="linear",
                        choices=["linear", "expected_delivery"],
                        help="RIFT byte cap 策略：linear=基线；expected_delivery=I3 期望交付定价 (默认: linear)")
    parser.add_argument("--rift_budget_ratio", type=float, default=0.45,
                        help="RIFT 可靠性缩放前的 A+B 上行字节预算比例 (默认: 0.45)")
    parser.add_argument("--rift_quantum_bytes", type=int, default=4096,
                        help="RIFT 多选背包的保守字节量化粒度 (默认: 4096)")
    parser.add_argument("--rift_staleness_weight", type=float, default=0.1,
                        help="RIFT 产品空间收益的 block-staleness 权重 (默认: 0.1)")
    parser.add_argument("--rift_reliability_floor", type=float, default=0.25,
                        help="RIFT 预算可靠性缩放下界 (默认: 0.25)")
    parser.add_argument("--rift_guard_max_age", type=int, default=0,
                        help="RIFT-Guard 最大连续未发送分配机会；0 表示关闭 (默认: 0)")
    parser.add_argument("--rift_guard_reserve_ratio", type=float, default=0.5,
                        help="RIFT-Guard 为到龄 block 保留的量化预算比例 (默认: 0.5)")
    parser.add_argument("--rift_ef_decay", type=float, default=1.0,
                        help="RIFT-LEF 历史残差折扣 beta，1.0 退化为 D026 EF")
    parser.add_argument("--fedsonata_gain_target", type=float, default=0.85,
                        help="FedSONATA 保留 cap-optimal 期望产品空间收益的比例 (默认: 0.85)")
    parser.add_argument(
        "--fedsonata_schedule", type=str, default="fixed",
        choices=["fixed", "trajectory_optimal", "backload_ablation"],
        help="FedSONATA 有限时域调度；trajectory_optimal 前置高保真，backload 仅作反向消融",
    )
    parser.add_argument("--fedsonata_high_rounds", type=int, default=0,
                        help="非 fixed 调度中允许的高保真轮数")
    parser.add_argument("--fedsonata_tail_budget_ratio", type=float, default=0.35,
                        help="FedSONATA 非高保真轮的 RIFT cap 比例")
    parser.add_argument("--fedsonata_contraction_q", type=float, default=0.9,
                        help="累计轨迹上界的收缩系数 q，取值 [0,1]")
    parser.add_argument("--mec_output_dir", type=str, default=None,
                        help="MEC run 输出目录；默认 results/<run_id>")
    parser.add_argument("--mec_protocol_version", type=str, default=None,
                        help="实验协议版本标签；默认使用当前代码版本")
    parser.add_argument("--mec_code_version", type=str, default=None,
                        help="写入 config.json 的实际代码版本标签")

    return parser.parse_args(args=argv)


def _strip_double_dash(argv):
    if argv is None:
        return None
    if len(argv) > 0 and argv[0] == "--":
        return argv[1:]
    return argv


def generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:
    """
    生成每个客户端的资源能力（即可以训练的层比例）。
    
    参数:
        num_clients: 客户端总数
        heterogeneity_type: 'uniform' 或 '6-3-1'
        base_ratio: 基础比例
        
    返回:
        List[float]: 每个客户端的分配比例列表
    """
    if heterogeneity_type == "uniform":
        return [base_ratio] * num_clients
    elif heterogeneity_type == "1-1-1":
        n_low = int(0.33 * num_clients)
        n_mid = int(0.33 * num_clients)
        n_high = num_clients - n_low - n_mid
        ratios = []
        ratios.extend([base_ratio] * n_low)           # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)
        np.random.shuffle(ratios)
        return ratios
    elif heterogeneity_type == "6-3-1":
        # 模拟 Fed-HeLLo 论文中的资源分布
        # 60% 低资源 (基准比例, 如 0.25 或 0.5)
        # 30% 中资源 (基准 * 1.5 或 0.75)
        # 10% 高资源 (全量训练 1.0)
        n_low = int(0.6 * num_clients)
        n_mid = int(0.3 * num_clients)
        n_high = num_clients - n_low - n_mid
        
        # 定义不同等级的层比例
        # 注意：这里假设 base_ratio 是最低标准
        ratios = []
        ratios.extend([base_ratio] * n_low)           # Low
        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid) # Mid
        ratios.extend([1.0] * n_high)                 # High (Full Layers)
        
        # 打乱分配，使得 ID 不直接对应资源等级
        np.random.shuffle(ratios)
        return ratios
    
    return [base_ratio] * num_clients


def _build_flora_client_ranks(
    client_capacities: List[float],
    max_rank: int,
    policy: str = "capacity",
    min_rank: int = 1,
    explicit_ranks: Optional[str] = None,
) -> List[int]:
    """Map client resource capacities to heterogeneous FLoRA ranks."""
    num_clients = len(client_capacities)
    max_rank = max(1, int(max_rank))
    min_rank = max(1, min(int(min_rank), max_rank))

    if explicit_ranks is not None and str(explicit_ranks).strip() != "":
        ranks = [int(x.strip()) for x in str(explicit_ranks).split(",") if x.strip()]
        if len(ranks) == 1:
            ranks = ranks * num_clients
        if len(ranks) != num_clients:
            raise ValueError(
                f"flora_local_ranks 需要 1 个或 {num_clients} 个值，实际得到 {len(ranks)} 个"
            )
        return [max(min_rank, min(max_rank, int(rank))) for rank in ranks]

    if str(policy).lower().strip() == "uniform":
        return [max_rank] * num_clients

    return [
        max(min_rank, min(max_rank, int(round(max_rank * float(capacity)))))
        for capacity in client_capacities
    ]


_RANK_CONFIGURABLE_METHODS = {
    "flora",
    "flexlora",
    "flex-lora",
    "flex_lora",
    "fedsalora",
    "fedsa-lora",
    "fedsa_lora",
    "ffa-lora",
    "ffalora",
    "ffa_lora",
    "fedit",
    "fed-it",
    "fed_it",
}

_ZERO_PADDED_RANK_METHODS = _RANK_CONFIGURABLE_METHODS.difference(
    {"flora", "flexlora", "flex-lora", "flex_lora"}
)


def _build_method_client_ranks(
    method: str,
    client_capacities: List[float],
    max_rank: int,
    policy: str = "capacity",
    min_rank: int = 1,
    explicit_ranks: Optional[str] = None,
) -> List[int]:
    """Use cross-client rank heterogeneity only for FLoRA by default."""
    method_l = str(method).lower()
    if method_l == "flora" or (explicit_ranks is not None and str(explicit_ranks).strip()):
        return _build_flora_client_ranks(
            client_capacities=client_capacities,
            max_rank=max_rank,
            policy=policy,
            min_rank=min_rank,
            explicit_ranks=explicit_ranks,
        )
    return [max(1, int(max_rank))] * len(client_capacities)


def _resolve_heterogeneous_rank_config(args) -> Tuple[str, int, Optional[str]]:
    """Resolve neutral rank options while accepting the original FLoRA aliases."""
    policy = getattr(args, "hetero_rank_policy", None) or args.flora_rank_policy
    min_rank = getattr(args, "hetero_min_rank", None)
    if min_rank is None:
        min_rank = args.flora_min_rank
    explicit_ranks = getattr(args, "hetero_local_ranks", None)
    if explicit_ranks is None:
        explicit_ranks = args.flora_local_ranks
    return str(policy), int(min_rank), explicit_ranks
    

def evaluate_global_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """在测试集上评估全局模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:
    total = 0
    for _, v in d.items():
        total += _tensor_nbytes(v)
    return total


def _delta_A_nbytes(delta_A: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_A)


def _delta_B_nbytes(delta_B: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(delta_B)


def _resolved_subset_indices(dataset) -> Optional[List[int]]:
    """Resolve nested torch Subset indices to the original dataset index space."""
    if not hasattr(dataset, "indices") or not hasattr(dataset, "dataset"):
        return None
    local = [int(index) for index in dataset.indices]
    parent = _resolved_subset_indices(dataset.dataset)
    if parent is None:
        return local
    return [int(parent[index]) for index in local]


def _client_data_split_hash(clients) -> str:
    payload = []
    for client in clients:
        payload.append({
            "client": int(client.client_id),
            "train": _resolved_subset_indices(client.train_loader.dataset),
            "val": (
                _resolved_subset_indices(client.validation_loader.dataset)
                if getattr(client, "validation_loader", None) is not None else None
            ),
            "test": _resolved_subset_indices(client.test_loader.dataset),
        })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _classifier_module(model):
    """Locate a ViT task head across timm, torchvision and Transformers wrappers."""
    getter = getattr(model, "get_classifier", None)
    if callable(getter):
        try:
            module = getter()
            if module is not None:
                return module
        except Exception:
            pass
    for attr in ("head", "classifier", "fc"):
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "state_dict"):
            return module
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        getter = getattr(backbone, "get_classifier", None)
        if callable(getter):
            try:
                module = getter()
                if module is not None:
                    return module
            except Exception:
                pass
        for attr in ("head", "classifier", "fc"):
            module = getattr(backbone, attr, None)
            if module is not None and hasattr(module, "state_dict"):
                return module
    return None


def _get_classifier_state(model) -> Dict[str, "torch.Tensor"]:
    module = _classifier_module(model)
    if module is None:
        return {}
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _set_classifier_state(model, state: Dict[str, "torch.Tensor"]) -> None:
    if not state:
        return
    module = _classifier_module(model)
    if module is None:
        raise RuntimeError("model has no classifier/head module")
    current = module.state_dict()
    compatible = {
        k: v.to(device=current[k].device, dtype=current[k].dtype)
        for k, v in state.items()
        if k in current and tuple(current[k].shape) == tuple(v.shape)
    }
    module.load_state_dict(compatible, strict=False)


def _weighted_classifier_state(states, weights):
    """Weighted average of compatible classifier state dictionaries."""
    if not states:
        return {}
    keys = set(states[0].keys())
    if any(set(s.keys()) != keys for s in states):
        raise ValueError("classifier state dictionaries have incompatible keys")
    total = float(sum(float(w) for w in weights))
    if total <= 0.0:
        total = float(len(states))
        weights = [1.0] * len(states)
    out = {}
    for key in sorted(keys):
        ref = states[0][key]
        acc = ref.detach().clone().mul(0.0)
        for state, weight in zip(states, weights):
            acc.add_(state[key].to(device=acc.device, dtype=acc.dtype), alpha=float(weight) / total)
        out[key] = acc
    return out


def _A_matrices_nbytes(A_matrices: Dict[str, torch.Tensor]) -> int:
    return _dict_tensor_nbytes(A_matrices)


def _compress_lora_matrices_to_rank(
    A_matrices: Dict[str, torch.Tensor],
    B_matrices: Dict[str, torch.Tensor],
    target_rank: int,
    alpha: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    import torch

    compressed_A: Dict[str, torch.Tensor] = {}
    compressed_B: Dict[str, torch.Tensor] = {}
    target_rank = max(1, int(target_rank))
    for name, A in A_matrices.items():
        B = B_matrices.get(name)
        if B is None:
            continue
        source_rank = max(1, int(A.shape[0]))
        full_update = torch.matmul(B, A) * (float(alpha) / float(source_rank))
        rank = max(1, min(target_rank, int(min(full_update.shape))))
        if torch.norm(full_update).item() <= 1e-12:
            if int(A.shape[0]) >= rank:
                compressed_A[name] = A[:rank, :].clone()
            else:
                pad = torch.randn(rank - int(A.shape[0]), int(A.shape[1]), device=A.device, dtype=A.dtype) * 0.02
                compressed_A[name] = torch.cat([A.clone(), pad], dim=0)
            compressed_B[name] = torch.zeros(int(B.shape[0]), rank, device=B.device, dtype=B.dtype)
            continue
        try:
            u, s, vh = torch.linalg.svd(full_update, full_matrices=False)
        except RuntimeError:
            u, s, vh = torch.linalg.svd(full_update.float(), full_matrices=False)
            u = u.to(device=full_update.device, dtype=full_update.dtype)
            s = s.to(device=full_update.device, dtype=full_update.dtype)
            vh = vh.to(device=full_update.device, dtype=full_update.dtype)
        scale = float(alpha) / float(rank)
        if scale <= 0.0:
            scale = 1.0
        compressed_B[name] = u[:, :rank]
        compressed_A[name] = (s[:rank] / scale).view(-1, 1) * vh[:rank, :]
    return compressed_A, compressed_B


def _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:
    n = 0
    for name in allocated_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_A.numel())
        n += int(layer.lora_B.numel())
    return n


def _compute_vision_fedhello_fim_scores(
    *,
    model,
    lora_layers: Dict[str, object],
    dataloader,
    device,
    max_samples: int,
) -> tuple[Dict[str, float], int]:
    """Empirical FIM for Vision Fed-HeLLo using a disjoint proxy loader."""
    import torch
    import torch.nn as nn

    params = list(model.parameters())
    old_flags = [bool(p.requires_grad) for p in params]
    old_training = bool(model.training)
    for p in params:
        p.requires_grad = False
    for layer in lora_layers.values():
        layer.lora_A.requires_grad = True
        layer.lora_B.requires_grad = True
    scores = {str(name): 0.0 for name in lora_layers}
    seen = 0
    limit = max(1, int(max_samples))
    criterion = nn.CrossEntropyLoss()
    model.eval()
    try:
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            for idx in range(int(targets.shape[0])):
                if seen >= limit:
                    break
                model.zero_grad(set_to_none=True)
                output = model(images[idx : idx + 1])
                logits = getattr(output, "logits", output)
                criterion(logits, targets[idx : idx + 1]).backward()
                for name, layer in lora_layers.items():
                    for factor in (layer.lora_A, layer.lora_B):
                        if factor.grad is not None:
                            scores[str(name)] += float(torch.sum(factor.grad.detach().float() ** 2).item())
                seen += 1
            if seen >= limit:
                break
    finally:
        model.zero_grad(set_to_none=True)
        for p, flag in zip(params, old_flags):
            p.requires_grad = flag
        model.train(old_training)
    if seen == 0:
        raise ValueError("Fed-HeLLo vision proxy loader yielded no samples")
    return {name: value / float(seen) for name, value in scores.items()}, seen


def _b_only_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], train_B_layers: Set[str]) -> int:
    n = 0
    for name in train_B_layers:
        layer = client_lora_layers.get(name)
        if layer is None:
            continue
        n += int(layer.lora_B.numel())
    return n


def vision_main(argv=None):
    """联邦学习主循环"""
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    effective_lora_lr = float(args.lora_lr if args.lora_lr is not None else args.client_lr)
    effective_classifier_lr = float(
        args.classifier_lr if args.classifier_lr is not None else effective_lora_lr
    )
    if effective_lora_lr <= 0.0 or effective_classifier_lr <= 0.0:
        raise ValueError("lora_lr and classifier_lr must be positive")
    if int(args.validation_patience) < 0:
        raise ValueError("validation_patience must be non-negative")
    if float(args.validation_min_delta) < 0.0:
        raise ValueError("validation_min_delta must be non-negative")
    if int(args.min_rounds_before_stop) < 1:
        raise ValueError("min_rounds_before_stop must be >= 1")
    if (
        int(args.validation_patience) > 0 or bool(args.restore_best_validation)
    ) and int(args.eval_every) != 1:
        raise ValueError("validation checkpointing/stopping requires --eval_every 1")
    if (
        args.evaluation_scope == "server_global_test"
        and args.classifier_policy != "server_global"
    ):
        raise ValueError(
            "server_global_test requires --classifier_policy server_global"
        )
    if str(args.method).lower() == "fedanon":
        args.method = "fedharp"

    # ---- FedHARP-MEC: translate --mec_method into a learning policy + network
    # policy. Learning policy drives the existing `--method` code path; the
    # network policy governs bandwidth/power/deadline/block coupling.
    mec_network_policy = "equal_fixed"   # default when --mec off
    if args.mec_method is not None:
        prefix = args.mec_method.split("_")[0]
        learning_map = {
            "fedharp": "fedharp",
            "fedhello": "fedhello",
            "fedra": "fedra",
            "fedsalora": "fedsalora",
            "ffa": "ffa-lora",
            "flora": "flora",
            "full": "fedit",
            "proposed": "fedharp",   # Proposed joint optimizer uses FedHARP learning
        }
        if prefix not in learning_map:
            raise ValueError(f"unknown mec_method prefix: {args.mec_method}")
        args.method = learning_map[prefix]
        if args.mec_method in {
            "proposed_joint_cora", "proposed_joint_rift_cora",
            "proposed_joint_rift_ef_cora",
            "proposed_joint_rift_ef_cora_noduration",
            "proposed_joint_rift_ef_cora_no_reliability",
        }:
            mec_network_policy = "convex_joint"
        elif args.mec_method in {
            "proposed_joint", "proposed_joint_adaptive_ab", "proposed_joint_rift",
            "proposed_joint_rift_ef", "proposed_joint_fedsonata",
        }:
            mec_network_policy = "proposed_joint"
        elif args.mec_method.endswith("bapft_ba"):
            mec_network_policy = "bapft_ba"
        else:
            mec_network_policy = "equal_fixed"
        args.mec = True
        if args.mec_method == "proposed_joint_adaptive_ab":
            args.fedharp_matrix_mode = "adaptive"
        elif args.mec_method == "fedhello_equal_fixed":
            args.fedharp_matrix_mode = "ab"
        elif args.mec_method in {
            "proposed_joint_rift", "proposed_joint_rift_ef",
            "proposed_joint_rift_cora", "proposed_joint_rift_ef_cora",
            "proposed_joint_rift_ef_cora_noduration",
            "proposed_joint_rift_ef_equal_fixed",
            "proposed_joint_rift_ef_cora_no_reliability", "proposed_joint_fedsonata",
        }:
            # Train both factors locally; RIFT filters the transmitted deltas
            # in effective BA product space after local training.
            args.fedharp_matrix_mode = "ab"
    else:
        mec_network_policy = "equal_fixed"

    # FS23: independent radio corrector dispatch (final payload plan only).
    _radio_policy_map = {
        "cora": "convex_joint",
        "equal_fixed": "equal_fixed",
        "bapft_style": "bapft_ba",
    }
    if str(args.radio_policy) == "auto":
        radio_policy = mec_network_policy
    else:
        radio_policy = _radio_policy_map[str(args.radio_policy)]
    # FS23: decouple the effective-age composition weight from the radio
    # outer age coefficient (PROTOCOL_CN: both fixed at 1 in the protocol).
    age_delivery_weight = (
        float(args.mec_omega_net) if args.age_delivery_weight is None
        else float(args.age_delivery_weight)
    )
    radio_outer_age_weight = (
        float(args.mec_omega_net) if args.radio_outer_age_weight is None
        else float(args.radio_outer_age_weight)
    )

    method_l = str(args.method).lower()
    if method_l in {"fedsalora", "fedsa-lora", "fedsa_lora", "ffa-lora", "ffalora", "ffa_lora"}:
        if "--heterogeneity_type" not in raw_argv:
            args.heterogeneity_type = "uniform"
        if "--allocation_ratio" not in raw_argv:
            args.allocation_ratio = 1.0

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    
    # 设置随机种子
    global np
    global torch
    global nn
    global DataLoader
    global CIFAR10Dataset
    global CIFAR100Dataset
    global ChestXray8Dataset
    global FedHarpClient
    global FedHarpServer
    global create_vit_model
    global create_client_model
    global set_seed
    global get_device
    global save_checkpoint
    global create_directories
    global print_model_info

    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency 'numpy'. Install it before running vision experiments.") from e

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependency 'torch'. Install it before running vision experiments.") from e

    from client import FedHarpClient
    from dataset import CIFAR10Dataset, CIFAR100Dataset, ChestXray8Dataset
    from models import create_client_model, create_vit_model
    from server import FedHarpServer
    from utils import (
        create_directories,
        get_device,
        print_model_info,
        save_checkpoint,
        set_seed,
    )

    if bool(args.strict_determinism):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(args.seed)
    if bool(args.strict_determinism):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
    
    # 获取计算设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    # 创建目录
    if args.save_checkpoints:
        create_directories(args.checkpoint_dir)

    safe_method = str(args.method).replace("/", "_").replace(" ", "_")
    safe_dataset = str(args.dataset).replace("/", "_").replace(" ", "_")
    safe_model = str(args.model_name).replace("/", "_").replace(" ", "_")
    safe_alpha = str(args.alpha).replace("/", "_").replace(" ", "_")
    safe_partition = str(args.data_partition).replace("/", "_").replace(" ", "_")
    safe_heterogeneity_type = str(args.heterogeneity_type).replace("/", "_").replace(" ", "_")
    safe_b_train_every = str(args.b_train_every).replace("/", "_").replace(" ", "_")
    log_filename = f"Traininglog-{safe_method}-{safe_partition}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_b_train_every}--{safe_heterogeneity_type}.txt"
    log_path = os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)

    # 将本次训练的配置写入日志文件（覆盖旧内容）
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Training configuration (arguments):\n")
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")
            f.write("\nRound logs:\n")
    except Exception as e:
        print(f"[警告] 无法写入日志文件 {log_path}: {e}")
    
    print("\n" + "="*80)
    print("FedHARP 实验流程")
    print("="*80)
    print("[配置] 本次实验参数")
    print(f"  方法: {args.method}")
    print(f"  数据集: {args.dataset.upper()} | 划分: {args.data_partition} | alpha={args.alpha}")
    print(f"  客户端: {args.num_clients} 个 | 通信轮数: {args.num_rounds}")
    print(f"  资源异构: {args.heterogeneity_type} | 基础层预算比例: {args.allocation_ratio}")
    print(f"  LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  设备: {device}")
    print("="*80 + "\n")
    
    # 加载数据集
    print(f"[数据] 正在加载 {args.dataset.upper()} 数据集...")
    try:
        splits = [float(x) for x in args.client_splits.split(",")]
    except:
        splits = [0.9, 0.1]
    
    if args.dataset == "cifar100":
        dataset = CIFAR100Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits,
            data_partition=args.data_partition,
            proxy_reserve_size=int(getattr(args, "reserve_proxy_slice", 0)),
            proxy_reserve_mode=str(getattr(args, "reserve_proxy_slice_from", "pre_partition")),
        )
        num_classes = 100
    elif args.dataset == "chestxray8":
        dataset = ChestXray8Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits,
            data_partition=args.data_partition
        )
        num_classes = int(getattr(dataset, "num_classes", 15))
    else:
        dataset = CIFAR10Dataset(
            data_dir=args.data_dir,
            alpha=args.alpha,
            num_clients=args.num_clients,
            seed=args.seed,
            splits=splits,
            data_partition=args.data_partition,
            proxy_reserve_size=int(getattr(args, "reserve_proxy_slice", 0)),
            proxy_reserve_mode=str(getattr(args, "reserve_proxy_slice_from", "pre_partition")),
        )
        num_classes = 10
    
    # 获取全局测试集
    global_test_loader = dataset.get_test_dataloader(batch_size=args.test_batch_size)

    # Fed-HeLLo FIM proxy loader. Construction is deferred until after the
    # client train dataloaders exist: the 'unused_train' reserve mode carves the
    # slice from client train examples that data_fraction never consumes, which
    # is only known once those subsets have been drawn (D047 §3.4 v2).
    fedhello_proxy_loader = None
    fedhello_proxy_source_label = None
    
    # 创建全局模型
    print("\n[模型] 正在创建带有 LoRA 的 Vision Transformer...")
    global_model, global_lora_layers = create_vit_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        pretrained_source=args.pretrained_source,
        pretrained_checkpoint_path=args.pretrained_path,
        modelscope_model_id=args.ms_model_id,
        modelscope_revision=args.ms_revision,
        modelscope_cache_dir=args.ms_cache_dir,
        modelscope_checkpoint_file=args.ms_checkpoint_file,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    global_model.to(device)
    print_model_info(global_model, global_lora_layers)
    
    # --- 关键修改：生成异构的客户端资源能力 ---
    print(f"\n[资源] 正在生成客户端资源分布 ({args.heterogeneity_type})...")
    client_capacities = generate_client_resources(
        args.num_clients, 
        args.heterogeneity_type, 
        args.allocation_ratio
    )
    # 打印资源分布概览
    avg_cap = np.mean(client_capacities)
    print(f"  层训练比例: 平均={avg_cap:.2f}, 最小={min(client_capacities):.2f}, 最大={max(client_capacities):.2f}")
    if args.num_clients <= 20:
        print(f"  详细分布: {['{:.2f}'.format(c) for c in client_capacities]}")

    hetero_client_ranks: Optional[List[int]] = None
    rank_policy, rank_min_rank, rank_local_ranks = _resolve_heterogeneous_rank_config(args)
    if method_l in _RANK_CONFIGURABLE_METHODS:
        hetero_client_ranks = _build_method_client_ranks(
            method=method_l,
            client_capacities=client_capacities,
            max_rank=args.lora_rank,
            policy=rank_policy,
            min_rank=rank_min_rank,
            explicit_ranks=rank_local_ranks,
        )
        rank_mode = "heterogeneous_rank" if len(set(hetero_client_ranks)) > 1 else "fixed_rank"
        print(
            f"  {args.method} 秩口径: mode={rank_mode}, policy={rank_policy}, "
            f"平均={np.mean(hetero_client_ranks):.2f}, "
            f"最小={min(hetero_client_ranks)}, 最大={max(hetero_client_ranks)}"
        )
        if args.num_clients <= 20:
            print(f"  客户端秩: {hetero_client_ranks}")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                if method_l == "flora":
                    aggregation = "stacking"
                elif method_l in {"flexlora", "flex-lora", "flex_lora"}:
                    aggregation = "svd_full_update_average"
                elif rank_mode == "heterogeneous_rank":
                    aggregation = "zero_padded_weighted_average"
                else:
                    aggregation = "weighted_average"
                for cid, (capacity, rank) in enumerate(zip(client_capacities, hetero_client_ranks)):
                    f.write(
                        f"type=rank_assignment, mode={rank_mode}, method={method_l}, "
                        f"aggregation={aggregation}, client={cid}, "
                        f"capacity={float(capacity):.4f}, rank={int(rank)}\n"
                    )
        except Exception as e:
            print(f"[警告] 无法写入异构秩分配日志: {e}")

    # 初始化服务器
    print("\n[服务器] 正在初始化 Fed-HARP 服务器...")
    # 注意：这里我们假设 server.py 已经更新了 __init__ 来接收 client_capacities
    # 如果 server.py 尚未更新，它可能会忽略这个参数，但我们在 main 中显式生成它以体现逻辑。
    # 为了兼容旧版 server 代码，我们保留 allocation_ratio 参数，但期望 server 使用 client_capacities
    server = FedHarpServer(
        model=global_model,
        lora_layers=global_lora_layers,
        num_clients=args.num_clients,
        method=args.method,
        allocation_ratio=args.allocation_ratio, # 作为默认值
        aggregation_lr=args.aggregation_lr,
        struct_to_data_rounds=args.struct_to_data_rounds,
        seed=args.seed,
        allocation_temperature=args.allocation_temperature,
        coverage_prior_weight=args.coverage_prior_weight,
        sample_weight_power=args.sample_weight_power,
        fedharp_matrix_mode=args.fedharp_matrix_mode,
    )
    # 手动将生成的异构能力注入到 server (模拟 Server 知晓客户端能力)
    if hasattr(server, 'client_layer_counts'):
        # 将比例转换为具体的层数
        num_total_layers = len(global_lora_layers)
        for cid, cap in enumerate(client_capacities):
            server.client_layer_counts[cid] = max(1, int(num_total_layers * cap))
        print("  已注入客户端层预算。")
    if method_l == "fedhello":
        server.fedhello_rgd_rounds = max(0, int(args.fedhello_rgd_rounds))
        server.fedhello_fim_every = max(1, int(args.fedhello_fim_every))
        server.fedhello_geometry = str(args.fedhello_geometry)
        print(
            "  Fed-HeLLo-style: "
            f"geometry={server.fedhello_geometry}, "
            f"rgd_rounds={server.fedhello_rgd_rounds}, "
            f"fim_every={server.fedhello_fim_every}, "
            f"proxy_size={int(args.fedhello_fim_proxy_size)}"
        )
    
    # 初始化客户端
    clients = []
    client_sample_counts = {}
    
    print(f"\n[客户端] 正在初始化 {args.num_clients} 个客户端...")
    for client_id in range(args.num_clients):
        # 获取本地训练集
        train_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.batch_size,
            shuffle=True,
            split="train",
            fraction=args.data_fraction
        )
        client_val_loader = None
        if "val" in getattr(dataset, "client_datasets", [{}])[client_id]:
            client_val_loader = dataset.get_client_dataloader(
                client_id,
                batch_size=args.test_batch_size,
                shuffle=False,
                split="val",
                fraction=1.0,
            )
        # 获取本地测试集
        client_test_loader = dataset.get_client_dataloader(
            client_id,
            batch_size=args.test_batch_size,
            shuffle=False,
            split="test",
            fraction=1.0
        )
        
        client_sample_counts[client_id] = len(train_loader.dataset)
        
        # 创建客户端模型副本
        client_model, client_lora_layers = create_client_model(
            global_model,
            global_lora_layers
        )
        client_model.to(device)
        if method_l in _RANK_CONFIGURABLE_METHODS:
            if hetero_client_ranks is None:
                raise RuntimeError(f"{args.method} 客户端秩尚未初始化")
            client_rank = int(hetero_client_ranks[client_id])
            for layer in client_lora_layers.values():
                layer.reset_lora(rank=client_rank)
        
        client = FedHarpClient(
            client_id=client_id,
            model=client_model,
            lora_layers=client_lora_layers,
            train_loader=train_loader,
            validation_loader=client_val_loader,
            test_loader=client_test_loader, 
            device=device,
            lr=args.client_lr,
            lora_lr=(args.lora_lr if args.lora_lr is not None else args.client_lr),
            classifier_lr=(
                args.classifier_lr
                if args.classifier_lr is not None
                else (args.lora_lr if args.lora_lr is not None else args.client_lr)
            ),
            method=args.method,
            warmup_steps=args.warmup_steps,
            warmup_lr=args.warmup_lr
        )
        clients.append(client)
        if client_id < 5:
            print(f"  客户端 {client_id}: train={len(train_loader.dataset)}")
    
    if args.num_clients > 5:
        print(f"  其余 {args.num_clients - 5} 个客户端已初始化。")

    client_data_split_hash = _client_data_split_hash(clients)
    print(f"  客户端实际 train/val/test 子集哈希: {client_data_split_hash}")

    # Fed-HeLLo FIM proxy: now that the client train subsets are drawn, the
    # 'unused_train' reserve can carve a slice disjoint from every example used
    # for training, validation or reporting, without touching the partition.
    if method_l == "fedhello":
        if str(args.fedhello_proxy_source) == "train_slice":
            if str(getattr(args, "reserve_proxy_slice_from", "pre_partition")) == "unused_train":
                dataset.reserve_unused_train_proxy()
                fedhello_proxy_source_label = getattr(
                    dataset, "proxy_source_label", None
                ) or "cifar100_client_train_unused_by_data_fraction"
            else:
                fedhello_proxy_source_label = "cifar100_train_reserved_slice_disjoint_from_clients"
            fedhello_proxy_loader = dataset.get_proxy_dataloader(batch_size=args.test_batch_size)
        else:
            fedhello_proxy_loader = global_test_loader
            fedhello_proxy_source_label = "official_cifar_test_unreported_in_personalized_protocol"
        print(f"  Fed-HeLLo FIM proxy source: {fedhello_proxy_source_label}")
    
    # 确定每轮参与的客户端
    clients_per_round = args.clients_per_round or args.num_clients
    # Dedicated RNG prevents method-specific numpy calls from changing the
    # sampled-client trajectory.  Every matched policy/seed pair therefore
    # receives the same ordered client list in each common round.
    client_selection_seed = int(args.seed) + 761
    client_selection_rng = np.random.RandomState(client_selection_seed)
    
    # 开始联邦学习主循环
    print("\n" + "="*80)
    print("[训练] 开始联邦训练")
    print("="*80)
    
    best_accuracy = 0.0
    training_history = []

    method_l = str(args.method).lower()
    is_fedharp_family = method_l in {"fedharp", "fedharp_r", "fedharp_g", "fedharp_l", "fedharp_a"}
    fedharp_matrix_mode = str(getattr(args, "fedharp_matrix_mode", "a")).lower().replace("-", "_")
    fedharp_upload_B = is_fedharp_family and fedharp_matrix_mode in {"b", "ab", "adaptive"}
    fedsonata_enabled = bool(args.mec_method == "proposed_joint_fedsonata")
    rift_enabled = bool(args.mec_method in {
        "proposed_joint_rift", "proposed_joint_rift_ef",
        "proposed_joint_rift_cora", "proposed_joint_rift_ef_cora",
        "proposed_joint_rift_ef_cora_noduration",
        "proposed_joint_rift_ef_equal_fixed",
        "proposed_joint_rift_ef_cora_no_reliability", "proposed_joint_fedsonata",
    })
    rift_ef_enabled = bool(args.mec_method in {
        "proposed_joint_rift_ef", "proposed_joint_rift_ef_cora",
        "proposed_joint_rift_ef_cora_noduration",
        "proposed_joint_rift_ef_equal_fixed",
        "proposed_joint_rift_ef_cora_no_reliability",
        "proposed_joint_fedsonata",
    })
    rift_duration_value_enabled = bool(
        rift_enabled
        and args.mec_method != "proposed_joint_rift_ef_cora_noduration"
    )
    rift_guard_max_age = int(args.rift_guard_max_age)
    rift_ef_decay = float(args.rift_ef_decay)
    if not 0.0 < rift_ef_decay <= 1.0:
        raise ValueError("rift_ef_decay must be in (0, 1]")
    fedsonata_gain_target = float(args.fedsonata_gain_target)
    if fedsonata_enabled and not 0.0 < fedsonata_gain_target <= 1.0:
        raise ValueError("fedsonata_gain_target must be in (0, 1]")
    fedsonata_schedule = str(args.fedsonata_schedule).strip().lower()
    fedsonata_high_rounds = int(args.fedsonata_high_rounds)
    fedsonata_tail_budget_ratio = float(args.fedsonata_tail_budget_ratio)
    fedsonata_contraction_q = float(args.fedsonata_contraction_q)
    if fedsonata_enabled:
        if not 0.0 < fedsonata_tail_budget_ratio <= float(args.rift_budget_ratio):
            raise ValueError(
                "fedsonata_tail_budget_ratio must be in (0, rift_budget_ratio]"
            )
        if not 0.0 <= fedsonata_contraction_q <= 1.0:
            raise ValueError("fedsonata_contraction_q must be in [0, 1]")
        if fedsonata_schedule != "fixed" and not (
            1 <= fedsonata_high_rounds < int(args.num_rounds)
        ):
            raise ValueError(
                "scheduled FedSONATA requires high_rounds in [1, num_rounds-1]"
            )
    rift_guard_enabled = bool(rift_enabled and not rift_ef_enabled and rift_guard_max_age > 0)
    if rift_guard_max_age < 0:
        raise ValueError("rift_guard_max_age must be nonnegative")
    if rift_guard_enabled and not 0.0 < float(args.rift_guard_reserve_ratio) <= 1.0:
        raise ValueError("RIFT-Guard requires reserve ratio in (0, 1]")
    # Consecutive allocated opportunities where a block received neither factor.
    # This is distinct from scheduler/network staleness and is updated after each
    # RIFT choice.  Guard bounds this conditional omission age under reserve feasibility.
    rift_guard_ages = {cid: {} for cid in range(args.num_clients)}
    rift_ef_state = FactorResidualState(decay=rift_ef_decay) if rift_ef_enabled else None
    compression_method = str(getattr(args, "compression", "none"))
    compression_ratio = float(getattr(args, "compression_ratio", 0.1))
    compression_levels = int(getattr(args, "compression_levels", 4))
    compression_residual: dict = {}
    b_train_every = int(args.b_train_every) if (is_fedharp_family and method_l != "fedharp_l") else 0
    lpc_projection_enabled = (
        is_fedharp_family
        and b_train_every > 0
        and str(getattr(args, "lpc_mode", "train")).lower() == "projection"
    )
    lpc_projection_events = 0
    full_b_methods = {
        "fedhello",
        "flora",
        "fedra",
        "ffa-lora",
        "ffalora",
        "ffa_lora",
        "flexlora",
        "flex-lora",
        "flex_lora",
        "fedit",
        "fed-it",
        "fed_it",
    }
    client_personal_A_states = {cid: clients[cid].get_all_A_matrices() for cid in range(args.num_clients)}
    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}
    global_classifier_state = _get_classifier_state(global_model)
    client_personal_classifier_states = {
        cid: clients[cid].get_classifier_state() for cid in range(args.num_clients)
    }
    validation_enabled = bool(
        int(args.validation_patience) > 0
        or bool(args.restore_best_validation)
        or any(getattr(c, "validation_loader", None) is not None for c in clients)
    )
    if validation_enabled and not all(
        getattr(c, "validation_loader", None) is not None for c in clients
    ):
        raise ValueError(
            "validation protocol requires a three-way client split, e.g. --client_splits 0.8,0.1,0.1"
        )
    best_validation_accuracy = float("-inf")
    best_validation_round = 0
    validation_bad_rounds = 0
    validation_snapshot = None
    stopped_early = False
    validation_stopper = (
        ValidationEarlyStopper(
            min_rounds=int(args.min_rounds_before_stop),
            max_rounds=int(args.num_rounds),
            patience=int(args.validation_patience),
            min_delta=float(args.validation_min_delta) / 100.0,
        )
        if int(args.validation_patience) > 0 else None
    )

    # ------------------------------------------------------------------
    # FedHARP-MEC context setup (bypass extension; no change to original
    # FedHARP path when args.mec is False).
    # ------------------------------------------------------------------
    mec_enabled = bool(args.mec)
    # MEC comparisons use FedRA with the paper-matched Share-A/Keep-B scope.
    # Native FedRA A+B results remain legacy evidence, not a fair comparator.
    fedra_a_only = bool(mec_enabled and method_l == "fedra")
    server.fedra_a_only = fedra_a_only
    server_global_adapter_available = bool(
        fedharp_upload_B or (method_l in full_b_methods and not fedra_a_only)
    )
    if args.evaluation_scope == "server_global_test":
        if not server_global_adapter_available:
            raise ValueError(
                "server_global_test is incompatible with a private/local LoRA factor; "
                "use personalized_client_holdout for this policy"
            )
        if hetero_client_ranks is not None and len(set(hetero_client_ranks)) > 1:
            raise ValueError(
                "server_global_test requires one server adapter rank, not heterogeneous client ranks"
            )
    mec_profiles = None
    mec_trace = None
    mec_staleness = None
    mec_metrics = None
    mec_layer_names = list(global_lora_layers.keys())
    mec_scheduler_cfg = SchedulerConfig(
        mu_c=args.sched_mu_c,
        mu_s=args.sched_mu_s,
        mu_f=args.sched_mu_f,
        gamma=args.sched_gamma,
        lambda_cov=args.sched_lambda_cov,
    )
    mec_est_success = est_success_default(list(range(args.num_clients)), 1.0)
    mec_trace_hash = "n/a"
    mec_run_output_dir = None
    mec_config_path = None
    mec_metrics_path = None
    if mec_enabled:
        mec_profiles = build_default_profiles(
            num_clients=args.num_clients,
            heterogeneity_type=args.heterogeneity_type,
            base_ratio=args.allocation_ratio,
            total_bandwidth_mhz=args.mec_total_bw_mhz,
            power_max_dbm=args.mec_power_max_dbm,
            channel_model=args.mec_channel,
            dropout_model=args.mec_dropout_model,
            dropout_rate=args.mec_dropout_rate,
        )
        if not 0.0 <= float(args.mec_channel_correlation) < 1.0:
            raise ValueError("mec_channel_correlation must be in [0, 1)")
        if float(args.mec_channel_gain_mean) <= 0.0:
            raise ValueError("mec_channel_gain_mean must be positive")
        if float(args.mec_duration_hazard_rate) < 0.0:
            raise ValueError("mec_duration_hazard_rate must be non-negative")
        mec_profiles.channel.correlation = float(args.mec_channel_correlation)
        mec_profiles.channel.rayleigh_mean = float(args.mec_channel_gain_mean)
        if args.mec_dropout_model == "gilbert_elliott":
            p_ge = float(args.mec_ge_good_to_bad)
            if not 0.0 <= p_ge <= 1.0:
                raise ValueError("mec_ge_good_to_bad must be in [0, 1]")
            if float(args.mec_dropout_rate) <= 0.0:
                q_ge = 1.0
                p_ge = 0.0
            else:
                q_ge = p_ge * (1.0 - float(args.mec_dropout_rate)) / float(args.mec_dropout_rate)
            if not 0.0 < q_ge <= 1.0:
                raise ValueError(
                    "Gilbert-Elliott parameters infeasible: choose a smaller good->bad probability"
                )
            mec_profiles.dropout.ge_good_to_bad = p_ge
            mec_profiles.dropout.ge_bad_to_good = q_ge
        # FS23 network scenarios: pre-declared GE / hazard schedules.
        # BASE keeps the exact archived construction (byte-identical).
        fs23_ge_schedule = None
        fs23_hazard_schedule = None
        fs23_scenario = str(getattr(args, "network_scenario", "base"))
        fs23_base_hazard = float(args.mec_duration_hazard_rate)
        if fs23_scenario == "ideal":
            # PROTOCOL_CN IDEAL: availability=1, hazard=0, same channel.
            from mec.profiles import DropoutProfile as _DP
            mec_profiles.dropout = _DP(model="none", dropout_rate=0.0)
            fs23_base_hazard = 0.0
        elif fs23_scenario == "shift":
            # PROTOCOL_CN SHIFT: rounds 1-10 main; 11-15 stress; 16-20 back.
            # Explicit (p, q) per phase; GE state continuous (no re-draw).
            _p1, _q1 = float(args.mec_ge_good_to_bad), (
                float(args.mec_ge_good_to_bad) * (1.0 - float(args.mec_dropout_rate))
                / float(args.mec_dropout_rate)
                if float(args.mec_dropout_rate) > 0 else 1.0
            )
            fs23_ge_schedule = (
                (1, 10, _p1, _q1),
                (11, 15, 0.2, 0.3),
                (16, int(args.num_rounds), _p1, _q1),
            )
            fs23_hazard_schedule = (
                (1, 10, fs23_base_hazard),
                (11, 15, 0.2),
                (16, int(args.num_rounds), fs23_base_hazard),
            )
        mec_trace = TraceGenerator(
            args.seed, args.num_clients, mec_profiles.channel, mec_profiles.dropout,
            ge_schedule=fs23_ge_schedule,
            hazard_schedule=fs23_hazard_schedule,
            default_hazard=fs23_base_hazard,
        )
        mec_staleness = EffectiveStaleness(
            omega_net=age_delivery_weight,
            age_view=str(getattr(args, "age_view", "dual")),
        )
        fs23_last_delta_norm = {}
        fs23_alloc_freq = {}
        mec_rel_view = ReliabilityView(
            raw=mec_est_success,
            view=str(getattr(args, "reliability_view", "observed")),
            r_init=1.0,
        )

        def mec_hazard_for_round(round_num: int) -> float:
            """Duration hazard for this round under the active scenario."""
            if fs23_hazard_schedule is not None:
                return float(mec_trace.hazard_rate(int(round_num)))
            return fs23_base_hazard
        mec_metrics = MecMetrics()
        # model-based compute cost per sample (see SYSTEM_MODEL sec.2 / D006).
        # Order-of-magnitude estimates for a ViT-B LoRA backward; these are
        # NOT device measurements.
        set_compute_costs(c0=2.0e6, c_lora=8.0e4)
        mec_trace_hash = compute_trace_hash(
            args.seed, args.num_clients, mec_profiles.channel,
            mec_profiles.dropout, args.num_rounds,
            duration_hazard_rate=fs23_base_hazard,
            ge_schedule=fs23_ge_schedule,
            hazard_schedule=fs23_hazard_schedule,
        )
        # Output directory + run manifest files (DONE/FAILED/config/metrics).
        lpc_mode_tag = (
            "_lpcproj"
            if str(getattr(args, "lpc_mode", "train")).lower() == "projection"
            else ""
        )
        run_id = (
            f"mec_{args.mec_method or 'custom'}_"
            f"{args.dataset}_{args.num_clients}cl_{args.mec_channel}_"
            f"dr{args.mec_dropout_rate}_bw{args.mec_total_bw_mhz}_s{args.seed}"
            f"{lpc_mode_tag}"
        )
        mec_run_output_dir = args.mec_output_dir or os.path.join("results", run_id)
        os.makedirs(mec_run_output_dir, exist_ok=True)
        mec_config_path = os.path.join(mec_run_output_dir, "config.json")
        mec_metrics_path = os.path.join(mec_run_output_dir, "metrics.jsonl")
        mec_run_log_path = os.path.join(mec_run_output_dir, "run.log")

        def mec_log(msg: str) -> None:
            """Append a lifecycle/diagnostic line to the per-run run.log."""
            try:
                ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                with open(mec_run_log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{ts} {msg}\n")
            except Exception:
                pass

        # Register a shutdown guard: if the process exits without writing DONE
        # (crash / traceback / OOM), stamp FAILED so the auditor marks it.
        _mec_out_dir = mec_run_output_dir

        def _mec_atexit() -> None:
            d = _mec_out_dir
            if d and not os.path.exists(os.path.join(d, "DONE")):
                try:
                    mec_log("process exited before DONE (no traceback captured here)")
                except Exception:
                    pass
                try:
                    with open(os.path.join(d, "FAILED"), "w", encoding="utf-8") as fh:
                        fh.write("process exited before DONE (no traceback captured here)\n")
                except Exception:
                    pass

        atexit.register(_mec_atexit)
        # Skip if a valid DONE already exists (RESULT_AUDIT sec.1).
        if os.path.exists(os.path.join(mec_run_output_dir, "DONE")):
            print(f"[MEC] 已完成 run 跳过: {mec_run_output_dir}")
            mec_log("skipped: DONE already present")
            return
        else:
            try:
                config = {
                    "run_id": run_id,
                    "code_version": args.mec_code_version or (
                        "mec-v1.6.0-d034-fedsonata-cora-pc"
                        if str(args.mec_protocol_version or "").startswith("mec-v1.6-d034")
                        else "mec-v1.5.1-d033-fedsonata-cora-pc"
                        if str(args.mec_protocol_version or "").startswith("mec-v1.5.1-d033")
                        else "mec-v1.4.0-d032-cifar100-stabilized"
                    ),
                    "protocol_version": (
                        args.mec_protocol_version or "mec-v1.4.0-d032-cifar100-stabilized"
                    ),
                    "seed": args.seed,
                    "trace_hash": mec_trace_hash,
                    "fs23_intervention": {
                        "radio_policy_cli": str(getattr(args, "radio_policy", "auto")),
                        "radio_policy_dispatch": radio_policy,
                        "age_view": str(getattr(args, "age_view", "dual")),
                        "reliability_view": str(getattr(args, "reliability_view", "observed")),
                        "age_delivery_weight": age_delivery_weight,
                        "radio_outer_age_weight": radio_outer_age_weight,
                        "network_scenario": fs23_scenario,
                        "ge_schedule": (list(fs23_ge_schedule) if fs23_ge_schedule else None),
                        "hazard_schedule": (list(fs23_hazard_schedule) if fs23_hazard_schedule else None),
                    },
                    "method": args.method,
                    "mec_method": args.mec_method,
                    "network_policy": mec_network_policy,
                    "matrix_mode": fedharp_matrix_mode,
                    "adaptive_matrix_threshold": (
                        float(args.adaptive_matrix_threshold)
                        if fedharp_matrix_mode == "adaptive"
                        else None
                    ),
                    "factor_policy": (
                        "fedsonata" if fedsonata_enabled
                        else ("rift_lef" if rift_ef_decay < 1.0 else "rift_ef") if rift_ef_enabled
                        else "rift_guard" if rift_guard_enabled
                        else "rift" if rift_enabled
                        else fedharp_matrix_mode
                    ),
                    "rift_budget_ratio": float(args.rift_budget_ratio) if rift_enabled else None,
                    "rift_quantum_bytes": int(args.rift_quantum_bytes) if rift_enabled else None,
                    "rift_staleness_weight": float(args.rift_staleness_weight) if rift_enabled else None,
                    "rift_reliability_floor": float(args.rift_reliability_floor) if rift_enabled else None,
                    "rift_reliability_weight": (
                        0.0 if args.mec_method == "proposed_joint_rift_ef_cora_no_reliability"
                        else 1.0 if rift_enabled else None
                    ),
                    "rift_guard_max_age": rift_guard_max_age if rift_guard_enabled else None,
                    "rift_guard_reserve_ratio": (
                        float(args.rift_guard_reserve_ratio) if rift_guard_enabled else None
                    ),
                    "rift_ef_decay": rift_ef_decay if rift_ef_enabled else None,
                    "rift_duration_value_enabled": (
                        rift_duration_value_enabled if rift_enabled else None
                    ),
                    "fedsonata_gain_target": (
                        fedsonata_gain_target if fedsonata_enabled else None
                    ),
                    "fedsonata_schedule": (
                        fedsonata_schedule if fedsonata_enabled else None
                    ),
                    "fedsonata_high_rounds": (
                        fedsonata_high_rounds if fedsonata_enabled else None
                    ),
                    "fedsonata_tail_budget_ratio": (
                        fedsonata_tail_budget_ratio if fedsonata_enabled else None
                    ),
                    "fedsonata_contraction_q": (
                        fedsonata_contraction_q if fedsonata_enabled else None
                    ),
                    "strict_determinism": bool(args.strict_determinism),
                    "lpc_mode": str(getattr(args, "lpc_mode", "train")),
                    "projection_ridge": (
                        float(args.projection_ridge)
                        if str(getattr(args, "lpc_mode", "train")).lower() == "projection"
                        else None
                    ),
                    "fedra_scope": "a" if fedra_a_only else "native",
                    "reliability_beta": args.mec_reliability_beta,
                    "dataset": args.dataset,
                    "num_classes": int(num_classes),
                    "data_partition": args.data_partition,
                    "alpha": float(args.alpha),
                    "data_fraction": float(args.data_fraction),
                    "client_splits": str(args.client_splits),
                    "client_data_split_hash": client_data_split_hash,
                    "model_name": args.model_name,
                    "lora_rank": int(args.lora_rank),
                    "lora_alpha": float(args.lora_alpha),
                    "heterogeneity_type": args.heterogeneity_type,
                    "allocation_ratio": float(args.allocation_ratio),
                    "fedhello_geometry": (
                        str(args.fedhello_geometry) if method_l == "fedhello" else None
                    ),
                    "fedhello_rgd_rounds": (
                        int(args.fedhello_rgd_rounds) if method_l == "fedhello" else None
                    ),
                    "fedhello_fim_every": (
                        int(args.fedhello_fim_every) if method_l == "fedhello" else None
                    ),
                    "fedhello_fim_proxy_size": (
                        int(args.fedhello_fim_proxy_size) if method_l == "fedhello" else None
                    ),
                    "fedhello_proxy_source": (
                        fedhello_proxy_source_label if method_l == "fedhello" else None
                    ),
                    "reserve_proxy_slice": int(getattr(args, "reserve_proxy_slice", 0)),
                    "batch_size": int(args.batch_size),
                    "test_batch_size": int(args.test_batch_size),
                    "client_lr": float(args.client_lr),
                    "lora_lr": float(args.lora_lr if args.lora_lr is not None else args.client_lr),
                    "classifier_lr": float(
                        args.classifier_lr
                        if args.classifier_lr is not None
                        else (args.lora_lr if args.lora_lr is not None else args.client_lr)
                    ),
                    "num_epochs": int(args.num_epochs),
                    "validation_patience": int(args.validation_patience),
                    "validation_min_delta": float(args.validation_min_delta),
                    "min_rounds_before_stop": int(args.min_rounds_before_stop),
                    "restore_best_validation": bool(args.restore_best_validation),
                    "evaluation_scope": str(args.evaluation_scope),
                    "classifier_policy": str(args.classifier_policy),
                    "network_mode": str(mec_network_policy),
                    "metric_split": "personalized_client_holdout",
                    "accuracy_global_status": (
                        "server_global_test" if args.evaluation_scope == "server_global_test"
                        else "personalized_client_holdout"
                    ),
                    "num_clients": args.num_clients,
                    "num_rounds": args.num_rounds,
                    "clients_per_round": clients_per_round,
                    "client_selection_seed": client_selection_seed,
                    "client_selection_rng": "numpy_randomstate_dedicated",
                    "channel_model": args.mec_channel,
                    "channel_correlation": float(args.mec_channel_correlation),
                    "channel_gain_mean": float(args.mec_channel_gain_mean),
                    "dropout_model": args.mec_dropout_model,
                    "dropout_rate": args.mec_dropout_rate,
                    "ge_good_to_bad": (
                        float(mec_profiles.dropout.ge_good_to_bad)
                        if args.mec_dropout_model == "gilbert_elliott" else None
                    ),
                    "ge_bad_to_good": (
                        float(mec_profiles.dropout.ge_bad_to_good)
                        if args.mec_dropout_model == "gilbert_elliott" else None
                    ),
                    "duration_hazard_rate": float(args.mec_duration_hazard_rate),
                    "total_bandwidth_mhz": args.mec_total_bw_mhz,
                    "power_max_dbm": args.mec_power_max_dbm,
                    "d_min_s": args.mec_d_min,
                    "d_max_s": args.mec_d_max,
                    "lambda_e": args.mec_lambda_e,
                    "lambda_u": args.mec_lambda_u,
                    "deadline_grid_size": int(args.mec_deadline_grid_size),
                    "compression": str(getattr(args, "compression", "none")),
                    "compression_ratio": float(getattr(args, "compression_ratio", 0.1)),
                    "compression_levels": int(getattr(args, "compression_levels", 4)),
                    "deadline_mode": str(getattr(args, "mec_deadline_mode", "fill")),
                    "partial_arrival": bool(getattr(args, "mec_partial_arrival", False)),
                    "omega_net": args.mec_omega_net,
                    "cora_rift_coupling": (
                        "upper_bound_predictor_then_actual_payload_corrector"
                        if rift_enabled and mec_network_policy == "convex_joint"
                        else None
                    ),
                    "cora_rift_reference_rate_source": (
                        REFERENCE_RATE_SOURCE
                        if rift_enabled and mec_network_policy == "convex_joint"
                        else None
                    ),
                    "cora_optimality_scope": (
                        "global fixed-deadline bandwidth/power; exact declared deadline grid; "
                        "conditional quantized RIFT; no end-to-end FL global optimum"
                        if mec_network_policy == "convex_joint" else None
                    ),
                    "cli": raw_argv,
                    "energy_model": "model-based simulated",
                }
                with open(mec_config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                mec_log(f"run started: method={args.mec_method} channel={args.mec_channel} "
                        f"dropout={args.mec_dropout_model}@{args.mec_dropout_rate} "
                        f"bw={args.mec_total_bw_mhz}MHz pmax={args.mec_power_max_dbm}dBm")
            except Exception as e:
                print(f"[警告] 无法写入 MEC config: {e}")

    wire_controller = WireController(args) if args.wire_codec != "none" else None
    for round_num in range(1, args.num_rounds + 1):
        round_start_time = time.time()
        fedsonata_round_policy = None
        effective_rift_budget_ratio = float(args.rift_budget_ratio)
        effective_fedsonata_gain_target = (
            fedsonata_gain_target if fedsonata_enabled else None
        )
        if fedsonata_enabled:
            fedsonata_round_policy = resolve_round_policy(
                schedule=fedsonata_schedule,
                round_number=round_num,
                num_rounds=int(args.num_rounds),
                high_round_count=fedsonata_high_rounds,
                high_budget_ratio=float(args.rift_budget_ratio),
                high_gain_target=fedsonata_gain_target,
                low_budget_ratio=fedsonata_tail_budget_ratio,
                contraction_q=fedsonata_contraction_q,
            )
            effective_rift_budget_ratio = fedsonata_round_policy.budget_ratio
            effective_fedsonata_gain_target = (
                fedsonata_round_policy.gain_target_ratio
            )
        
        print(f"\n{'='*80}")
        print(f"[轮次] {round_num}/{args.num_rounds}")
        print(f"{'='*80}")
        
        # 步骤 1: 服务器生成分配映射
        fedhello_refresh_fim = False
        fedhello_fim_seen = 0
        if method_l == "fedhello":
            fedhello_refresh_fim = should_refresh_fim(
                round_num=round_num,
                rgd_rounds=int(args.fedhello_rgd_rounds),
                fim_every=int(args.fedhello_fim_every),
                has_scores=bool(server.fedhello_fim_scores),
            )
            if fedhello_refresh_fim:
                fim_scores, fedhello_fim_seen = _compute_vision_fedhello_fim_scores(
                    model=global_model,
                    lora_layers=global_lora_layers,
                    dataloader=fedhello_proxy_loader,
                    device=device,
                    max_samples=int(args.fedhello_fim_proxy_size),
                )
                server.set_fedhello_fim_scores(fim_scores, round_num=round_num)
        allocation_map = server.start_round()
        if method_l == "fedhello":
            fedhello_phase = "fim" if bool(server.fedhello_fim_scores) else "rgd"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"round={round_num}, type=fedhello_hla, phase={fedhello_phase}, "
                    f"geometry={server.fedhello_geometry}, refresh_fim={fedhello_refresh_fim}, "
                    f"fim_samples={fedhello_fim_seen}, "
                    f"proxy_source={fedhello_proxy_source_label}, proxy_disjoint_from_reported=True\n"
                )
        server.print_allocation_stats(allocation_map)
        
        # 步骤 2: 选择客户端
        selected_clients = client_selection_rng.choice(
            args.num_clients,
            size=min(clients_per_round, args.num_clients),
            replace=False
        )
        
        print(f"[步骤 2] 选择客户端: 本轮活跃 {len(selected_clients)} / {args.num_clients}")

        # ------------------------------------------------------------------
        # FedHARP-MEC: reliability-aware block allocation for proposed_* methods.
        # For equal_fixed/bapft_ba baselines we keep the learning strategy's
        # native allocation (server.start_round) so block policy is matched.
        # ------------------------------------------------------------------
        mec_alloc = allocation_map
        mec_lpc_mask = {}
        if mec_enabled and args.mec_method and args.mec_method.startswith("proposed"):
            mec_alloc = {}
            st_state = StalenessState(mec_staleness)
            for cid in range(args.num_clients):
                budget = int(server.client_layer_counts.get(cid, max(1, len(mec_layer_names) // 2)))
                if args.mec_method == "proposed_random_blocks":
                    # FedRA-style random block selection (no reliability).
                    import random as _rnd
                    # Python 3.11 rejects tuple seeds.  Use a deterministic
                    # integer mix without process-randomized hash().
                    random_block_seed = (
                        int(args.seed) * 1_000_003
                        + int(round_num) * 10_007
                        + int(cid) * 101
                    ) & ((1 << 63) - 1)
                    _rnd.seed(random_block_seed)
                    blocks = list(mec_layer_names)
                    _rnd.shuffle(blocks)
                    mec_alloc[cid] = set(blocks[:budget])
                else:
                    reliability_weight = 0.0 if args.mec_method in {
                        "proposed_no_reliability",
                        "proposed_joint_rift_ef_cora_no_reliability",
                    } else 1.0
                    raw_sensitivity = server.client_sensitivities.get(cid, {})
                    sensitivity = {
                        ln: max(0.0, float(raw_sensitivity.get(ln, 0.0) or 0.0))
                        for ln in mec_layer_names
                    }
                    max_sensitivity = max(sensitivity.values(), default=0.0)
                    if max_sensitivity <= 1e-12:
                        sensitivity = {ln: 0.5 for ln in mec_layer_names}
                    else:
                        sensitivity = {
                            ln: value / max_sensitivity for ln, value in sensitivity.items()
                        }

                    compute_cost = {}
                    payload_cost = {}
                    for ln in mec_layer_names:
                        layer = global_lora_layers[ln]
                        a_tensor = layer.get_A()
                        b_tensor = layer.get_B()
                        compute_cost[ln] = float(a_tensor.numel() + b_tensor.numel())
                        if fedharp_matrix_mode in {"adaptive", "ab"}:
                            payload_cost[ln] = float(
                                a_tensor.numel() * a_tensor.element_size()
                                + b_tensor.numel() * b_tensor.element_size()
                            )
                        else:
                            payload_cost[ln] = float(
                                a_tensor.numel() * a_tensor.element_size()
                            )
                    max_compute = max(compute_cost.values(), default=1.0)
                    max_payload = max(payload_cost.values(), default=1.0)
                    compute_cost = {
                        ln: value / max(max_compute, 1.0) for ln, value in compute_cost.items()
                    }
                    payload_cost = {
                        ln: value / max(max_payload, 1.0) for ln, value in payload_cost.items()
                    }
                    mec_alloc[cid] = reliability_aware_selection(
                        cid, mec_layer_names, budget, st_state, mec_rel_view,
                        sensitivity, compute_cost, payload_cost,
                        mec_scheduler_cfg, reliability_weight=reliability_weight,
                    )
            allocation_map = mec_alloc
        if wire_controller is not None:
            allocation_map = wire_controller.before_round(round_num, allocation_map, mec_est_success, server, mec_staleness)
        # Pre-compute effective-staleness-driven LPC mask for this round.
        if mec_enabled:
            fs23_score_pool = []
            for cid in range(args.num_clients):
                # LPC is a FedHARP-family mechanism.  Other baselines may use
                # the same MEC metrics object, but they neither train LPC nor
                # expose a meaningful LPC event count.
                lpc_applicable = (
                    is_fedharp_family
                    and b_train_every > 0
                    and args.mec_method != "proposed_no_lpc"
                )
                if lpc_applicable:
                    _trig = str(getattr(args, "lpc_trigger", "age"))
                    _alloc = allocation_map.get(cid, set())
                    for _n in mec_layer_names:
                        _prev = float(fs23_alloc_freq.get((cid, _n), 0.0))
                        fs23_alloc_freq[(cid, _n)] = (
                            0.8 * _prev + 0.2 * (1.0 if _n in _alloc else 0.0)
                        )
                    if _trig == "age":
                        mec_lpc_mask[cid] = lpc_trigger_mask(
                            mec_staleness, cid, _alloc, b_train_every
                        )
                    else:
                        _scored = []
                        for _n in _alloc:
                            _age = mec_staleness.effective(cid, _n)
                            if int(round(_age)) < 1:
                                continue
                            _dn = float(fs23_last_delta_norm.get((cid, _n), 0.0))
                            _rf = float(fs23_alloc_freq.get((cid, _n), 0.0))
                            if _dn * _rf > 0.0:
                                _scored.append((_dn * _rf, cid, _n))
                        if _trig == "value_budget":
                            mec_lpc_mask[cid] = {n for _, _, n in _scored}
                        else:
                            _thr = float(getattr(args, "lpc_score_threshold", 0.0))
                            mec_lpc_mask[cid] = {
                                n for s, _, n in _scored if s > _thr
                            }
                        fs23_score_pool.extend(_scored)
                else:
                    mec_lpc_mask[cid] = set()
            # S3 fix1: GLOBAL per-round correction budget over all candidates.
            _k = int(getattr(args, "lpc_round_budget", 0))
            if _k > 0 and fs23_score_pool:
                fs23_score_pool.sort(reverse=True)
                _keep = {(c, n) for _, c, n in fs23_score_pool[:_k]}
                for _cid in range(args.num_clients):
                    mec_lpc_mask[_cid] = {
                        _n for _n in mec_lpc_mask.get(_cid, set())
                        if (_cid, _n) in _keep
                    }
            elif (
                str(getattr(args, "lpc_trigger", "age")) == "value_budget"
                and _k <= 0
            ):
                # fail-safe: value_budget without K selects nothing (the first
                # screening run showed "no budget == trigger all" is an
                # explosion, not a default).
                for _cid in range(args.num_clients):
                    mec_lpc_mask[_cid] = set()

        # 步骤 3: 客户端接收全局 A 并执行本地训练
        global_A = server.get_global_A_matrices()
        upload_b_method = (
            fedharp_upload_B
            or (str(args.method).lower() in full_b_methods and not fedra_a_only)
        )
        global_B = server.get_global_B_matrices() if upload_b_method else None
        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)
        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0
        downlink_classifier_mb = (
            _dict_tensor_nbytes(global_classifier_state) / (1024.0 * 1024.0)
            if args.classifier_policy == "server_global" else 0.0
        )
        client_updates = {}
        client_updates_B = {}
        client_classifier_states_after = {}
        client_metrics_ab = []
        mec_round_info = []  # collected per-client for joint solve + arrival

        # CORA predictor: RIFT needs a link rate before its selected payload is
        # known.  Build a true-byte upper bound from the declared RIFT cap,
        # solve CORA on that bound, and use the resulting rate in the exact
        # quantized RIFT subproblem.  The final actual-payload CORA solve below
        # is the corrector.  We claim conditional subproblem optimality only.
        cora_rift_preallocation = None
        cora_reference_rates = {}
        if rift_enabled and mec_network_policy == "convex_joint":
            pre_clients = [int(client_id) for client_id in selected_clients]
            pre_h_vec = mec_trace.channel_vector(round_num)
            pre_channel = {
                client_id: float(pre_h_vec[client_id]) for client_id in pre_clients
            }
            pre_compute = {}
            pre_full_ab_bytes = {}
            pre_classifier_bytes = {}
            pre_stale = {}
            for client_id in pre_clients:
                layers = allocation_map.get(client_id, set())
                pre_full_ab_bytes[client_id] = int(sum(
                    _A_matrices_nbytes({name: global_A[name]})
                    + _A_matrices_nbytes({name: global_B[name]})
                    for name in layers
                ))
                pre_classifier_bytes[client_id] = int(
                    _dict_tensor_nbytes(global_classifier_state)
                    if args.classifier_policy == "server_global" else 0
                )
                dev = mec_profiles.device(client_id)
                cycles = float(client_sample_counts.get(client_id, 0)) * (
                    mec_profiles.c0_cycles_per_sample
                    + len(layers) * mec_profiles.c_lora_cycles_per_sample
                )
                pre_compute[client_id] = cycles / dev.freq_hz
                pre_stale[client_id] = float(mec_staleness.effective(
                    client_id,
                    next(iter(layers), mec_layer_names[0]),
                ))
            cora_rift_preallocation = solve_rift_cora_preallocation(
                clients=pre_clients,
                full_ab_bytes=pre_full_ab_bytes,
                classifier_bytes=pre_classifier_bytes,
                compute_time_s=pre_compute,
                channel_gain=pre_channel,
                reliability=mec_rel_view,
                budget_ratio=effective_rift_budget_ratio,
                reliability_floor=float(args.rift_reliability_floor),
                metadata_bits=mec_profiles.s_meta_bits,
                total_bandwidth_hz=mec_profiles.total_bandwidth_hz,
                power=mec_profiles.power_profile,
                n0_w_hz=mec_profiles.channel.n0_w_hz,
                d_min=float(args.mec_d_min),
                d_max=float(args.mec_d_max),
                lambda_e=float(args.mec_lambda_e),
                lambda_u=float(args.mec_lambda_u),
                omega_net=radio_outer_age_weight,
                prev_staleness=pre_stale,
                duration_hazard_rate=mec_hazard_for_round(round_num),
                deadline_grid_size=int(args.mec_deadline_grid_size),
            )
            cora_reference_rates = dict(cora_rift_preallocation.reference_rate_bps)
            print(
                f"  [MEC-CORA predictor] {cora_rift_preallocation.plan.solver_ms:.2f} ms, "
                f"deadline={cora_rift_preallocation.plan.deadline_s:.3f}s, "
                f"rate_source={cora_rift_preallocation.rate_source}"
            )

        for i, client_id in enumerate(selected_clients):
            client_idx = int(client_id)
            client = clients[client_idx]
            allocated_layers = allocation_map[client_idx]
            
            # Snapshot the retained factor state before synchronization for
            # projection-mode LPC (FedHARP Local Projection Corrector). The
            # explicit per-client snapshots are used because the live module
            # may hold a restored evaluation state.
            lpc_projection_old_A = None
            lpc_projection_old_B = None
            if lpc_projection_enabled:
                lpc_projection_old_A = {
                    name: tensor.clone()
                    for name, tensor in client_personal_A_states.get(
                        client_idx, client.get_all_A_matrices()
                    ).items()
                }
                lpc_projection_old_B = {
                    name: tensor.clone()
                    for name, tensor in client_personal_B_states.get(
                        client_idx, client.get_all_B_matrices()
                    ).items()
                }
            client_global_A = global_A
            client_global_B = global_B
            fedharp_b_sharing = is_fedharp_family and fedharp_matrix_mode == "b"
            if fedharp_b_sharing:
                client_global_A = client_personal_A_states[client_idx]
            if method_l in {"flexlora", "flex-lora", "flex_lora"} and global_B is not None:
                first_layer = next(iter(client.lora_layers.values()))
                client_rank = max(1, int(first_layer.lora_A.shape[0]))
                client_global_A, client_global_B = _compress_lora_matrices_to_rank(
                    global_A,
                    global_B,
                    target_rank=client_rank,
                    alpha=float(args.lora_alpha),
                )
            elif method_l in _ZERO_PADDED_RANK_METHODS:
                if hetero_client_ranks is None:
                    raise RuntimeError(f"{args.method} 客户端秩尚未初始化")
                client_global_A, client_global_B = client.prepare_global_matrices_for_rank(
                    global_A,
                    global_B,
                    target_rank=int(hetero_client_ranks[client_idx]),
                )
            if fedharp_b_sharing:
                if client_global_B is None:
                    raise ValueError("FedHARP-B requires the global B matrices")
                client.set_A_matrices(client_global_A)
                client.set_B_matrices(client_global_B)
            else:
                receive_method = "fedharp" if fedra_a_only else args.method
                client.receive_global_matrices(
                    client_global_A,
                    client_global_B,
                    method=receive_method,
                )
            if args.classifier_policy == "server_global" and global_classifier_state:
                client.set_classifier_state(global_classifier_state)
            if method_l == "flora":
                if hetero_client_ranks is None:
                    raise RuntimeError("FLoRA 客户端秩尚未初始化")
                client.flora_merge_and_reset(target_rank=hetero_client_ranks[client_idx])

            client_downlink_A_mb = (
                0.0
                if fedharp_b_sharing
                else _A_matrices_nbytes(client_global_A) / (1024.0 * 1024.0)
            )
            client_downlink_B_mb = (
                _A_matrices_nbytes(client_global_B) / (1024.0 * 1024.0)
                if client_global_B is not None
                else 0.0
            )

            if b_train_every > 0 and is_fedharp_family and method_l != "fedharp_l":
                if mec_enabled:
                    # Effective-staleness-driven LPC (SYSTEM_MODEL sec.5);
                    # network failures also raise LPC demand via mec_staleness.
                    stale_layers = mec_lpc_mask.get(client_idx, set())
                else:
                    stale_map = server.staleness.get(client_idx, {})
                    stale_layers = {
                        name for name, tau in stale_map.items()
                        if int(tau) == int(b_train_every)
                    }
                if len(stale_layers) > 0:
                    if lpc_projection_enabled:
                        # Closed-form FedHARP Local Projection Corrector:
                        # transport the retained factor pair across the new
                        # basis instead of running a task-loss correction pass.
                        if fedharp_matrix_mode == "b":
                            projected_factors, lpc_diag = project_factor_dicts_A(
                                lpc_projection_old_A,
                                lpc_projection_old_B,
                                client.get_all_A_matrices(),
                                client.get_all_B_matrices(),
                                stale_layers,
                                ridge=float(args.projection_ridge),
                            )
                            client.set_A_matrices(projected_factors)
                        else:
                            projected_factors, lpc_diag = project_factor_dicts(
                                lpc_projection_old_A,
                                lpc_projection_old_B,
                                client.get_all_A_matrices(),
                                client.get_all_B_matrices(),
                                stale_layers,
                                ridge=float(args.projection_ridge),
                            )
                            client.set_B_matrices(projected_factors)
                        lpc_projection_events += len(lpc_diag)
                        for _lpc_layer, _lpc_d in sorted(lpc_diag.items()):
                            print(
                                f"round={round_num}, type=lpc_projection_event, "
                                f"client={client_idx}, layer={_lpc_layer}, "
                                f"mode={fedharp_matrix_mode}, "
                                f"ridge={float(args.projection_ridge):.8g}, "
                                f"residual_before={float(_lpc_d['residual_before']):.8g}, "
                                f"residual_after={float(_lpc_d['residual_after']):.8g}, "
                                f"relative_reduction={float(_lpc_d['relative_reduction']):.8g}"
                            )
                    elif fedharp_matrix_mode == "b":
                        # Symmetric LPC realigns stale local A against the
                        # newly downloaded global B. DSA remains based on the
                        # B-gradient measured during standard local training.
                        client.train_A_only(stale_layers, num_epochs=args.b_num_epochs)
                    else:
                        new_b_sensitivity = client.train_B_only(
                            stale_layers,
                            num_epochs=args.b_num_epochs,
                        )
                        server.update_client_sensitivities(client_idx, new_b_sensitivity)
            
            # 本地训练
            if i < 3:
                print(f"  [步骤 3] 客户端 {client_idx} 本地训练: 分配层数={len(allocated_layers)}")

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            classifier_before = client.get_classifier_state()
            delta_A, delta_B, b_sensitivity = client.local_train(
                allocated_layers=allocated_layers,
                num_epochs=args.num_epochs,
                method=args.method,
                matrix_update_mode=fedharp_matrix_mode,
                adaptive_matrix_threshold=args.adaptive_matrix_threshold,
            )
            train_time_sec = time.time() - t0
            classifier_after = client.get_classifier_state()
            if classifier_after:
                client_classifier_states_after[client_idx] = classifier_after
            if torch.cuda.is_available() and device.type == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
            else:
                peak_mem_gb = 0.0

            rift_selection: Optional[RiftSelection] = None
            rift_selection_ms = 0.0
            rift_guard_mandatory_count = 0
            rift_guard_max_age_after = 0
            rift_ef_corrected_A = None
            rift_ef_corrected_B = None
            rift_ef_residual_before_l2 = 0.0
            rift_delivery_rate_bps = None
            rift_delivery_rate_source = None
            if rift_enabled:
                if delta_B is None:
                    raise ValueError("RIFT requires full local A+B deltas before selection")
                a_after = client.get_all_A_matrices()
                b_after = client.get_all_B_matrices()
                selection_delta_A = delta_A
                selection_delta_B = delta_B
                selection_a_after = a_after
                selection_b_after = b_after
                if rift_ef_enabled:
                    rift_ef_corrected_A, rift_ef_corrected_B, rift_ef_residual_before_l2 = (
                        rift_ef_state.corrected(client_idx, delta_A, delta_B)
                    )
                    selection_delta_A = rift_ef_corrected_A
                    selection_delta_B = rift_ef_corrected_B
                    selection_a_after = {
                        name: a_after[name] + (selection_delta_A[name] - delta_A[name])
                        for name in selection_delta_A
                    }
                    selection_b_after = {
                        name: b_after[name] + (selection_delta_B[name] - delta_B[name])
                        for name in selection_delta_B
                    }
                if rift_duration_value_enabled:
                    if client_idx in cora_reference_rates:
                        rift_delivery_rate_bps = float(cora_reference_rates[client_idx])
                        rift_delivery_rate_source = REFERENCE_RATE_SOURCE
                    else:
                        # Legacy D026--D032 protocol retained only for the
                        # explicitly named legacy-radio comparison.
                        equal_bw = mec_profiles.total_bandwidth_hz / max(1, len(selected_clients))
                        current_h = mec_trace.channel_gain(round_num, client_idx)
                        rift_delivery_rate_bps = shannon_rate(
                            equal_bw,
                            mec_profiles.power_profile.p_max_w,
                            current_h,
                            mec_profiles.channel.n0_w_hz,
                        )
                        rift_delivery_rate_source = "legacy_equal_bandwidth_pmax_estimate"
                else:
                    rift_delivery_rate_source = "disabled_ablation"
                stale_for_client = {
                    name: float(mec_staleness.effective(client_idx, name))
                    for name in allocated_layers
                }
                rift_candidate_names = sorted(set(selection_delta_A).intersection(selection_delta_B))
                mandatory_layers = (
                    oldest_guard_layers(
                        rift_candidate_names,
                        rift_guard_ages[client_idx],
                        rift_guard_max_age,
                    ) if rift_guard_enabled else []
                )
                rift_t0 = time.perf_counter()
                wire_selector = wire_controller.select if wire_controller is not None else select_rift_updates
                wire_kwargs = {"round_num": round_num, "client_id": client_idx} if wire_controller is not None else {}
                if args.wire_codec in {"lossaware_rift", "lookahead_rift", "balanced_lookahead_rift", "counterfactual_rift", "edge_random_rift"}:
                    # The gradient proxy is computed from the same local
                    # training minibatches and is passed only to the matched
                    # raw-wire selector.  The old random arm does not see it.
                    wire_kwargs.update({
                        "grad_W": dict(getattr(client, "last_effective_gradients", {}) or {}),
                        "loss_aware_age_limit": 999,
                        "factor_debt": (
                            wire_controller.factor_debt.get(client_idx, {})
                            if wire_controller is not None else {}
                        ),
                    })
                if args.wire_codec == "counterfactual_rift":
                    # Score candidate action plans on a deterministic batch
                    # drawn from the client's training split.  This is a
                    # training-only counterfactual objective; no validation or
                    # test samples enter selection.
                    _cf_names = sorted(set(selection_delta_A).intersection(selection_delta_B))
                    _cf_before_A = {
                        name: selection_a_after[name] - selection_delta_A[name]
                        for name in _cf_names
                    }
                    _cf_before_B = {
                        name: selection_b_after[name] - selection_delta_B[name]
                        for name in _cf_names
                    }
                    _cf_data = _cf_target = None
                    try:
                        _cf_dataset = client.train_loader.dataset
                        _cf_items = [_cf_dataset[j] for j in range(min(16, len(_cf_dataset)))]
                        _cf_data = torch.stack([item[0] for item in _cf_items]).to(device)
                        _cf_target = torch.as_tensor([int(item[1]) for item in _cf_items], device=device)
                    except Exception:
                        _cf_data = _cf_target = None

                    def _counterfactual_score(candidate_A, candidate_B):
                        if _cf_data is None or _cf_target is None:
                            return float("inf")
                        try:
                            absolute_A = {
                                name: _cf_before_A[name] + candidate_A.get(name, torch.zeros_like(_cf_before_A[name]))
                                for name in _cf_names
                            }
                            absolute_B = {
                                name: _cf_before_B[name] + candidate_B.get(name, torch.zeros_like(_cf_before_B[name]))
                                for name in _cf_names
                            }
                            client.set_A_matrices(absolute_A)
                            client.set_B_matrices(absolute_B)
                            client.model.eval()
                            with torch.no_grad():
                                logits = client.model(_cf_data)
                                value = torch.nn.functional.cross_entropy(logits, _cf_target)
                            return float(value.detach().cpu().item())
                        except Exception:
                            return float("inf")
                        finally:
                            try:
                                client.set_A_matrices(selection_a_after)
                                client.set_B_matrices(selection_b_after)
                                client.model.train()
                            except Exception:
                                pass

                    wire_kwargs["counterfactual_score"] = _counterfactual_score
                rift_selection = wire_selector(
                    **wire_kwargs,
                    a_after=selection_a_after,
                    b_after=selection_b_after,
                    delta_a=selection_delta_A,
                    delta_b=selection_delta_B,
                    budget_ratio=effective_rift_budget_ratio,
                    reliability=float(mec_rel_view[client_idx]),
                    staleness=stale_for_client,
                    staleness_weight=float(args.rift_staleness_weight),
                    reliability_floor=float(args.rift_reliability_floor),
                    quantum_bytes=int(args.rift_quantum_bytes),
                    mandatory_layers=mandatory_layers,
                    guard_reserve_ratio=(
                        float(args.rift_guard_reserve_ratio) if rift_guard_enabled else 0.0
                    ),
                    delivery_rate_bps=rift_delivery_rate_bps,
                    cap_hazard_rate=float(mec_hazard_for_round(round_num)),
                    duration_hazard_rate=(
                        float(mec_hazard_for_round(round_num))
                        if rift_duration_value_enabled else 0.0
                    ),
                    gain_target_ratio=(
                        effective_fedsonata_gain_target if fedsonata_enabled else None
                    ),
                )
                rift_selection_ms = (time.perf_counter() - rift_t0) * 1000.0
                delta_A = rift_selection.updates_A
                delta_B = rift_selection.updates_B
                for _n, _t in (selection_delta_A or {}).items():
                    fs23_last_delta_norm[(client_idx, _n)] = float(
                        _t.detach().float().norm()
                    )
                for _n, _t in (selection_delta_B or {}).items():
                    fs23_last_delta_norm[(client_idx, _n)] = max(
                        float(fs23_last_delta_norm.get((client_idx, _n), 0.0)),
                        float(_t.detach().float().norm()),
                    )
                if rift_guard_enabled:
                    for name in rift_candidate_names:
                        if rift_selection.choices.get(name, "none") == "none":
                            rift_guard_ages[client_idx][name] = (
                                int(rift_guard_ages[client_idx].get(name, 0)) + 1
                            )
                        else:
                            rift_guard_ages[client_idx][name] = 0
                    rift_guard_mandatory_count = int(rift_selection.guard_mandatory_count)
                    rift_guard_max_age_after = max(
                        (int(rift_guard_ages[client_idx].get(name, 0)) for name in rift_candidate_names),
                        default=0,
                    )
                    if rift_guard_max_age_after >= rift_guard_max_age:
                        raise AssertionError(
                            f"RIFT-Guard age bound violated: {rift_guard_max_age_after} "
                            f">= {rift_guard_max_age}"
                        )
                if i < 3:
                    print(
                        f"  [RIFT] 客户端 {client_idx}: bytes="
                        f"{rift_selection.selected_bytes}/{rift_selection.budget_bytes}, "
                        f"gain={rift_selection.captured_gain_ratio:.3f}, "
                        f"time={rift_selection_ms:.2f}ms, choices={rift_selection.option_counts}, "
                        f"guard_mandatory={rift_guard_mandatory_count}, "
                        f"guard_max_age={rift_guard_max_age_after}"
                    )

            compressed_bits_A = 0
            compressed_bits_B = 0
            if compression_method != "none":
                _res = compression_residual.setdefault(client_idx, {})
                delta_A, _bits_a, _res_a = compress_updates_dict(
                    delta_A, compression_method, ratio=compression_ratio,
                    levels=compression_levels, residual=_res.get("A"),
                )
                _res["A"] = _res_a
                compressed_bits_A = int(_bits_a)
                if upload_b_method and delta_B is not None:
                    delta_B, _bits_b, _res_b = compress_updates_dict(
                        delta_B, compression_method, ratio=compression_ratio,
                        levels=compression_levels, residual=_res.get("B"),
                    )
                    _res["B"] = _res_b
                    compressed_bits_B = int(_bits_b)
            compressed_bits_round = compressed_bits_A + compressed_bits_B
            # True-byte accounting: with compression enabled the transmitted
            # payload is the compressed size, not the dense tensor size, so the
            # MEC byte budget reflects what is really sent.
            if compression_method != "none":
                upload_A_bytes_eff = int(round(compressed_bits_A / 8.0))
                upload_B_bytes_eff = int(round(compressed_bits_B / 8.0))
            else:
                upload_A_bytes_eff = _delta_A_nbytes(delta_A)
                upload_B_bytes_eff = (
                    _delta_B_nbytes(delta_B)
                    if (upload_b_method and delta_B is not None) else 0
                )
            if wire_controller is not None:
                # A/B share one codec container; these legacy columns are not factor-cost attribution.
                upload_A_bytes_eff = int(rift_selection.selected_bytes)
                upload_B_bytes_eff = 0
            uplink_A_mb = upload_A_bytes_eff / (1024.0 * 1024.0)
            if upload_b_method and delta_B is None:
                raise ValueError(f"{args.method} 需要上传 B，但 delta_B 为空")
            uplink_B_mb = upload_B_bytes_eff / (1024.0 * 1024.0)
            uplink_classifier_mb = (
                _dict_tensor_nbytes(classifier_after) / (1024.0 * 1024.0)
                if args.classifier_policy == "server_global" else 0.0
            )
            uplink_mb = uplink_A_mb + uplink_B_mb + uplink_classifier_mb
            uploaded_A_layers = set(delta_A.keys())
            uploaded_B_layers = set(delta_B.keys()) if (upload_b_method and delta_B is not None) else set()
            upload_ab_blocks = len(uploaded_A_layers & uploaded_B_layers)
            upload_a_only_blocks = len(uploaded_A_layers - uploaded_B_layers)
            upload_b_only_blocks = len(uploaded_B_layers - uploaded_A_layers)
            comm_mb = (
                client_downlink_A_mb + client_downlink_B_mb
                + downlink_classifier_mb + uplink_mb
            )
            train_samples = len(client.train_loader.dataset)
            trainable_numel = _ab_trainable_param_numel(client.lora_layers, allocated_layers)
            compute_flops = 3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)
            compute_tflops = compute_flops / 1e12
            flops_per_sec = compute_flops / max(1e-9, float(train_time_sec))
            first_client_layer = next(iter(client.lora_layers.values()), None)
            active_lora_rank = (
                int(first_client_layer.lora_A.shape[0])
                if first_client_layer is not None
                else 0
            )

            client_metrics_ab.append({
                "round": round_num,
                "client": client_idx,
                "phase": "ab_train",
                "lora_rank": int(active_lora_rank),
                "compute_tflops": float(compute_tflops),
                "compute_flops": float(compute_flops),
                "flops_per_sec": float(flops_per_sec),
                "peak_mem_gb": float(peak_mem_gb),
                "comm_mb": float(comm_mb),
                "train_time_sec": float(train_time_sec),
                "downlink_mb": float(
                    client_downlink_A_mb + client_downlink_B_mb + downlink_classifier_mb
                ),
                "uplink_mb": float(uplink_mb),
                "uplink_A_mb": float(uplink_A_mb),
                "uplink_B_mb": float(uplink_B_mb),
                "uplink_classifier_mb": float(uplink_classifier_mb),
                "upload_A_only_blocks": int(upload_a_only_blocks),
                "upload_B_only_blocks": int(upload_b_only_blocks),
                "upload_AB_blocks": int(upload_ab_blocks),
                "rift_none_blocks": int(rift_selection.option_counts["none"]) if rift_selection else 0,
                "rift_budget_bytes": int(rift_selection.budget_bytes) if rift_selection else 0,
                "rift_selected_bytes": int(rift_selection.selected_bytes) if rift_selection else 0,
                "rift_captured_gain": float(rift_selection.captured_gain) if rift_selection else 0.0,
                "rift_oracle_gain": float(rift_selection.oracle_gain) if rift_selection else 0.0,
                "rift_expected_captured_gain": (
                    float(rift_selection.expected_captured_gain) if rift_selection else 0.0
                ),
                "fedsonata_cap_optimal_expected_gain": (
                    float(rift_selection.cap_optimal_expected_gain) if rift_selection else 0.0
                ),
                "fedsonata_target_attainment_ratio": (
                    float(rift_selection.target_attainment_ratio) if rift_selection else 1.0
                ),
                "fedsonata_gain_target_met": (
                    bool(rift_selection.gain_target_met) if rift_selection else False
                ),
                "rift_delivery_probability_estimate": (
                    float(rift_selection.delivery_probability_estimate)
                    if rift_selection else 1.0
                ),
                "rift_selection_ms": float(rift_selection_ms),
                "rift_guard_mandatory_count": int(rift_guard_mandatory_count),
                "rift_guard_max_age_after": int(rift_guard_max_age_after),
                "rift_ef_residual_before_l2": float(rift_ef_residual_before_l2),
                "rift_delivery_rate_bps": (
                    float(rift_delivery_rate_bps) if rift_delivery_rate_bps is not None else None
                ),
                "rift_delivery_rate_source": rift_delivery_rate_source,
            })
            
            client_updates[client_idx] = delta_A

            if upload_b_method:
                client_updates_B[client_idx] = delta_B

            client_personal_A_states[client_idx] = client.get_all_A_matrices()
            client_personal_B_states[client_idx] = client.get_all_B_matrices()
            client_personal_classifier_states[client_idx] = client.get_classifier_state()
            
            # 上传 B 矩阵敏感度
            server.update_client_sensitivities(client_idx, b_sensitivity)

            # ------------------------------------------------------------------
            # FedHARP-MEC: collect per-client quantities for the joint resource
            # solve + arrival gate. Modeled payload uses the REAL delta_A bytes
            # (block selection changes size) plus s_meta control overhead.
            # ------------------------------------------------------------------
            if mec_enabled:
                active_blocks = len(allocated_layers)
                # Actual transmitted tensor payload. Adaptive A/B may upload
                # A, B, or both for each selected block.
                # Compressed (true) payload bytes when compression is enabled.
                delta_A_nbytes = upload_A_bytes_eff
                delta_B_nbytes = upload_B_bytes_eff
                classifier_upload_nbytes = (
                    _dict_tensor_nbytes(classifier_after)
                    if args.classifier_policy == "server_global" else 0
                )
                delta_upload_nbytes = (
                    delta_A_nbytes + delta_B_nbytes + classifier_upload_nbytes
                )
                payload_bits = mec_profiles.s_meta_bits + 8.0 * float(delta_upload_nbytes)
                # modeled compute cycles from profile freq (separate from the
                # measured wall-clock train_time_sec)
                dev = mec_profiles.device(client_idx)
                cycles = (train_samples * (
                    mec_profiles.c0_cycles_per_sample
                    + active_blocks * mec_profiles.c_lora_cycles_per_sample
                ))
                t_cmp_model = cycles / dev.freq_hz
                mec_round_info.append({
                    "cid": client_idx,
                    "active_blocks": active_blocks,
                    "payload_bits": payload_bits,
                    "t_cmp_model": t_cmp_model,
                    "delta_A": delta_A,
                    "delta_B": delta_B,
                    "upload_bytes": float(delta_upload_nbytes),
                    "compressed_bits": int(compressed_bits_round),
                    "upload_A_bytes": float(delta_A_nbytes),
                    "upload_B_bytes": float(delta_B_nbytes),
                    "upload_A_only_blocks": int(upload_a_only_blocks),
                    "upload_B_only_blocks": int(upload_b_only_blocks),
                    "upload_AB_blocks": int(upload_ab_blocks),
                    "rift_none_blocks": int(rift_selection.option_counts["none"]) if rift_selection else 0,
                    "rift_budget_bytes": int(rift_selection.budget_bytes) if rift_selection else 0,
                    "rift_selected_bytes": int(rift_selection.selected_bytes) if rift_selection else 0,
                    "rift_captured_gain": float(rift_selection.captured_gain) if rift_selection else 0.0,
                    "rift_oracle_gain": float(rift_selection.oracle_gain) if rift_selection else 0.0,
                    "rift_expected_captured_gain": (
                        float(rift_selection.expected_captured_gain) if rift_selection else 0.0
                    ),
                    "fedsonata_cap_optimal_expected_gain": (
                        float(rift_selection.cap_optimal_expected_gain) if rift_selection else 0.0
                    ),
                    "fedsonata_target_attainment_ratio": (
                        float(rift_selection.target_attainment_ratio) if rift_selection else 1.0
                    ),
                    "fedsonata_gain_target_met": (
                        bool(rift_selection.gain_target_met) if rift_selection else False
                    ),
                    "rift_delivery_probability_estimate": (
                        float(rift_selection.delivery_probability_estimate)
                        if rift_selection else 1.0
                    ),
                    "rift_selection_ms": float(rift_selection_ms),
                    "rift_guard_mandatory_count": int(rift_guard_mandatory_count),
                    "rift_guard_max_age_after": int(rift_guard_max_age_after),
                    "rift_choices": (dict(rift_selection.choices) if rift_selection else {}),
                    "rift_ef_corrected_A": rift_ef_corrected_A,
                    "rift_ef_corrected_B": rift_ef_corrected_B,
                    "rift_ef_residual_before_l2": float(rift_ef_residual_before_l2),
                    "rift_delivery_rate_bps": (
                        float(rift_delivery_rate_bps) if rift_delivery_rate_bps is not None else None
                    ),
                    "rift_delivery_rate_source": rift_delivery_rate_source,
                })

            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        "round={round}, type=client_metrics, phase={phase}, client={client}, "
                        "lora_rank={lora_rank}, "
                        "compute_flops={compute_flops:.3f}, compute_tflops={compute_tflops:.6f}, "
                        "flops_per_sec={flops_per_sec:.3f}, peak_mem_gb={peak_mem_gb:.6f}, "
                        "comm_mb={comm_mb:.6f}, train_time_sec={train_time_sec:.6f}, "
                        "downlink_mb={downlink_mb:.6f}, uplink_mb={uplink_mb:.6f}, "
                        "uplink_A_mb={uplink_A_mb:.6f}, uplink_B_mb={uplink_B_mb:.6f}, "
                        "upload_A_only_blocks={upload_A_only_blocks}, "
                        "upload_B_only_blocks={upload_B_only_blocks}, "
                        "upload_AB_blocks={upload_AB_blocks}\n".format(**client_metrics_ab[-1])
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")

        if len(client_metrics_ab) > 0:
            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics_ab]))
            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics_ab]))
            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics_ab]))
            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))
            ranks = [int(m["lora_rank"]) for m in client_metrics_ab]
            round_a_only = sum(int(m["upload_A_only_blocks"]) for m in client_metrics_ab)
            round_b_only = sum(int(m["upload_B_only_blocks"]) for m in client_metrics_ab)
            round_ab = sum(int(m["upload_AB_blocks"]) for m in client_metrics_ab)
            print(
                f"[统计] 本轮客户端均值: 计算={avg_compute:.4f} TFLOPs, "
                f"显存峰值={avg_mem:.4f} GB, 通信={avg_comm:.2f} MB/round, "
                f"训练时间={avg_time:.2f} s/round, "
                f"LoRA秩={np.mean(ranks):.2f}[{min(ranks)},{max(ranks)}]"
            )
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=round_metrics, phase=ab_train, "
                        f"avg_lora_rank={np.mean(ranks):.4f}, min_lora_rank={min(ranks)}, max_lora_rank={max(ranks)}, "
                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "
                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}, "
                        f"upload_A_only_blocks={round_a_only}, upload_B_only_blocks={round_b_only}, "
                        f"upload_AB_blocks={round_ab}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")
        
        # ------------------------------------------------------------------
        # FedHARP-MEC: joint resource solve + arrival gate + staleness update.
        # Only updates that arrive on time enter aggregation (SYSTEM_MODEL
        # sec.4). Non-arrived updates are removed from client_updates so the
        # aggregator never sees them. Effective staleness is updated for the
        # NEXT round's block selection / LPC trigger.
        # ------------------------------------------------------------------
        mec_round_summary = None
        if mec_enabled and len(mec_round_info) > 0:
            clients_k = [c["cid"] for c in mec_round_info]
            s_up = {c["cid"]: c["payload_bits"] for c in mec_round_info}
            t_cmp = {c["cid"]: c["t_cmp_model"] for c in mec_round_info}
            # Same trace realization for this round (deterministic per seed).
            h_vec = mec_trace.channel_vector(round_num)
            d_vec = mec_trace.dropout_vector(round_num)
            h = {c["cid"]: float(h_vec[c["cid"]]) for c in mec_round_info}
            base_dropped = {c["cid"]: bool(d_vec[c["cid"]]) for c in mec_round_info}
            dropped = dict(base_dropped)

            prev_stale = {
                c["cid"]: float(mec_staleness.effective(c["cid"],
                    next(iter(allocation_map.get(c["cid"], set())), mec_layer_names[0])))
                for c in mec_round_info
            }
            n0 = mec_profiles.channel.n0_w_hz

            deadline = args.mec_d_max
            solver_ms_round = 0.0
            bandwidth = {k: mec_profiles.total_bandwidth_hz / len(clients_k) for k in clients_k}
            power_w = {k: mec_profiles.power_profile.p_max_w for k in clients_k}
            resource_objective = float("nan")
            if cora_rift_preallocation is not None and compression_method == "none":
                # The CORA predictor bound is defined for the dense RIFT payload;
                # compression baselines re-encode the same update, so the bound is
                # not a like-for-like constraint and is skipped when compression is
                # enabled.
                for item in mec_round_info:
                    upper = float(cora_rift_preallocation.payload_upper_bits[item["cid"]])
                    if float(item["payload_bits"]) > upper + 1e-9:
                        raise AssertionError(
                            "CORA predictor payload bound violated for client "
                            f"{item['cid']}: {item['payload_bits']} > {upper}"
                        )
            unserved_cids: set = set()
            if args.mec_method != "proposed_equal_bandwidth":
                if radio_policy == "convex_joint":
                    plan = solve_convex_round(
                        clients_k, s_up, t_cmp, h,
                        mec_profiles.total_bandwidth_hz,
                        mec_profiles.power_profile, n0,
                        args.mec_d_min, args.mec_d_max,
                        lambda_e=float(args.mec_lambda_e),
                        lambda_u=float(args.mec_lambda_u),
                        omega_net=radio_outer_age_weight,
                        prev_staleness=prev_stale,
                        est_success=mec_rel_view,
                        duration_hazard_rate=float(args.mec_duration_hazard_rate),
                        deadline_grid_size=int(args.mec_deadline_grid_size),
                        transmit_mode=str(getattr(args, "mec_deadline_mode", "fill")),
                        allow_partial_arrival=bool(getattr(args, "mec_partial_arrival", False)),
                    )
                    deadline = plan.deadline_s
                    bandwidth = dict(plan.bandwidth_hz)
                    power_w = dict(plan.power_w)
                    # Partial-arrival fallback: clients the solver did not serve
                    # (zero bandwidth) still attempt an equal-share best-effort
                    # upload at max power.  They finish after the deadline and are
                    # reported as deadline_miss -- the same degradation semantics
                    # used on the text side (D052).
                    if bool(getattr(args, "mec_partial_arrival", False)):
                        equal_bw = float(mec_profiles.total_bandwidth_hz) / max(1, len(clients_k))
                        for _cid in clients_k:
                            if float(bandwidth.get(_cid, 0.0)) <= 0.0:
                                bandwidth[_cid] = equal_bw
                                power_w[_cid] = float(mec_profiles.power_profile.p_max_w)
                                unserved_cids.add(_cid)
                    solver_ms_round = float(plan.solver_ms)
                    resource_objective = float(plan.objective)
                    print(
                        f"  [MEC-CORA] solver {plan.solver_ms:.2f} ms, "
                        f"feasible={plan.feasible}, deadline={deadline:.3f}s"
                    )
                    if not plan.feasible:
                        raise RuntimeError(
                            "convex resource solver infeasible within deadline bounds "
                            f"[{args.mec_d_min}, {args.mec_d_max}]"
                        )
                elif radio_policy == "proposed_joint" and args.mec_method != "proposed_fixed_deadline":
                    plan = solve_round(
                        clients_k, s_up, t_cmp, h,
                        mec_profiles.total_bandwidth_hz, mec_profiles.power_profile,
                        n0, args.mec_d_min, args.mec_d_max, args.mec_lambda_e,
                        radio_outer_age_weight, prev_stale, mec_rel_view,
                        lambda_u=float(args.mec_lambda_u),
                    )
                    deadline = plan.deadline_s
                    bandwidth = plan.bandwidth_hz
                    power_w = plan.power_w
                    solver_ms_round = float(plan.solver_ms)
                    print(f"  [MEC] solver {plan.solver_ms:.2f} ms, feasible={plan.feasible}, "
                          f"deadline={deadline:.3f}s")
                    if not plan.feasible:
                        raise RuntimeError(
                            f"resource solver infeasible within deadline bounds "
                            f"[{args.mec_d_min}, {args.mec_d_max}]"
                        )
                elif radio_policy == "bapft_ba":
                    # BAPFT-style: allocate bandwidth so all non-dropped clients
                    # finish simultaneously at equal power (no reliability score).
                    deadline = max(t_cmp[c] for c in clients_k) + 1.0
                    for c in clients_k:
                        t_avail = deadline - t_cmp[c]
                        if t_avail <= 0:
                            continue
                        # min bandwidth at p_max to meet deadline
                        from mec.resource_solver import _min_bandwidth_for_power
                        b_req = _min_bandwidth_for_power(
                            s_up[c], mec_profiles.power_profile.p_max_w, h[c], n0, t_avail
                        )
                        bandwidth[c] = min(b_req, mec_profiles.total_bandwidth_hz) if math.isfinite(b_req) else mec_profiles.total_bandwidth_hz
                        power_w[c] = mec_profiles.power_profile.p_max_w
                    # renormalise to total budget
                    tot = sum(bandwidth.values())
                    if tot > mec_profiles.total_bandwidth_hz:
                        for c in clients_k:
                            bandwidth[c] *= mec_profiles.total_bandwidth_hz / tot

            # Arrival gate + per-client modeled quantities.
            arrived_any = False
            for c in mec_round_info:
                cid = c["cid"]
                _bw = float(bandwidth[cid])
                _pw = float(power_w[cid])
                rate = (
                    shannon_rate(_bw, _pw, h[cid], n0)
                    if (_bw > 0.0 and _pw > 0.0)
                    else 0.0
                )
                t_up = c["payload_bits"] / rate if rate > 0 else float("inf")
                duration_probability = (
                    1.0 - math.exp(-mec_hazard_for_round(round_num) * t_up)
                    if math.isfinite(t_up) else 1.0
                )
                duration_event = mec_trace.duration_failure(
                    round_num,
                    cid,
                    t_up,
                    mec_hazard_for_round(round_num),
                )
                duration_dropped = bool(
                    (not base_dropped[cid])
                    and duration_event
                    and cid not in unserved_cids
                )
                dropped[cid] = bool(base_dropped[cid] or duration_dropped)
                c["base_dropped"] = bool(base_dropped[cid])
                c["duration_dropped"] = bool(duration_dropped)
                c["duration_failure_probability"] = float(duration_probability)
                c["rift_delivery_probability_corrector"] = float(
                    math.exp(-mec_hazard_for_round(round_num) * t_up)
                    if math.isfinite(t_up) else 0.0
                )
                res = model_client(
                    client_id=cid, active_blocks=c["active_blocks"], lpc_blocks=0,
                    n_samples=1,
                    device=mec_profiles.device(cid),
                    power=mec_profiles.power_profile,
                    h=h[cid], n0_w_hz=n0,
                    bandwidth_hz=bandwidth[cid], tx_power_w=power_w[cid],
                    dropout_flag=dropped[cid], deadline_s=deadline,
                    s_meta_bits=mec_profiles.s_meta_bits,
                    s_block_a_bits=mec_profiles.s_block_a_bits,
                    t_cmp_s=c["t_cmp_model"],
                )
                # Override modeled payload with the REAL delta_A bytes (block
                # selection changes size); recompute uplink latency/energy.
                res.payload_bits = c["payload_bits"]
                if base_dropped[cid]:
                    attempted_t_up = 0.0
                elif duration_dropped:
                    attempted_t_up = min(
                        t_up,
                        mec_trace.duration_failure_time(
                            round_num, cid, mec_hazard_for_round(round_num)
                        ),
                    )
                else:
                    attempted_t_up = min(
                        t_up, max(0.0, deadline - c["t_cmp_model"])
                    )
                transmitted_bits = min(
                    float(c["payload_bits"]), max(0.0, rate * attempted_t_up)
                )
                res.t_up_s = attempted_t_up
                res.e_tx_j = (power_w[cid] / mec_profiles.power_profile.pa_efficiency
                              + mec_profiles.power_profile.circuit_power_w) * attempted_t_up
                c["attempted_t_up_s"] = float(attempted_t_up)
                c["attempted_tx_energy_j"] = float(res.e_tx_j)
                c["transmitted_bits"] = float(transmitted_bits)
                arrived = (
                    (not dropped[cid])
                    and (c["t_cmp_model"] + t_up) <= deadline + 1e-9
                )
                res.arrived = arrived
                res.failure_type = (
                    "dropout" if base_dropped[cid]
                    else ("duration_dropout" if duration_dropped
                          else ("deadline_miss" if not arrived else "none"))
                )
                res.dropped = dropped[cid]
                c["failure_type_final"] = str(res.failure_type)
                c["arrived_final"] = bool(arrived)
                if rift_ef_enabled:
                    if wire_controller is not None:
                        ef_stats = wire_controller.record_arrival(
                            round_num, cid, arrived, c["delta_A"], c["delta_B"],
                            c["rift_ef_corrected_A"] or {}, c["rift_ef_corrected_B"] or {}, rift_ef_state,
                        )
                    else:
                        ef_stats = rift_ef_state.acknowledge(
                            cid, c["rift_ef_corrected_A"] or {}, c["rift_ef_corrected_B"] or {},
                            c.get("rift_choices", {}), arrived=arrived,
                        )
                    c["rift_ef_cleared_A"] = int(ef_stats["cleared_a"])
                    c["rift_ef_cleared_B"] = int(ef_stats["cleared_b"])
                if not arrived:
                    # Remove from aggregation inputs (must NOT enter aggregator).
                    client_updates.pop(cid, None)
                    client_updates_B.pop(cid, None)
                else:
                    arrived_any = True
                # Update effective staleness (allocation + delivery) for next round.
                update_effective_staleness(
                    mec_staleness, cid, allocation_map.get(cid, set()),
                    mec_layer_names, arrived, res.failure_type,
                )
                update_success_ewma(
                    mec_est_success,
                    cid,
                    arrived,
                    beta=float(args.mec_reliability_beta),
                )

            # FS23: per-round raw vs consumer-visible state record.
            try:
                _sv_rows = []
                for (_c_id, _b_name) in sorted(mec_staleness.alloc.keys()):
                    if mec_layer_names and _b_name not in mec_layer_names:
                        continue
                    _sv_rows.append({
                        "client": int(_c_id),
                        "block": _b_name,
                        "raw_alloc": int(mec_staleness.alloc.get((_c_id, _b_name), 0)),
                        "raw_net": int(mec_staleness.net.get((_c_id, _b_name), 0)),
                        "raw_effective": float(mec_staleness.raw_effective(_c_id, _b_name)),
                        "visible_effective": float(mec_staleness.effective(_c_id, _b_name)),
                    })
                _sv_fail = {}
                for _c in mec_round_info:
                    _ft = str(_c.get("failure_type_final", "unknown"))
                    _sv_fail[_ft] = _sv_fail.get(_ft, 0) + 1
                with open(os.path.join(mec_run_output_dir, "STATE_VIEWS.jsonl"), "a",
                          encoding="utf-8") as _sv_fh:
                    _sv_fh.write(json.dumps({
                        "round": int(round_num),
                        "reliability_raw": {
                            str(k): float(v) for k, v in mec_est_success.items()
                        },
                        "reliability_visible": {
                            str(k): float(mec_rel_view[k]) for k in mec_est_success
                        },
                        "age_rows": _sv_rows,
                        "failure_counts": _sv_fail,
                    }) + "\n")
            except Exception as _sv_exc:
                mec_log(f"STATE_VIEWS write failed: {_sv_exc}")
            if wire_controller is not None:
                wire_controller.finish_round()
            download_bytes = (
                downlink_A_mb + downlink_B_mb + downlink_classifier_mb
            ) * 1024.0 * 1024.0
            mec_metrics.add_round(
                round_num=round_num,
                client_results=[
                    {
                        # NOTE (D017): the enclosing `for c in mec_round_info`
                        # loop leaks `cid` as the LAST client's id into this
                        # comprehension scope (comprehensions share the enclosing
                        # function scope). Using bare `cid` here made EVERY
                        # client's `arrived` flag equal the last client's, so
                        # on_time_rate collapsed to 0.00 even when clients arrived
                        # -> the auditor wrongly invalidated runs and the stale
                        # on_time signal masked a converged fine-tune. Always read
                        # the per-element id explicitly.
                        "arrived": (c["cid"] in client_updates),
                        "dropped": dropped[c["cid"]],
                        "t_cmp_s": c["t_cmp_model"],
                        "t_up_s": c.get("attempted_t_up_s", 0.0),
                        "e_cmp_j": mec_profiles.device(c["cid"]).kappa * c["t_cmp_model"] * mec_profiles.device(c["cid"]).freq_hz ** 2
                                   + mec_profiles.device(c["cid"]).idle_power_w * c["t_cmp_model"],
                        "e_tx_j": c.get("attempted_tx_energy_j", 0.0),
                        "payload_bits": c["payload_bits"],
                        "transmitted_bits": c.get("transmitted_bits", 0.0),
                    }
                    for c in mec_round_info
                ],
                wall_clock_s=time.time() - round_start_time,
                download_bytes=download_bytes,
                lpc_events=sum(len(mec_lpc_mask.get(c["cid"], set())) for c in mec_round_info),
                solver_ms=solver_ms_round,
            )
            # Keep adaptive A/B evidence in the machine-readable round record,
            # not only in run.log.  Counts/bytes are scheduled uplink payloads
            # (before dropout/deadline gating), while ``arrived_clients`` is the
            # number of updates that actually enter aggregation.
            mec_round_summary = {
                "matrix_mode": fedharp_matrix_mode,
                "adaptive_matrix_threshold": (
                    float(args.adaptive_matrix_threshold)
                    if fedharp_matrix_mode == "adaptive"
                    else None
                ),
                "scheduled_clients": int(len(mec_round_info)),
                "selected_clients": [int(client_id) for client_id in selected_clients],
                "arrived_clients": int(sum(
                    1 for item in mec_round_info if item["cid"] in client_updates
                )),
                "upload_A_only_blocks": int(sum(
                    item["upload_A_only_blocks"] for item in mec_round_info
                )),
                "upload_B_only_blocks": int(sum(
                    item["upload_B_only_blocks"] for item in mec_round_info
                )),
                "upload_AB_blocks": int(sum(
                    item["upload_AB_blocks"] for item in mec_round_info
                )),
                "scheduled_upload_A_bytes": float(sum(
                    item["upload_A_bytes"] for item in mec_round_info
                )),
                "scheduled_upload_B_bytes": float(sum(
                    item["upload_B_bytes"] for item in mec_round_info
                )),
                "deadline_s": float(deadline),
                "solver_ms": float(solver_ms_round),
                "resource_objective": (
                    float(resource_objective)
                    if math.isfinite(resource_objective) else None
                ),
                "cora_predictor_solver_ms": (
                    float(cora_rift_preallocation.plan.solver_ms)
                    if cora_rift_preallocation is not None else None
                ),
                "cora_predictor_deadline_s": (
                    float(cora_rift_preallocation.plan.deadline_s)
                    if cora_rift_preallocation is not None else None
                ),
                "cora_predictor_objective": (
                    float(cora_rift_preallocation.plan.objective)
                    if cora_rift_preallocation is not None else None
                ),
                "cora_rift_reference_rate_source": (
                    cora_rift_preallocation.rate_source
                    if cora_rift_preallocation is not None else None
                ),
                "cora_payload_bound_verified": (
                    bool(cora_rift_preallocation is not None)
                    if mec_network_policy == "convex_joint" and rift_enabled else None
                ),
                "bandwidth_hz": {
                    str(client_id): float(bandwidth[client_id])
                    for client_id in clients_k
                },
                "power_w": {
                    str(client_id): float(power_w[client_id])
                    for client_id in clients_k
                },
                "bandwidth_sum_hz": float(sum(bandwidth.values())),
                "bandwidth_min_hz": float(min(bandwidth.values())),
                "bandwidth_max_hz": float(max(bandwidth.values())),
                "reliability_mean": float(sum(
                    mec_est_success[item["cid"]] for item in mec_round_info
                ) / len(mec_round_info)),
                "rift_none_blocks": int(sum(
                    item.get("rift_none_blocks", 0) for item in mec_round_info
                )),
                "rift_budget_bytes": int(sum(
                    item.get("rift_budget_bytes", 0) for item in mec_round_info
                )),
                "rift_selected_bytes": int(sum(
                    item.get("rift_selected_bytes", 0) for item in mec_round_info
                )),
                "rift_captured_gain_ratio": float(
                    sum(item.get("rift_captured_gain", 0.0) for item in mec_round_info)
                    / max(1e-30, sum(item.get("rift_oracle_gain", 0.0) for item in mec_round_info))
                ) if rift_enabled else None,
                "rift_selection_ms": float(sum(
                    item.get("rift_selection_ms", 0.0) for item in mec_round_info
                )),
                "rift_guard_mandatory_count": int(sum(
                    item.get("rift_guard_mandatory_count", 0) for item in mec_round_info
                )),
                "rift_guard_max_age_after": int(max(
                    (item.get("rift_guard_max_age_after", 0) for item in mec_round_info),
                    default=0,
                )),
                "base_dropout_clients": int(sum(
                    bool(item.get("base_dropped", False)) for item in mec_round_info
                )),
                "duration_dropout_clients": int(sum(
                    bool(item.get("duration_dropped", False)) for item in mec_round_info
                )),
                "deadline_miss_clients": (
                    deadline_miss_count := int(sum(
                        (not bool(item.get("base_dropped", False)))
                        and (not bool(item.get("duration_dropped", False)))
                        and item["cid"] not in client_updates
                        for item in mec_round_info
                    ))
                ),
                "deadline_miss_count": deadline_miss_count,
                "compressed_upload_bytes_round": int(sum(
                    int(item.get("compressed_bits", 0)) for item in mec_round_info
                ) // 8),
                "duration_failure_probability_mean": float(sum(
                    item.get("duration_failure_probability", 0.0)
                    for item in mec_round_info
                ) / len(mec_round_info)),
                "attempted_upload_bytes_round": float(sum(
                    item.get("transmitted_bits", 0.0) for item in mec_round_info
                ) / 8.0),
                "rift_delivery_probability_estimate_mean": (
                    float(sum(
                        item.get("rift_delivery_probability_estimate", 1.0)
                        for item in mec_round_info
                    ) / len(mec_round_info)) if rift_enabled else None
                ),
                "rift_delivery_probability_corrector_mean": (
                    float(sum(
                        item.get("rift_delivery_probability_corrector", 1.0)
                        for item in mec_round_info
                    ) / len(mec_round_info)) if rift_enabled else None
                ),
                "rift_delivery_probability_calibration_mae": (
                    float(sum(abs(
                        item.get("rift_delivery_probability_estimate", 1.0)
                        - item.get("rift_delivery_probability_corrector", 1.0)
                    ) for item in mec_round_info) / len(mec_round_info))
                    if rift_enabled else None
                ),
                "rift_expected_captured_gain_ratio": (
                    float(sum(
                        item.get("rift_expected_captured_gain", 0.0)
                        for item in mec_round_info
                    ) / max(1e-30, sum(
                        item.get("rift_oracle_gain", 0.0)
                        for item in mec_round_info
                    ))) if rift_enabled else None
                ),
                "fedsonata_gain_target": (
                    effective_fedsonata_gain_target if fedsonata_enabled else None
                ),
                "fedsonata_target_met_rate": (
                    float(sum(
                        bool(item.get("fedsonata_gain_target_met", False))
                        for item in mec_round_info
                    ) / len(mec_round_info)) if (
                        fedsonata_enabled and effective_fedsonata_gain_target is not None
                    ) else None
                ),
                "fedsonata_target_attainment_ratio": (
                    float(sum(
                        item.get("rift_expected_captured_gain", 0.0)
                        for item in mec_round_info
                    ) / max(1e-30, sum(
                        item.get("fedsonata_cap_optimal_expected_gain", 0.0)
                        for item in mec_round_info
                    ))) if fedsonata_enabled else None
                ),
                "fedsonata_budget_utilization": (
                    float(sum(
                        item.get("rift_selected_bytes", 0) for item in mec_round_info
                    ) / max(1, sum(
                        item.get("rift_budget_bytes", 0) for item in mec_round_info
                    ))) if fedsonata_enabled else None
                ),
                "fedsonata_schedule": (
                    fedsonata_schedule if fedsonata_enabled else None
                ),
                "fedsonata_effective_budget_ratio": (
                    effective_rift_budget_ratio if fedsonata_enabled else None
                ),
                "fedsonata_high_fidelity_round": (
                    bool(fedsonata_round_policy.high_fidelity)
                    if fedsonata_round_policy is not None else None
                ),
                "fedsonata_trajectory_weight": (
                    float(fedsonata_round_policy.trajectory_weight)
                    if fedsonata_round_policy is not None else None
                ),
                "rift_ef_residual_before_l2": float(sum(
                    item.get("rift_ef_residual_before_l2", 0.0)
                    for item in mec_round_info
                )),
                "rift_ef_decay": rift_ef_decay if rift_ef_enabled else None,
                "rift_ef_residual_l2": (
                    float(rift_ef_state.total_l2()) if rift_ef_enabled else 0.0
                ),
                "rift_ef_residual_bytes": (
                    int(rift_ef_state.storage_bytes()) if rift_ef_enabled else 0
                ),
                "rift_ef_residual_factors": (
                    int(rift_ef_state.nonzero_factors()) if rift_ef_enabled else 0
                ),
                "rift_ef_cleared_A": int(sum(
                    item.get("rift_ef_cleared_A", 0) for item in mec_round_info
                )),
                "rift_ef_cleared_B": int(sum(
                    item.get("rift_ef_cleared_B", 0) for item in mec_round_info
                )),
            }
            mec_log(
                f"round {round_num} matrix_upload: "
                f"A_only={sum(c['upload_A_only_blocks'] for c in mec_round_info)} "
                f"B_only={sum(c['upload_B_only_blocks'] for c in mec_round_info)} "
                f"AB={sum(c['upload_AB_blocks'] for c in mec_round_info)} "
                f"A_bytes={sum(c['upload_A_bytes'] for c in mec_round_info):.0f} "
                f"B_bytes={sum(c['upload_B_bytes'] for c in mec_round_info):.0f} "
                f"reliability_mean={sum(mec_est_success[c['cid']] for c in mec_round_info) / len(mec_round_info):.4f} "
                f"base_drop={mec_round_summary['base_dropout_clients']} "
                f"duration_drop={mec_round_summary['duration_dropout_clients']} "
                f"deadline_miss={mec_round_summary['deadline_miss_clients']} "
                f"duration_p_mean={mec_round_summary['duration_failure_probability_mean']:.4f} "
                f"ef_residual_l2={mec_round_summary['rift_ef_residual_l2']:.6f} "
                f"ef_residual_bytes={mec_round_summary['rift_ef_residual_bytes']} "
                f"ef_residual_factors={mec_round_summary['rift_ef_residual_factors']} "
                f"ef_cleared_A={mec_round_summary['rift_ef_cleared_A']} "
                f"ef_cleared_B={mec_round_summary['rift_ef_cleared_B']}"
            )
            print(f"  [MEC] round {round_num}: arrived {sum(1 for c in mec_round_info if c['cid'] in client_updates)}/{len(mec_round_info)}, "
                  f"deadline={deadline:.3f}s, on_time={mec_metrics.last_on_time_rate:.2f}")

        # 步骤 4: 服务器聚合更新
        print(f"\n[步骤 4] 服务器聚合: 接收 {len(client_updates)} 个客户端的 LoRA 更新")
        if len(client_updates) == 0:
            # (D015) No client arrived this round -> scheduling failure. Do NOT
            # silently reuse the previous global model. Mark FAILED and stop.
            mec_log(f"FAILED: round {round_num} had 0 arrived clients "
                    f"(deadline={deadline:.3f}s too tight / all dropped)")
            with open(os.path.join(mec_run_output_dir, "FAILED"), "w") as _f:
                _f.write(f"round {round_num}: 0 arrived clients, deadline={deadline:.3f}s\n")
            raise RuntimeError(
                f"round {round_num}: 0 arrived clients (deadline={deadline:.3f}s) "
                f"-> scheduling failure"
            )
        server.aggregate_matrices(
            client_updates_A=client_updates,
            client_updates_B=(client_updates_B if upload_b_method else None),
            client_sample_counts=client_sample_counts,
            allocation_map=allocation_map
        )

        # Optional server-compatible classifier path.  The default D032
        # protocol remains personalized because A-only/FedHARP keeps B local;
        # when explicitly enabled, classifier states are averaged only from
        # on-time arrivals and broadcast before the next round.
        if args.classifier_policy == "server_global":
            arrived_classifier_states = [
                client_classifier_states_after[cid]
                for cid in sorted(client_updates)
                if cid in client_classifier_states_after and client_classifier_states_after[cid]
            ]
            arrived_classifier_weights = [
                float(client_sample_counts.get(cid, 1.0))
                for cid in sorted(client_updates)
                if cid in client_classifier_states_after and client_classifier_states_after[cid]
            ]
            if arrived_classifier_states:
                global_classifier_state = _weighted_classifier_state(
                    arrived_classifier_states, arrived_classifier_weights
                )
                _set_classifier_state(global_model, global_classifier_state)
                for client in clients:
                    client.set_classifier_state(global_classifier_state)
        elif method_l == "fedhello":
            # Fed-HeLLo needs a meaningful server-side head for proxy FIM.
            # Under the matched D034 personalized protocol this auxiliary head
            # is never broadcast and never used for reported evaluation.
            arrived_classifier_states = [
                client_classifier_states_after[cid]
                for cid in sorted(client_updates)
                if cid in client_classifier_states_after and client_classifier_states_after[cid]
            ]
            arrived_classifier_weights = [
                float(client_sample_counts.get(cid, 1.0))
                for cid in sorted(client_updates)
                if cid in client_classifier_states_after and client_classifier_states_after[cid]
            ]
            if arrived_classifier_states:
                fedhello_proxy_classifier = _weighted_classifier_state(
                    arrived_classifier_states, arrived_classifier_weights
                )
                _set_classifier_state(global_model, fedhello_proxy_classifier)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=fedhello_proxy_head, "
                        f"arrived_clients={len(arrived_classifier_states)}, "
                        "broadcast=False, evaluation_policy=client_local\n"
                    )

        # 步骤 5: 更新全局模型用于评估
        updated_global_A = server.get_global_A_matrices()
        for name, layer in global_lora_layers.items():
            if name in updated_global_A:
                layer.set_A(updated_global_A[name])

        if upload_b_method:
            updated_global_B = server.get_global_B_matrices()
            for name, layer in global_lora_layers.items():
                if name in updated_global_B:
                    layer.set_B(updated_global_B[name])
        
        # 步骤 5.5: 记录通信后的全局模型准确率 (已禁用，仅关注个性化)
        # print(f"\n正在评估通信后的全局模型准确率...")
        # global_model.eval()
        # comm_accuracy_results = evaluate_global_model(
        #     model=global_model,
        #     test_loader=global_test_loader,
        #     device=device
        # )
        # comm_accuracy = comm_accuracy_results['accuracy']
        # comm_loss = comm_accuracy_results['loss']
        # print(f"  [Round {round_num}] 通信后全局模型准确率: {comm_accuracy:.2f}%")
        # print(f"  [Round {round_num}] 通信后全局模型 Loss: {comm_loss:.4f}")

        # 追加写入通信后准确率到日志文件
        # try:
        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write(
        #             f"round={round_num}, type=post_communication, "
        #             f"accuracy={comm_accuracy:.4f}, loss={comm_loss:.6f}\n"
        #         )
        # except Exception as e:
        #     print(f"[警告] 无法写入日志文件 {log_path}: {e}")
        
        # 更新通信后的最佳准确率
        # if comm_accuracy > best_comm_accuracy:
        #     best_comm_accuracy = comm_accuracy
        #     print(f"  ✓ 通信后准确率创新高!")
        
        # 记录到训练历史中
        # training_history.append({
        #     'round': round_num,
        #     'accuracy': comm_accuracy,
        #     'loss': comm_loss,
        #     'type': 'post_communication'  # 标记为通信后准确率
        # })

        # 步骤 6: validation-only selection.  The official test loader is
        # intentionally not touched until the final restored checkpoint.
        if round_num % args.eval_every == 0 or round_num == args.num_rounds:
            eval_split_name = "val" if validation_enabled else "test"
            print(
                f"\n[步骤 6] 个性化评估: 在各客户端本地 {eval_split_name} 集上测试"
            )

            global_eval_A = server.get_global_A_matrices()
            global_eval_B = server.get_global_B_matrices() if upload_b_method else None
            fedsa_methods = {"fedsalora", "fedsa-lora", "fedsa_lora"}
            flex_methods = {"flexlora", "flex-lora", "flex_lora"}

            if is_fedharp_family and fedharp_matrix_mode == "a":
                eval_adapter_policy = "global_A+local_B"
            elif is_fedharp_family and fedharp_matrix_mode == "b":
                eval_adapter_policy = "local_A+global_B"
            elif is_fedharp_family:
                eval_adapter_policy = "global_A+global_B"
            elif method_l in fedsa_methods:
                eval_adapter_policy = "global_A+local_B"
            elif fedra_a_only:
                eval_adapter_policy = "global_A+local_B"
            else:
                eval_adapter_policy = "global_aggregated_adapter"

            print(
                f"  [评估口径] adapter={eval_adapter_policy}, "
                f"classifier={args.classifier_policy}, split={eval_split_name}"
            )
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type=eval_policy, "
                        f"adapter={eval_adapter_policy}, classifier={args.classifier_policy}, "
                        f"split={eval_split_name}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入评估口径: {e}")
            
            total_weighted_correct = 0.0
            total_samples = 0
            total_weighted_loss = 0.0
            
            for cid in range(args.num_clients):
                client = clients[cid]
                personal_A = client_personal_A_states.get(cid, client.get_all_A_matrices())
                personal_B = client_personal_B_states.get(cid, client.get_all_B_matrices())

                if is_fedharp_family and fedharp_matrix_mode == "a":
                    eval_A = global_eval_A
                    eval_B = personal_B
                elif is_fedharp_family and fedharp_matrix_mode == "b":
                    eval_A = personal_A
                    eval_B = global_eval_B
                elif is_fedharp_family:
                    eval_A = global_eval_A
                    eval_B = global_eval_B
                elif method_l in fedsa_methods:
                    eval_A = global_eval_A
                    eval_B = personal_B
                elif fedra_a_only:
                    eval_A = global_eval_A
                    eval_B = personal_B
                else:
                    eval_A = global_eval_A
                    eval_B = global_eval_B

                if method_l in _ZERO_PADDED_RANK_METHODS:
                    if hetero_client_ranks is None:
                        raise RuntimeError(f"{args.method} 客户端秩尚未初始化")
                    eval_A, eval_B = client.prepare_global_matrices_for_rank(
                        eval_A,
                        eval_B,
                        target_rank=int(hetero_client_ranks[cid]),
                    )
                elif method_l in flex_methods:
                    first_layer = next(iter(client.lora_layers.values()))
                    client_rank = max(1, int(first_layer.lora_A.shape[0]))
                    if eval_B is None:
                        raise RuntimeError("FlexLoRA 全局评估缺少 B 矩阵")
                    eval_A, eval_B = _compress_lora_matrices_to_rank(
                        eval_A,
                        eval_B,
                        target_rank=client_rank,
                        alpha=float(args.lora_alpha),
                    )

                client.set_A_matrices(eval_A)
                if eval_B is not None:
                    client.set_B_matrices(eval_B)
                
                eval_loader = (
                    client.validation_loader if validation_enabled else client.test_loader
                )
                if eval_loader is None or len(eval_loader.dataset) == 0:
                    continue

                results = client.evaluate(test_loader=eval_loader)
                acc = results['accuracy']
                loss = results['loss']
                num_samples = len(eval_loader.dataset)
                
                total_weighted_correct += (acc / 100.0) * num_samples
                total_weighted_loss += loss * num_samples
                total_samples += num_samples
                
                if cid < 3:
                    print(f"  客户端 {cid}: accuracy={acc:.2f}%, loss={loss:.4f}")

            if total_samples > 0:
                global_acc = 100.0 * total_weighted_correct / total_samples
                global_loss = total_weighted_loss / total_samples
            else:
                global_acc = 0.0
                global_loss = 0.0

            print(
                f"\n  [结果] 第 {round_num} 轮个性化加权 {eval_split_name} 准确率: "
                f"{global_acc:.2f}%"
            )
            print(f"  [结果] 第 {round_num} 轮个性化加权 loss: {global_loss:.4f}")
            
            training_history.append({
                'round': round_num,
                'accuracy': global_acc,
                'loss': global_loss,
                'type': 'validation' if validation_enabled else 'personalized',
                'split': eval_split_name,
            })

            # 追加写入个性化评估准确率到日志文件
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"round={round_num}, type={'validation' if validation_enabled else 'personalized'}, "
                        f"split={eval_split_name}, "
                        f"accuracy={global_acc:.4f}, loss={global_loss:.6f}\n"
                    )
            except Exception as e:
                print(f"[警告] 无法写入日志文件 {log_path}: {e}")
            
            if global_acc > best_accuracy:
                best_accuracy = global_acc
                print("  [最佳] 刷新历史最佳个性化准确率。")

            if validation_enabled:
                stop_decision = (
                    validation_stopper.update(round_num, float(global_acc) / 100.0)
                    if validation_stopper is not None else None
                )
                improved = (
                    bool(stop_decision.improved)
                    if stop_decision is not None
                    else global_acc >= (
                        best_validation_accuracy + float(args.validation_min_delta)
                    )
                )
                if improved:
                    best_validation_accuracy = (
                        float(stop_decision.best_score) * 100.0
                        if stop_decision is not None else float(global_acc)
                    )
                    best_validation_round = (
                        int(stop_decision.best_round)
                        if stop_decision is not None else int(round_num)
                    )
                    validation_bad_rounds = (
                        int(stop_decision.bad_rounds)
                        if stop_decision is not None else 0
                    )
                    validation_snapshot = {
                        "global_A": {
                            k: v.detach().clone()
                            for k, v in server.get_global_A_matrices().items()
                        },
                        "global_B": (
                            {
                                k: v.detach().clone()
                                for k, v in server.get_global_B_matrices().items()
                            }
                            if upload_b_method else None
                        ),
                        "classifier": _get_classifier_state(global_model),
                        "personal_A": {
                            cid: {
                                k: v.detach().clone() for k, v in state.items()
                            }
                            for cid, state in client_personal_A_states.items()
                        },
                        "personal_B": {
                            cid: {
                                k: v.detach().clone() for k, v in state.items()
                            }
                            for cid, state in client_personal_B_states.items()
                        },
                        "personal_classifier": {
                            cid: {
                                k: v.detach().clone() for k, v in state.items()
                            }
                            for cid, state in client_personal_classifier_states.items()
                        },
                    }
                    print(
                        f"  [validation] new best={best_validation_accuracy:.2f}% "
                        f"at round {best_validation_round}"
                    )
                else:
                    validation_bad_rounds = (
                        int(stop_decision.bad_rounds)
                        if stop_decision is not None
                        else validation_bad_rounds + 1
                    )
                    if (
                        stop_decision is not None
                        and stop_decision.stop
                        and stop_decision.reason == "validation_plateau"
                    ):
                        stopped_early = True
                        print(
                            f"  [validation] patience reached ({args.validation_patience}); "
                            f"truncate after round {round_num}"
                        )
        
        # 保存检查点
        if args.save_checkpoints and round_num % 10 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_round_{round_num}.pt"
            )
            save_checkpoint(
                model_state=global_model.state_dict(),
                lora_layers_state={name: layer.get_A() for name, layer in global_lora_layers.items()},
                round_num=round_num,
                filepath=checkpoint_path
            )
        
        round_time = time.time() - round_start_time
        print(f"\n[耗时] 第 {round_num} 轮: {round_time:.2f} 秒")

        # ------------------------------------------------------------------
        # FedHARP-MEC: write the audit-required round_summary.
        # Accuracy is stored in [0,1] (audit REQUIRED fields). When the
        # global-compatible evaluation path is disabled we mirror the
        # personalized metric; this is flagged in PROGRESS (eval scope).
        # ------------------------------------------------------------------
        if mec_enabled and mec_metrics is not None and mec_metrics_path is not None:
            eval_this_round = (round_num % args.eval_every == 0) or (round_num == args.num_rounds)
            if eval_this_round:
                acc_p = float(global_acc) / 100.0
                acc_g = acc_p  # global-compatible eval not yet separated (see PROGRESS)
                summary = make_round_summary(
                    round_num=round_num,
                    accuracy_personalized=acc_p * 100.0,
                    accuracy_global=acc_g * 100.0,
                    mec_latency_s=mec_metrics.last_mec_latency_s,
                    wall_clock_s=float(round_time),
                    compute_energy_j=mec_metrics.total_compute_j,
                    tx_energy_j=mec_metrics.total_tx_j,
                    on_time_rate=mec_metrics.last_on_time_rate,
                    lpc_events=mec_metrics.total_lpc_events,
                    upload_bytes=mec_metrics.total_upload_bytes,
                    download_bytes=mec_metrics.total_download_bytes,
                    accuracy_in_percent=True,
                )
                if mec_round_summary is not None:
                    summary.update(mec_round_summary)
                summary["evaluation_scope"] = (
                    "server_global_validation"
                    if args.evaluation_scope == "server_global_test"
                    else "personalized_client_validation"
                )
                summary["metric_split"] = "validation" if validation_enabled else "legacy_holdout"
                summary["accuracy_global_is_proxy"] = bool(
                    args.evaluation_scope != "server_global_test"
                )
                try:
                    with open(mec_metrics_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                    mec_log(
                        f"round {round_num} summary: acc_p={acc_p:.4f} on_time={summary['on_time_rate']:.2f} "
                        f"mec_lat={summary['mec_latency_s']:.3f}s compute_E={summary['compute_energy_j']:.4f}J "
                        f"tx_E={summary['tx_energy_j']:.4f}J up={summary['upload_bytes']}B down={summary['download_bytes']}B"
                    )
                except Exception as e:
                    print(f"[警告] 无法写入 MEC metrics: {e}")
        if stopped_early:
            break
    
    completed_rounds = int(round_num)
    restored_best_validation = False
    # FS23: capture the FINAL-ROUND state before any restore so that the
    # round-20 endpoint and the validation-selected endpoint are reported
    # separately (PROTOCOL_CN: legacy restore_best_validation output must NOT
    # stand in for the round-20 result).
    final_round_snapshot = {
        "global_A": {
            k: v.detach().clone()
            for k, v in server.get_global_A_matrices().items()
        },
        "global_B": (
            {
                k: v.detach().clone()
                for k, v in server.get_global_B_matrices().items()
            }
            if upload_b_method else None
        ),
        "classifier": _get_classifier_state(global_model),
        "personal_A": {
            cid: {k: v.detach().clone() for k, v in state.items()}
            for cid, state in client_personal_A_states.items()
        },
        "personal_B": {
            cid: {k: v.detach().clone() for k, v in state.items()}
            for cid, state in client_personal_B_states.items()
        },
        "personal_classifier": {
            cid: {k: v.detach().clone() for k, v in state.items()}
            for cid, state in client_personal_classifier_states.items()
        },
    }
    _fs23_eval_phases = [("final_round", final_round_snapshot)]
    if (
        validation_enabled
        and bool(args.restore_best_validation)
        and validation_snapshot is not None
    ):
        _fs23_eval_phases.append(("validation_selected", validation_snapshot))
        restored_best_validation = True

    # FS23: reporting-only holdout evaluation, once per phase (final-round
    # model first, then the restored validation-selected model).  No value
    # produced here can alter the checkpoint or stopping decision.
    _fs23_phase_metrics = {}
    for _fs23_phase_name, _fs23_snap in _fs23_eval_phases:
        server.set_global_A_matrices(_fs23_snap["global_A"])
        if _fs23_snap.get("global_B") is not None:
            server.set_global_B_matrices(_fs23_snap["global_B"])
        client_personal_A_states = _fs23_snap["personal_A"]
        client_personal_B_states = _fs23_snap["personal_B"]
        client_personal_classifier_states = _fs23_snap["personal_classifier"]
        global_classifier_state = _fs23_snap["classifier"]
        _set_classifier_state(global_model, global_classifier_state)
        final_test_accuracy = 0.0
        final_test_loss = 0.0
        final_test_samples = 0
        final_test_dataset = "official_test" if (
            args.evaluation_scope == "server_global_test"
        ) else "personalized_client_holdout"
        final_global_A = _fs23_snap["global_A"]
        final_global_B = _fs23_snap.get("global_B")
        if args.evaluation_scope == "server_global_test":
            for name, layer in global_lora_layers.items():
                if name in final_global_A:
                    layer.set_A(final_global_A[name])
                if final_global_B is not None and name in final_global_B:
                    layer.set_B(final_global_B[name])
            _set_classifier_state(global_model, global_classifier_state)
            final_result = evaluate_global_model(
                global_model, global_test_loader, device=device
            )
            final_test_accuracy = float(final_result["accuracy"])
            final_test_loss = float(final_result["loss"])
            final_test_samples = int(len(global_test_loader.dataset))
        else:
            final_correct = 0.0
            final_loss_sum = 0.0
            fedsa_methods_final = {"fedsalora", "fedsa-lora", "fedsa_lora"}
            flex_methods_final = {"flexlora", "flex-lora", "flex_lora"}
            for cid, client in enumerate(clients):
                personal_A = client_personal_A_states.get(cid, client.get_all_A_matrices())
                personal_B = client_personal_B_states.get(cid, client.get_all_B_matrices())
                if is_fedharp_family and fedharp_matrix_mode == "a":
                    eval_A, eval_B = final_global_A, personal_B
                elif is_fedharp_family and fedharp_matrix_mode == "b":
                    eval_A, eval_B = personal_A, final_global_B
                elif is_fedharp_family:
                    eval_A, eval_B = final_global_A, final_global_B
                elif method_l in fedsa_methods_final or fedra_a_only:
                    eval_A, eval_B = final_global_A, personal_B
                else:
                    eval_A, eval_B = final_global_A, final_global_B

                if method_l in _ZERO_PADDED_RANK_METHODS:
                    if hetero_client_ranks is None:
                        raise RuntimeError(f"{args.method} client ranks are unavailable")
                    eval_A, eval_B = client.prepare_global_matrices_for_rank(
                        eval_A, eval_B, target_rank=int(hetero_client_ranks[cid])
                    )
                elif method_l in flex_methods_final:
                    first_layer = next(iter(client.lora_layers.values()))
                    client_rank = max(1, int(first_layer.lora_A.shape[0]))
                    if eval_B is None:
                        raise RuntimeError("FlexLoRA final evaluation requires B")
                    eval_A, eval_B = _compress_lora_matrices_to_rank(
                        eval_A, eval_B, client_rank, float(args.lora_alpha)
                    )
                client.set_A_matrices(eval_A)
                if eval_B is not None:
                    client.set_B_matrices(eval_B)
                if args.classifier_policy == "client_local":
                    client.set_classifier_state(
                        client_personal_classifier_states.get(cid, {})
                    )
                else:
                    client.set_classifier_state(global_classifier_state)
                if client.test_loader is None or len(client.test_loader.dataset) == 0:
                    continue
                result = client.evaluate(test_loader=client.test_loader)
                n_test = int(len(client.test_loader.dataset))
                final_correct += float(result["accuracy"]) * n_test / 100.0
                final_loss_sum += float(result["loss"]) * n_test
                final_test_samples += n_test
            if final_test_samples > 0:
                final_test_accuracy = 100.0 * final_correct / final_test_samples
                final_test_loss = final_loss_sum / final_test_samples
        _fs23_phase_metrics[_fs23_phase_name] = {
            "accuracy": float(final_test_accuracy) / 100.0,
            "loss": float(final_test_loss),
            "num_samples": int(final_test_samples),
            "test_dataset": str(final_test_dataset),
        }
    # FS23: legacy `accuracy` keeps its meaning -- the restored
    # validation-selected holdout when restore is on, else the final round.
    _fs23_default_phase = (
        "validation_selected"
        if "validation_selected" in _fs23_phase_metrics
        else "final_round"
    )
    final_test_accuracy = (
        _fs23_phase_metrics[_fs23_default_phase]["accuracy"] * 100.0
    )
    final_test_loss = _fs23_phase_metrics[_fs23_default_phase]["loss"]
    final_test_samples = _fs23_phase_metrics[_fs23_default_phase]["num_samples"]
    for _fs23_phase_name, _fs23_pm in _fs23_phase_metrics.items():
        print(
            f"[FS23] holdout[{_fs23_phase_name}] "
            f"accuracy={_fs23_pm['accuracy'] * 100.0:.2f}% "
            f"loss={_fs23_pm['loss']:.4f} samples={_fs23_pm['num_samples']}"
        )

    stopping_record = {
        "selection_split": "validation" if validation_enabled else "legacy_holdout",
        "test_used_for_selection": False,
        "completed_rounds": completed_rounds,
        "maximum_rounds": int(args.num_rounds),
        "stopped_early": bool(stopped_early),
        "best_validation_round": int(best_validation_round),
        "best_validation_accuracy": (
            float(best_validation_accuracy)
            if math.isfinite(best_validation_accuracy) else None
        ),
        "validation_patience": int(args.validation_patience),
        "validation_min_delta_pp": float(args.validation_min_delta),
        "restored_best_validation": bool(restored_best_validation),
    }
    final_metrics_record = {
        "evaluation_scope": str(args.evaluation_scope),
        "classifier_policy": str(args.classifier_policy),
        "dataset": str(args.dataset),
        "test_dataset": final_test_dataset,
        "test_used_for_selection": False,
        "selected_checkpoint_round": int(
            best_validation_round if restored_best_validation else completed_rounds
        ),
        "completed_rounds": completed_rounds,
        "accuracy": float(final_test_accuracy) / 100.0,
        "loss": float(final_test_loss),
        "num_samples": int(final_test_samples),
        "fs23_phase_holdout": _fs23_phase_metrics,
        "accuracy_final_round": (
            _fs23_phase_metrics.get("final_round", {}) or {}
        ).get("accuracy"),
        "accuracy_validation_selected": (
            _fs23_phase_metrics.get("validation_selected", {}) or {}
        ).get("accuracy"),
    }
    if mec_enabled and mec_run_output_dir is not None:
        with open(
            os.path.join(mec_run_output_dir, "STOPPING.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(stopping_record, fh, ensure_ascii=False, indent=2)
        with open(
            os.path.join(mec_run_output_dir, "final_metrics.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(final_metrics_record, fh, ensure_ascii=False, indent=2)
        if stopped_early:
            with open(
                os.path.join(mec_run_output_dir, "EARLY_STOPPED"), "w", encoding="utf-8"
            ) as fh:
                fh.write(
                    f"completed_rounds={completed_rounds} best_round={best_validation_round}\n"
                )
        # FS23: persist both endpoint checkpoints (PROTOCOL_CN delivery checks).
        try:
            _ckpt_dir = (
                args.checkpoint_dir if args.save_checkpoints else mec_run_output_dir
            )
            os.makedirs(_ckpt_dir, exist_ok=True)
            _torch = __import__("torch")
            _torch.save(
                final_round_snapshot,
                os.path.join(_ckpt_dir, "checkpoint_final_round.pt"),
            )
            if validation_snapshot is not None:
                _torch.save(
                    validation_snapshot,
                    os.path.join(_ckpt_dir, "checkpoint_validation_selected.pt"),
                )
            mec_log(
                "FS23 endpoint checkpoints saved: "
                f"final_round + validation_selected -> {_ckpt_dir}"
            )
        except Exception as _ckpt_exc:
            mec_log(f"FS23 endpoint checkpoint save failed: {_ckpt_exc}")

    # 最终总结
    print("\n" + "="*80)
    print("训练完成")
    print("="*80)
    print(f"[总结] 历史最佳选择指标: {best_accuracy:.2f}%")
    if str(getattr(args, "lpc_mode", "train")).lower() == "projection":
        print(f"[总结] LPC projection events={lpc_projection_events}")
    print(
        f"[总结] selected checkpoint test ({final_test_dataset}): "
        f"accuracy={final_test_accuracy:.2f}% loss={final_test_loss:.4f} "
        f"samples={final_test_samples}"
    )
    # print(f"历史最佳通信后准确率: {best_comm_accuracy:.2f}%")
    
    # 分别显示通信后和个性化的最终准确率
    # comm_history = [e for e in training_history if e.get('type') == 'post_communication']
    personalized_history = [e for e in training_history if e.get('type') == 'personalized']
    
    # if comm_history:
    #     print(f"最终轮次通信后准确率: {comm_history[-1]['accuracy']:.2f}%")
    if personalized_history:
        print(f"[总结] 最终轮次个性化准确率: {personalized_history[-1]['accuracy']:.2f}%")

    if (
        hasattr(server, "uses_partial_layer_allocation")
        and server.uses_partial_layer_allocation()
    ):
        server.print_underallocated_coverage_stats()
    
    if len(training_history) > 0:
        print("\n[总结] 训练历史记录片段 (每 10 轮):")
        # print("  通信后准确率:")
        # for i, entry in enumerate(comm_history):
        #     if i % 10 == 0 or i == len(comm_history) - 1:
        #         print(f"    轮次 {entry['round']:3d}: 准确率 = {entry['accuracy']:.2f}%, Loss = {entry['loss']:.4f}")
        if personalized_history:
            print("  个性化准确率:")
            for i, entry in enumerate(personalized_history):
                if i % 10 == 0 or i == len(personalized_history) - 1:
                    print(f"    轮次 {entry['round']:3d}: 准确率 = {entry['accuracy']:.2f}%, Loss = {entry['loss']:.4f}")

    # ------------------------------------------------------------------
    # FedHARP-MEC: mark the run DONE on successful completion. The atexit
    # guard above stamps FAILED on abnormal exit; this overrides it.
    # ------------------------------------------------------------------
    if mec_enabled and mec_run_output_dir is not None and mec_metrics_path is not None:
        try:
            n_rows = 0
            if os.path.exists(mec_metrics_path):
                with open(mec_metrics_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            n_rows += 1
            last_round_ok = bool(
                n_rows >= completed_rounds
                and (
                    completed_rounds >= int(args.num_rounds)
                    or bool(stopped_early)
                )
            )
            if last_round_ok:
                with open(os.path.join(mec_run_output_dir, "DONE"), "w", encoding="utf-8") as fh:
                    fh.write(
                        f"completed rounds={completed_rounds} rows={n_rows} "
                        f"stopped_early={bool(stopped_early)}\n"
                    )
                mec_log(
                    f"DONE: completed rounds={completed_rounds} rows={n_rows} "
                    f"stopped_early={bool(stopped_early)}"
                )
                print(f"[MEC] run 完成并写入 DONE: {mec_run_output_dir}")
            else:
                with open(os.path.join(mec_run_output_dir, "FAILED"), "w", encoding="utf-8") as fh:
                    fh.write(
                        f"incomplete: rows={n_rows}, completed={completed_rounds}, "
                        f"maximum={args.num_rounds}\n"
                    )
                mec_log(
                    f"FAILED: incomplete rows={n_rows}, completed={completed_rounds}, "
                    f"maximum={args.num_rounds}"
                )
                print(f"[MEC] run 未完成，写入 FAILED")
        except Exception as e:
            print(f"[警告] 无法写入 MEC DONE/FAILED: {e}")


_EMBEDDED_TASK_MODULES = {}
def _get_embedded_task_module(task_name: str):
    t = str(task_name).strip().lower()
    if t in _EMBEDDED_TASK_MODULES:
        return _EMBEDDED_TASK_MODULES[t]
    src = _EMBEDDED_TASK_SOURCES.get(t)
    if src is None:
        raise SystemExit(f"未知内嵌任务模块: {task_name}")
    m = types.ModuleType(f"embedded_{t}")
    m.__dict__["__name__"] = f"embedded_{t}"
    m.__dict__["__file__"] = f"<embedded:{t}>"
    sys.modules[m.__dict__["__name__"]] = m
    exec(compile(src, m.__dict__["__file__"], "exec"), m.__dict__)
    _EMBEDDED_TASK_MODULES[t] = m
    return m

def _run_embedded_task(task_name: str, argv=None):
    m = _get_embedded_task_module(task_name)
    fn = getattr(m, "main", None)
    if fn is None:
        raise SystemExit(f"任务 {task_name} 缺少 main(argv) 入口")
    return fn(argv)

_EMBEDDED_TASK_SOURCES = {
    'roberta': '"""\nRoBERTa + GLUE 的 Fed-HARP 联邦训练入口。\n"""\n\nimport argparse\nimport csv\nimport json\nimport os\n\nos.environ["TOKENIZERS_PARALLELISM"] = "false"\n\nimport re\nimport time\nfrom dataclasses import dataclass\nfrom typing import Dict, List, Optional, Sequence, Set, Tuple, Union\n\nimport pandas as pd\nimport numpy as np\ntry:\n    import torch\n    import torch.nn as nn\n    from torch.utils.data import DataLoader, Dataset\nexcept ModuleNotFoundError as e:\n    raise SystemExit("Missing dependency \'torch\'. Run in fa_gpu or install requirements.") from e\n\nfrom client import FedHarpClient\nfrom server import FedHarpServer\nfrom utils import create_directories, get_device, set_seed\n\n\ndef parse_args(argv: Optional[Sequence[str]] = None):\n    p = argparse.ArgumentParser(description="Fed-ANON on GLUE with RoBERTa + LoRA")\n\n    p.add_argument(\n        "--method",\n        type=str,\n        default="fedharp",\n        choices=[\n            "fedharp",\n            "fedharp_r",\n            "fedharp_g",\n            "fedharp_l",\n            "fedharp_a",\n            "fedhello",\n            "fedra",\n            "flora",\n            "fedsalora",\n            "fedsa-lora",\n            "fedsa_lora",\n            "FedSA-LoRA",\n            "ffa-lora",\n            "ffalora",\n            "ffa_lora",\n            "flexlora",\n            "flex-lora",\n            "flex_lora",\n        ],\n    )\n    p.add_argument("--model_path", type=str, default="./Roberta")\n    p.add_argument("--hf_endpoint", type=str, default=None)\n    p.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "qnli"])\n    p.add_argument("--data_path", type=str, default=None)\n    p.add_argument("--sst2_path", type=str, default=None)\n    p.add_argument("--datasets_cache_dir", type=str, default=None)\n\n    p.add_argument("--alpha", type=float, default=0.5)\n    p.add_argument("--client_splits", type=str, default="0.9,0.1")\n    p.add_argument("--num_clients", type=int, default=10)\n    p.add_argument("--data_fraction", type=float, default=1.0)\n    p.add_argument("--max_train_samples", type=int, default=None)\n    p.add_argument("--max_eval_samples", type=int, default=None)\n\n    p.add_argument("--num_rounds", type=int, default=1)\n    p.add_argument("--clients_per_round", type=int, default=None)\n    p.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"])\n    p.add_argument("--allocation_ratio", type=float, default=0.5)\n    p.add_argument("--aggregation_lr", type=float, default=1.0)\n    p.add_argument("--struct_to_data_rounds", type=int, default=20)\n\n    p.add_argument("--lora_rank", type=int, default=8)\n    p.add_argument("--lora_alpha", type=float, default=16.0)\n    p.add_argument("--lora_dropout", type=float, default=0.0)\n    p.add_argument("--lora_targets", type=str, default="query,value")\n\n    p.add_argument("--batch_size", type=int, default=16)\n    p.add_argument("--test_batch_size", type=int, default=64)\n    p.add_argument("--num_epochs", type=int, default=1)\n    p.add_argument("--b_train_every", type=int, default=25)\n    p.add_argument("--b_num_epochs", type=int, default=1)\n    p.add_argument("--client_lr", type=float, default=2e-4)\n    p.add_argument("--max_length", type=int, default=128)\n    p.add_argument("--eval_every", type=int, default=1)\n    p.add_argument("--eval_on", type=str, default="val", choices=["val", "client_holdout"])\n\n    p.add_argument("--seed", type=int, default=42)\n    p.add_argument("--device", type=str, default="cuda")\n    p.add_argument("--save_checkpoints", action="store_true")\n    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")\n\n    return p.parse_args(args=argv)\n\n\ndef _read_sst2_tsv(path: str) -> Tuple[List[str], List[int]]:\n    texts: List[str] = []\n    labels: List[int] = []\n    with open(path, "r", encoding="utf-8") as f:\n        reader = csv.DictReader(f, delimiter="\\t")\n        for row in reader:\n            s = row.get("sentence")\n            y = row.get("label")\n            if s is None or y is None:\n                continue\n            try:\n                labels.append(int(y))\n            except Exception:\n                continue\n            texts.append(str(s))\n    return texts, labels\n\n\ndef _read_sst2_parquet(path: str) -> Tuple[List[str], List[int]]:\n    df = pd.read_parquet(path)\n    if "sentence" not in df.columns or "label" not in df.columns:\n        raise ValueError(f"Parquet file {path} must contain \'sentence\' and \'label\' columns. Found: {df.columns}")\n    \n    texts = df["sentence"].astype(str).tolist()\n    labels = df["label"].astype(int).tolist()\n    return texts, labels\n\n\nTextInput = Union[str, Tuple[str, str]]\n\ndef _read_qnli_tsv(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:\n    texts: List[Tuple[str, str]] = []\n    labels: List[int] = []\n    with open(path, "r", encoding="utf-8") as f:\n        reader = csv.DictReader(f, delimiter="\\t")\n        for row in reader:\n            q = row.get("question")\n            s = row.get("sentence")\n            y = row.get("label")\n            if q is None or s is None or y is None:\n                continue\n            y_str = str(y).strip()\n            if y_str.lower() == "entailment":\n                yy = 0\n            elif y_str.lower() == "not_entailment":\n                yy = 1\n            else:\n                try:\n                    yy = int(y_str)\n                except Exception:\n                    continue\n            texts.append((str(q), str(s)))\n            labels.append(int(yy))\n    return texts, labels\n\n\ndef _read_qnli_parquet(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:\n    df = pd.read_parquet(path)\n    if "question" not in df.columns or "sentence" not in df.columns or "label" not in df.columns:\n        raise ValueError(\n            f"Parquet file {path} must contain \'question\', \'sentence\' and \'label\' columns. Found: {df.columns}"\n        )\n    texts = list(zip(df["question"].astype(str).tolist(), df["sentence"].astype(str).tolist()))\n    labels = df["label"].astype(int).tolist()\n    return texts, labels\n\n\ndef _read_qnli_jsonl(path: str) -> Tuple[List[Tuple[str, str]], List[int]]:\n    texts: List[Tuple[str, str]] = []\n    labels: List[int] = []\n    with open(path, "r", encoding="utf-8") as f:\n        for line in f:\n            s = line.strip()\n            if not s:\n                continue\n            try:\n                obj = json.loads(s)\n            except Exception:\n                continue\n\n            if "text1" in obj and "text2" in obj:\n                q = obj.get("text1")\n                sent = obj.get("text2")\n            else:\n                q = obj.get("question")\n                sent = obj.get("sentence")\n\n            y = obj.get("label")\n            if q is None or sent is None or y is None:\n                continue\n\n            yy: Optional[int]\n            if isinstance(y, str):\n                y_str = y.strip().lower().replace(" ", "_")\n                if y_str == "entailment":\n                    yy = 0\n                elif y_str in {"not_entailment", "not_entail"}:\n                    yy = 1\n                else:\n                    try:\n                        yy = int(y_str)\n                    except Exception:\n                        yy = None\n            else:\n                try:\n                    yy = int(y)\n                except Exception:\n                    yy = None\n\n            if yy is None or int(yy) < 0:\n                continue\n\n            texts.append((str(q), str(sent)))\n            labels.append(int(yy))\n    return texts, labels\n\n\ndef _candidate_dataset_dir(*paths: Optional[str]) -> Optional[str]:\n    for p in paths:\n        if p and os.path.isdir(p):\n            return p\n    return None\n\n\ndef _load_dataset(args) -> Tuple[List[TextInput], List[int], List[TextInput], List[int]]:\n    dataset = str(args.dataset).lower()\n    candidate_dir = _candidate_dataset_dir(\n        args.data_path,\n        args.sst2_path if dataset == "sst2" else None,\n        "./data/sst-2" if dataset == "sst2" else None,\n        "./data/sst2" if dataset == "sst2" else None,\n        "./data/SST-2" if dataset == "sst2" else None,\n        "./data/qnli" if dataset == "qnli" else None,\n        "./data/QNLI" if dataset == "qnli" else None,\n        "./QNLI" if dataset == "qnli" else None,\n        "./qnli" if dataset == "qnli" else None,\n    )\n\n    if candidate_dir is not None:\n        train_parquet = os.path.join(candidate_dir, "train-00000-of-00001.parquet")\n        val_parquet = os.path.join(candidate_dir, "validation-00000-of-00001.parquet")\n        if os.path.isfile(train_parquet) and os.path.isfile(val_parquet):\n            if dataset == "sst2":\n                print(f"Loading SST-2 from Parquet: {candidate_dir}")\n                train_texts, train_labels = _read_sst2_parquet(train_parquet)\n                val_texts, val_labels = _read_sst2_parquet(val_parquet)\n                return train_texts, train_labels, val_texts, val_labels\n            if dataset == "qnli":\n                print(f"Loading QNLI from Parquet: {candidate_dir}")\n                train_texts, train_labels = _read_qnli_parquet(train_parquet)\n                val_texts, val_labels = _read_qnli_parquet(val_parquet)\n                return train_texts, train_labels, val_texts, val_labels\n\n        train_tsv = os.path.join(candidate_dir, "train.tsv")\n        dev_tsv = os.path.join(candidate_dir, "dev.tsv")\n        if not os.path.isfile(dev_tsv):\n            dev_tsv = os.path.join(candidate_dir, "validation.tsv")\n        if os.path.isfile(train_tsv) and os.path.isfile(dev_tsv):\n            if dataset == "sst2":\n                print(f"Loading SST-2 from TSV: {candidate_dir}")\n                train_texts, train_labels = _read_sst2_tsv(train_tsv)\n                val_texts, val_labels = _read_sst2_tsv(dev_tsv)\n                return train_texts, train_labels, val_texts, val_labels\n            if dataset == "qnli":\n                print(f"Loading QNLI from TSV: {candidate_dir}")\n                train_texts, train_labels = _read_qnli_tsv(train_tsv)\n                val_texts, val_labels = _read_qnli_tsv(dev_tsv)\n                return train_texts, train_labels, val_texts, val_labels\n\n        train_jsonl = os.path.join(candidate_dir, "train.jsonl")\n        dev_jsonl = os.path.join(candidate_dir, "dev.jsonl")\n        if not os.path.isfile(dev_jsonl):\n            dev_jsonl = os.path.join(candidate_dir, "validation.jsonl")\n        if dataset == "qnli" and os.path.isfile(train_jsonl) and os.path.isfile(dev_jsonl):\n            print(f"Loading QNLI from JSONL: {candidate_dir}")\n            train_texts, train_labels = _read_qnli_jsonl(train_jsonl)\n            val_texts, val_labels = _read_qnli_jsonl(dev_jsonl)\n            return train_texts, train_labels, val_texts, val_labels\n\n        if args.data_path is not None or (args.sst2_path is not None and dataset == "sst2"):\n            raise SystemExit(f"Invalid dataset path: {candidate_dir}")\n\n    try:\n        from datasets import load_dataset\n    except Exception as e:\n        raise SystemExit("Missing dependency \'datasets\'. Install with: pip install datasets") from e\n\n    try:\n        raw = load_dataset("glue", dataset, cache_dir=args.datasets_cache_dir)\n    except Exception as e:\n        raise SystemExit(\n            f"Cannot load {dataset} from the Hub. Provide --data_path, or set --hf_endpoint / proxy."\n        ) from e\n\n    train_split = raw["train"]\n    val_split = raw["validation"]\n    if dataset == "sst2":\n        train_texts = list(train_split["sentence"])\n        train_labels = [int(x) for x in train_split["label"]]\n        val_texts = list(val_split["sentence"])\n        val_labels = [int(x) for x in val_split["label"]]\n        return train_texts, train_labels, val_texts, val_labels\n    if dataset == "qnli":\n        train_texts = list(zip(train_split["question"], train_split["sentence"]))\n        train_labels = [int(x) for x in train_split["label"]]\n        val_texts = list(zip(val_split["question"], val_split["sentence"]))\n        val_labels = [int(x) for x in val_split["label"]]\n        return train_texts, train_labels, val_texts, val_labels\n    raise SystemExit(f"Unsupported dataset: {dataset}")\n\n\ndef _safe_token(s: str) -> str:\n    return str(s).replace("/", "_").replace(" ", "_")\n\n\ndef _build_log_path(args) -> str:\n    safe_method = _safe_token(args.method)\n    safe_alpha = _safe_token(args.alpha)\n    safe_dataset = _safe_token(args.dataset)\n    safe_model = _safe_token(os.path.basename(os.path.abspath(args.model_path)) or "roberta")\n    safe_heterogeneity = _safe_token(args.heterogeneity_type)\n    log_filename = f"Traininglog-{safe_method}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_heterogeneity}.txt"\n    return os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)\n\n\ndef generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:\n    if heterogeneity_type == "uniform":\n        return [base_ratio] * num_clients\n    if heterogeneity_type == "6-3-1":\n        n_low = int(0.6 * num_clients)\n        n_mid = int(0.3 * num_clients)\n        n_high = num_clients - n_low - n_mid\n        ratios: List[float] = []\n        ratios.extend([base_ratio] * n_low)\n        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid)\n        ratios.extend([1.0] * n_high)\n        np.random.shuffle(ratios)\n        return ratios\n    if heterogeneity_type == "1-1-1":\n        n_low = int(num_clients // 3)\n        n_mid = int(num_clients // 3)\n        n_high = int(num_clients) - int(n_low) - int(n_mid)\n        ratios: List[float] = []\n        ratios.extend([base_ratio] * int(n_low))\n        ratios.extend([min(1.0, base_ratio * 1.5)] * int(n_mid))\n        ratios.extend([1.0] * int(n_high))\n        np.random.shuffle(ratios)\n        return ratios\n    return [base_ratio] * num_clients\n\n\ndef _build_fedra_allocation_map(\n    *,\n    layer_names: List[str],\n    selected_clients: np.ndarray,\n    client_max_layers: Dict[int, int],\n    seed: int,\n    num_clients: int,\n) -> Dict[int, Set[str]]:\n    rng = np.random.RandomState(int(seed))\n    suffixes = (".query", ".value", ".q_proj", ".v_proj")\n    group_to_layers: Dict[str, List[str]] = {}\n    for name in layer_names:\n        group_id = name\n        for suf in suffixes:\n            if name.endswith(suf):\n                group_id = name[: -len(suf)]\n                break\n        group_to_layers.setdefault(group_id, []).append(name)\n    group_ids = sorted(group_to_layers.keys())\n\n    allocation_map: Dict[int, Set[str]] = {cid: set() for cid in range(int(num_clients))}\n    for cid in selected_clients:\n        cid_int = int(cid)\n        budget = int(client_max_layers.get(cid_int, max(1, len(layer_names))))\n        budget = max(1, min(budget, len(layer_names)))\n        target_budget = int(rng.randint(1, budget + 1))\n\n        perm = rng.permutation(group_ids)\n        chosen_groups: List[str] = []\n        used = 0\n        for gid in perm:\n            members = group_to_layers.get(str(gid), [])\n            if len(members) == 0:\n                continue\n            if used + len(members) > target_budget:\n                continue\n            chosen_groups.append(str(gid))\n            used += len(members)\n            if used >= target_budget:\n                break\n\n        if len(chosen_groups) == 0:\n            smallest = min(group_ids, key=lambda g: len(group_to_layers.get(str(g), [])))\n            chosen_groups = [str(smallest)]\n\n        allocated: Set[str] = set()\n        for gid in chosen_groups:\n            allocated.update(group_to_layers.get(str(gid), []))\n        allocation_map[cid_int] = allocated\n    return allocation_map\n\n\ndef _unfreeze_classifier_params(model: nn.Module):\n    for attr in ("classifier", "score"):\n        if hasattr(model, attr):\n            head = getattr(model, attr)\n            if isinstance(head, nn.Module):\n                for p in head.parameters():\n                    p.requires_grad = True\n\n\ndef _freeze_all_params(model: nn.Module):\n    for p in model.parameters():\n        p.requires_grad = False\n\n\ndef _extract_peft_lora_layers(model: nn.Module, adapter_name: str = "default") -> Dict[str, nn.Module]:\n    class _Adapter:\n        def __init__(self, peft_linear_module: nn.Module, adapter: str):\n            self._m = peft_linear_module\n            self._adapter = adapter\n            self.init_rank = int(self.lora_A.shape[0])\n\n        @property\n        def lora_A(self) -> torch.nn.Parameter:\n            return self._m.lora_A[self._adapter].weight\n\n        @property\n        def lora_B(self) -> torch.nn.Parameter:\n            return self._m.lora_B[self._adapter].weight\n\n        def get_A(self) -> torch.Tensor:\n            return self.lora_A.data.clone()\n\n        def set_A(self, A: torch.Tensor):\n            A = A.to(device=self.lora_A.device, dtype=self.lora_A.dtype).clone()\n            if tuple(A.shape) != tuple(self.lora_A.shape):\n                old_requires_grad = bool(self.lora_A.requires_grad)\n                in_features = int(A.shape[1])\n                out_features = int(A.shape[0])\n                new_A = nn.Linear(in_features, out_features, bias=False).to(device=A.device, dtype=A.dtype)\n                new_A.weight.data.copy_(A)\n                new_A.weight.requires_grad = old_requires_grad\n                self._m.lora_A[self._adapter] = new_A\n                self._update_peft_rank_meta(rank=out_features)\n            else:\n                self.lora_A.data = A\n\n        def get_B(self) -> torch.Tensor:\n            return self.lora_B.data.clone()\n\n        def set_B(self, B: torch.Tensor):\n            B = B.to(device=self.lora_B.device, dtype=self.lora_B.dtype).clone()\n            if tuple(B.shape) != tuple(self.lora_B.shape):\n                old_requires_grad = bool(self.lora_B.requires_grad)\n                in_features = int(B.shape[1])\n                out_features = int(B.shape[0])\n                new_B = nn.Linear(in_features, out_features, bias=False).to(device=B.device, dtype=B.dtype)\n                new_B.weight.data.copy_(B)\n                new_B.weight.requires_grad = old_requires_grad\n                self._m.lora_B[self._adapter] = new_B\n                self._update_peft_rank_meta(rank=in_features)\n            else:\n                self.lora_B.data = B\n\n        def freeze_A(self):\n            self.lora_A.requires_grad = False\n\n        def unfreeze_A(self):\n            self.lora_A.requires_grad = True\n\n        def freeze_B(self):\n            self.lora_B.requires_grad = False\n\n        def unfreeze_B(self):\n            self.lora_B.requires_grad = True\n\n        def freeze_all(self):\n            self.freeze_A()\n            self.freeze_B()\n\n        def unfreeze_all(self):\n            self.unfreeze_A()\n            self.unfreeze_B()\n\n        def get_trainable_params(self) -> List[torch.nn.Parameter]:\n            params: List[torch.nn.Parameter] = []\n            if self.lora_A.requires_grad:\n                params.append(self.lora_A)\n            if self.lora_B.requires_grad:\n                params.append(self.lora_B)\n            return params\n\n        def _get_base_weight(self) -> torch.nn.Parameter:\n            base_layer = getattr(self._m, "base_layer", None)\n            if base_layer is not None and hasattr(base_layer, "weight"):\n                return base_layer.weight\n            if hasattr(self._m, "weight"):\n                return getattr(self._m, "weight")\n            raise ValueError(f"Unsupported PEFT LoRA module type: {type(self._m)}")\n\n        def _get_scaling(self) -> float:\n            scaling = getattr(self._m, "scaling", None)\n            if isinstance(scaling, dict) and self._adapter in scaling:\n                try:\n                    return float(scaling[self._adapter])\n                except Exception:\n                    pass\n            lora_alpha = getattr(self._m, "lora_alpha", None)\n            if isinstance(lora_alpha, dict) and self._adapter in lora_alpha:\n                try:\n                    alpha = float(lora_alpha[self._adapter])\n                    r = int(self.lora_A.shape[0])\n                    return alpha / float(r) if r > 0 else 0.0\n                except Exception:\n                    pass\n            return 1.0\n\n        def _update_peft_rank_meta(self, rank: int):\n            rank = int(rank)\n            r_map = getattr(self._m, "r", None)\n            if isinstance(r_map, dict):\n                r_map[self._adapter] = rank\n            scaling_map = getattr(self._m, "scaling", None)\n            lora_alpha = getattr(self._m, "lora_alpha", None)\n            if isinstance(scaling_map, dict) and isinstance(lora_alpha, dict) and self._adapter in lora_alpha:\n                try:\n                    scaling_map[self._adapter] = float(lora_alpha[self._adapter]) / float(rank) if rank > 0 else 0.0\n                except Exception:\n                    pass\n\n        def reset_lora(self, rank: Optional[int] = None):\n            target_rank = int(rank if rank is not None else self.init_rank)\n            A_old = self.lora_A\n            B_old = self.lora_B\n            A_requires_grad = bool(A_old.requires_grad)\n            B_requires_grad = bool(B_old.requires_grad)\n            device = A_old.device\n            dtype = A_old.dtype\n\n            in_features = int(A_old.shape[1])\n            out_features = int(B_old.shape[0])\n\n            new_A = nn.Linear(in_features, target_rank, bias=False).to(device=device, dtype=dtype)\n            new_B = nn.Linear(target_rank, out_features, bias=False).to(device=device, dtype=dtype)\n            new_A.weight.data.normal_(mean=0.0, std=0.02)\n            new_B.weight.data.zero_()\n            new_A.weight.requires_grad = A_requires_grad\n            new_B.weight.requires_grad = B_requires_grad\n\n            self._m.lora_A[self._adapter] = new_A\n            self._m.lora_B[self._adapter] = new_B\n            self._update_peft_rank_meta(rank=target_rank)\n\n        def merge_lora_weights(self):\n            r = int(self.lora_A.shape[0])\n            if r <= 0:\n                return\n            scaling = self._get_scaling()\n            delta_w = (self.lora_B @ self.lora_A) * float(scaling)\n            base_w = self._get_base_weight()\n            with torch.no_grad():\n                base_w.data = base_w.data + delta_w.to(device=base_w.device, dtype=base_w.dtype)\n\n        def merge_lora_into_base_and_reset(self, rank: Optional[int] = None):\n            self.merge_lora_weights()\n            self.reset_lora(rank=rank)\n\n    lora_layers: Dict[str, nn.Module] = {}\n    for name, module in model.named_modules():\n        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):\n            continue\n        try:\n            _ = module.lora_A[adapter_name].weight\n            _ = module.lora_B[adapter_name].weight\n        except Exception:\n            continue\n        lora_layers[name] = _Adapter(module, adapter=adapter_name)\n    return lora_layers\n\n\ndef create_roberta_lora_model(\n    *,\n    model_path: str,\n    num_labels: int,\n    lora_rank: int,\n    lora_alpha: float,\n    lora_dropout: float,\n    target_modules: Sequence[str],\n):\n    try:\n        from transformers import AutoModelForSequenceClassification, AutoTokenizer\n        import transformers.utils.import_utils\n        import transformers.modeling_utils\n        transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None\n        transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None\n    except Exception as e:\n        raise SystemExit("Missing dependency \'transformers\'. Install requirements.") from e\n    try:\n        from peft import LoraConfig, get_peft_model\n    except Exception as e:\n        raise SystemExit("Missing dependency \'peft\'. Install requirements.") from e\n\n    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)\n    try:\n        model = AutoModelForSequenceClassification.from_pretrained(\n            model_path, \n            num_labels=int(num_labels), \n            weights_only=False,\n            low_cpu_mem_usage=False\n        )\n    except TypeError:\n        # Fallback for older transformers that don\'t support weights_only\n        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=int(num_labels))\n\n    lora_config = LoraConfig(\n        r=int(lora_rank),\n        lora_alpha=float(lora_alpha),\n        lora_dropout=float(lora_dropout),\n        target_modules=list(target_modules),\n        bias="none",\n    )\n    model = get_peft_model(model, lora_config)\n    lora_layers = _extract_peft_lora_layers(model, adapter_name="default")\n\n    _freeze_all_params(model)\n    _unfreeze_classifier_params(model)\n    for layer in lora_layers.values():\n        layer.unfreeze_all()\n\n    return model, tokenizer, lora_layers\n\n\ndef _dirichlet_partition(\n    labels: np.ndarray,\n    num_clients: int,\n    alpha: float,\n    seed: int,\n):\n    rng = np.random.RandomState(int(seed))\n    indices = np.arange(len(labels))\n    rng.shuffle(indices)\n    labels = labels[indices]\n\n    num_classes = int(labels.max()) + 1\n    client_indices: List[List[int]] = [[] for _ in range(int(num_clients))]\n    for cls in range(num_classes):\n        cls_idx = indices[labels == cls]\n        rng.shuffle(cls_idx)\n        proportions = rng.dirichlet([float(alpha)] * int(num_clients))\n        counts = (proportions * len(cls_idx)).astype(int)\n        counts[-1] = len(cls_idx) - int(counts[:-1].sum())\n        start = 0\n        for cid in range(int(num_clients)):\n            end = start + int(counts[cid])\n            client_indices[cid].extend(cls_idx[start:end].tolist())\n            start = end\n    for cid in range(int(num_clients)):\n        rng.shuffle(client_indices[cid])\n    return client_indices\n\n\n@dataclass\nclass EncodedText:\n    input_ids: torch.Tensor\n    attention_mask: torch.Tensor\n    labels: torch.Tensor\n\n\ndef _encode_texts(tokenizer, texts: Sequence[TextInput], labels: Sequence[int], max_length: int) -> EncodedText:\n    texts_list: List[TextInput] = list(texts)\n    if len(texts_list) > 0 and isinstance(texts_list[0], tuple) and len(texts_list[0]) == 2:\n        a: List[str] = []\n        b: List[str] = []\n        for t in texts_list:\n            if not isinstance(t, tuple) or len(t) != 2:\n                raise ValueError("Mixed text inputs: expected (text_a, text_b) pairs")\n            a.append(str(t[0]))\n            b.append(str(t[1]))\n        enc = tokenizer(\n            a,\n            b,\n            padding="max_length",\n            truncation=True,\n            max_length=int(max_length),\n            return_tensors="pt",\n        )\n    else:\n        enc = tokenizer(\n            [str(t) for t in texts_list],\n            padding="max_length",\n            truncation=True,\n            max_length=int(max_length),\n            return_tensors="pt",\n        )\n    y = torch.tensor(list(labels), dtype=torch.long)\n    return EncodedText(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=y)\n\n\nclass SST2ClientDataset(Dataset):\n    def __init__(self, encoded: EncodedText, indices: Sequence[int]):\n        idx = torch.tensor(list(indices), dtype=torch.long)\n        self.input_ids = encoded.input_ids.index_select(0, idx)\n        self.attention_mask = encoded.attention_mask.index_select(0, idx)\n        self.labels = encoded.labels.index_select(0, idx)\n\n    def __len__(self) -> int:\n        return int(self.labels.shape[0])\n\n    def __getitem__(self, i: int):\n        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}, self.labels[i]\n\n\nclass TextFedHarpClient(FedHarpClient):\n    def _train_epoch_collect_B_grad_norms(\n        self, warmup_mode: bool = False, include_A: bool = False\n    ) -> Tuple[float, Dict[str, float]]:\n        self.model.train()\n        total_loss = 0.0\n        num_batches = 0\n\n        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}\n        criterion = nn.CrossEntropyLoss()\n        self.optimizer = self._create_optimizer(warmup_mode=warmup_mode)\n        if self.optimizer is None:\n            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}\n\n        for _, (inputs, target) in enumerate(self.train_loader):\n            inputs = {k: v.to(self.device) for k, v in inputs.items()}\n            target = target.to(self.device)\n\n            self.optimizer.zero_grad()\n            out = self.model(**inputs)\n            logits = getattr(out, "logits", out)\n            loss = criterion(logits, target)\n            loss.backward()\n\n            for name, layer in self.lora_layers.items():\n                grad_sq = 0.0\n                grad_b = layer.lora_B.grad\n                if grad_b is not None:\n                    n_b = torch.norm(grad_b.detach(), p=2).item()\n                    grad_sq += float(n_b) * float(n_b)\n                if include_A:\n                    grad_a = layer.lora_A.grad\n                    if grad_a is not None:\n                        n_a = torch.norm(grad_a.detach(), p=2).item()\n                        grad_sq += float(n_a) * float(n_a)\n                if grad_sq <= 0.0:\n                    continue\n                n = float(grad_sq) ** 0.5\n                grad_norm_sq_sum[name] += n * n\n\n            self.optimizer.step()\n\n            total_loss += float(loss.item())\n            num_batches += 1\n\n        if num_batches == 0:\n            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}\n\n        grad_sensitivity = {name: (grad_norm_sq_sum[name] / num_batches) ** 0.5 for name in grad_norm_sq_sum.keys()}\n        return total_loss / num_batches, grad_sensitivity\n\n    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:\n        if test_loader is None:\n            if self.test_loader is None:\n                raise ValueError("未提供测试集 DataLoader，且客户端未配置本地 test_loader")\n            test_loader = self.test_loader\n\n        self.model.eval()\n        test_loss = 0.0\n        correct = 0\n        total = 0\n        criterion = nn.CrossEntropyLoss()\n\n        with torch.no_grad():\n            for inputs, target in test_loader:\n                inputs = {k: v.to(self.device) for k, v in inputs.items()}\n                target = target.to(self.device)\n                out = self.model(**inputs)\n                logits = getattr(out, "logits", out)\n                test_loss += float(criterion(logits, target).item())\n                pred = logits.argmax(dim=1)\n                correct += int(pred.eq(target).sum().item())\n                total += int(target.size(0))\n\n        accuracy = 100.0 * correct / total if total > 0 else 0.0\n        avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0\n        return {"loss": avg_loss, "accuracy": accuracy}\n\n\nclass RobertaFedHarpServer(FedHarpServer):\n    _layer_re = re.compile(r"(?:encoder\\\\.)?layer\\\\.(\\\\d+)")\n\n    def _get_layer_depth(self, layer_name: str) -> int:\n        m = self._layer_re.search(layer_name)\n        if not m:\n            return 0\n        try:\n            return int(m.group(1))\n        except Exception:\n            return 0\n\n\ndef _tensor_nbytes(t: torch.Tensor) -> int:\n    return int(t.numel() * t.element_size())\n\n\ndef _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:\n    total = 0\n    for _, v in d.items():\n        total += _tensor_nbytes(v)\n    return total\n\n\ndef _A_matrices_nbytes(A_matrices: Dict[str, torch.Tensor]) -> int:\n    return _dict_tensor_nbytes(A_matrices)\n\n\ndef _delta_A_nbytes(delta_A: Dict[str, torch.Tensor]) -> int:\n    return _dict_tensor_nbytes(delta_A)\n\ndef _delta_B_nbytes(delta_B: Dict[str, torch.Tensor]) -> int:\n    return _dict_tensor_nbytes(delta_B)\n\n\ndef _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:\n    n = 0\n    for name in allocated_layers:\n        layer = client_lora_layers.get(name)\n        if layer is None:\n            continue\n        n += int(layer.lora_A.numel())\n        n += int(layer.lora_B.numel())\n    return n\n\n\ndef _clone_model_and_layers(template_model: nn.Module, template_lora_layers: Dict[str, nn.Module]):\n    import copy\n\n    m = copy.deepcopy(template_model)\n    layers = _extract_peft_lora_layers(m, adapter_name="default")\n    for name, layer in layers.items():\n        src = template_lora_layers.get(name)\n        if src is None:\n            continue\n        layer.set_A(src.get_A())\n        layer.set_B(src.get_B())\n        layer.unfreeze_all()\n    _freeze_all_params(m)\n    _unfreeze_classifier_params(m)\n    return m, layers\n\n\ndef main(argv: Optional[Sequence[str]] = None):\n    args = parse_args(argv)\n\n    if args.hf_endpoint:\n        os.environ["HF_ENDPOINT"] = args.hf_endpoint\n\n    set_seed(args.seed)\n    np.random.seed(args.seed)\n\n    if args.device:\n        device = torch.device(args.device)\n    else:\n        device = get_device()\n\n    if device.type == "cuda" and torch.cuda.device_count() > 1:\n        dev = str(args.device)\n        if (":" not in dev) or dev.endswith(":0") or dev == "cuda":\n            print(f"检测到 {torch.cuda.device_count()} 张 GPU，可用 --device cuda:1 使用第二张 GPU")\n\n    if args.save_checkpoints:\n        create_directories(args.checkpoint_dir)\n\n    log_path = _build_log_path(args)\n    try:\n        with open(log_path, "w", encoding="utf-8") as f:\n            f.write("Training configuration (arguments):\\n")\n            for k, v in sorted(vars(args).items()):\n                f.write(f"{k} = {v}\\n")\n            f.write("\\nRound logs:\\n")\n    except Exception as e:\n        print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n    train_texts_all, train_labels_all, val_texts_all, val_labels_all = _load_dataset(args)\n\n    rng = np.random.RandomState(int(args.seed))\n    train_perm = rng.permutation(len(train_texts_all))\n    if args.max_train_samples is not None:\n        train_perm = train_perm[: int(args.max_train_samples)]\n    train_texts = [train_texts_all[int(i)] for i in train_perm.tolist()]\n    train_labels = np.asarray([int(train_labels_all[int(i)]) for i in train_perm.tolist()], dtype=np.int64)\n\n    val_perm = rng.permutation(len(val_texts_all))\n    if args.max_eval_samples is not None:\n        val_perm = val_perm[: int(args.max_eval_samples)]\n    val_texts = [val_texts_all[int(i)] for i in val_perm.tolist()]\n    val_labels = np.asarray([int(val_labels_all[int(i)]) for i in val_perm.tolist()], dtype=np.int64)\n\n    lora_targets = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]\n    global_model, tokenizer, global_lora_layers = create_roberta_lora_model(\n        model_path=args.model_path,\n        num_labels=2,\n        lora_rank=args.lora_rank,\n        lora_alpha=args.lora_alpha,\n        lora_dropout=args.lora_dropout,\n        target_modules=lora_targets,\n    )\n    global_model.to(device)\n    for layer in global_lora_layers.values():\n        layer.unfreeze_all()\n\n    encoded_train = _encode_texts(tokenizer, train_texts, train_labels.tolist(), max_length=args.max_length)\n    encoded_val = _encode_texts(tokenizer, val_texts, val_labels.tolist(), max_length=args.max_length)\n\n    client_indices = _dirichlet_partition(\n        train_labels,\n        num_clients=args.num_clients,\n        alpha=args.alpha,\n        seed=args.seed,\n    )\n    if int(len(train_labels)) < int(args.num_clients):\n        raise SystemExit(f"num_clients={args.num_clients} 大于训练样本数={len(train_labels)}，无法分配到每个客户端")\n\n    empty_clients = [cid for cid in range(int(args.num_clients)) if len(client_indices[cid]) == 0]\n    if empty_clients:\n        donors = sorted(range(int(args.num_clients)), key=lambda c: len(client_indices[c]), reverse=True)\n        for cid in empty_clients:\n            for d in donors:\n                if len(client_indices[d]) > 1:\n                    client_indices[cid].append(client_indices[d].pop())\n                    break\n\n        empty_clients = [cid for cid in range(int(args.num_clients)) if len(client_indices[cid]) == 0]\n        for cid in empty_clients:\n            donor = max(range(int(args.num_clients)), key=lambda c: len(client_indices[c]))\n            if len(client_indices[donor]) == 0:\n                break\n            client_indices[cid].append(client_indices[donor].pop())\n\n    try:\n        split_ratios = [float(x) for x in str(args.client_splits).split(",")]\n    except Exception:\n        split_ratios = [0.9, 0.1]\n    train_ratio = float(split_ratios[0]) / max(1e-12, float(sum(split_ratios)))\n\n    client_train_indices: List[List[int]] = []\n    client_test_indices: List[List[int]] = []\n    for cid in range(args.num_clients):\n        idxs = client_indices[cid]\n        if 0.0 < args.data_fraction < 1.0:\n            k = max(1, int(len(idxs) * float(args.data_fraction)))\n            idxs = idxs[:k]\n        split_point = int(len(idxs) * train_ratio)\n        if split_point <= 0:\n            split_point = 1\n        if split_point > len(idxs):\n            split_point = len(idxs)\n        client_train_indices.append(idxs[:split_point])\n        client_test_indices.append(idxs[split_point:])\n\n    client_capacities = generate_client_resources(args.num_clients, args.heterogeneity_type, args.allocation_ratio)\n    num_total_layers = len(global_lora_layers)\n\n    method = str(args.method).lower()\n    server = RobertaFedHarpServer(\n        model=global_model,\n        lora_layers=global_lora_layers,\n        num_clients=args.num_clients,\n        allocation_ratio=args.allocation_ratio,\n        aggregation_lr=args.aggregation_lr,\n        struct_to_data_rounds=args.struct_to_data_rounds,\n        seed=args.seed,\n        method=args.method,\n    )\n    if hasattr(server, "client_layer_counts"):\n        for cid, cap in enumerate(client_capacities):\n            server.client_layer_counts[cid] = max(1, int(num_total_layers * float(cap)))\n\n    clients: List[TextFedHarpClient] = []\n    client_sample_counts: Dict[int, int] = {}\n    eval_on = str(getattr(args, "eval_on", "val")).lower().strip()\n    if eval_on == "client_holdout":\n        print("Eval split: client_holdout (来自训练集内部切分，不等同于 GLUE validation)")\n    else:\n        print("Eval split: val (GLUE validation)")\n    for cid in range(args.num_clients):\n        train_ds = SST2ClientDataset(encoded_train, client_train_indices[cid])\n        if eval_on == "client_holdout":\n            test_ds = SST2ClientDataset(encoded_train, client_test_indices[cid])\n            if len(test_ds) == 0:\n                test_ds = SST2ClientDataset(encoded_val, list(range(len(encoded_val.labels))))\n        else:\n            test_ds = SST2ClientDataset(encoded_val, list(range(len(encoded_val.labels))))\n\n        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)\n        test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)\n\n        client_model, client_lora_layers = _clone_model_and_layers(global_model, global_lora_layers)\n        client_model.to(device)\n        client_sample_counts[cid] = len(train_ds)\n\n        clients.append(\n            TextFedHarpClient(\n                client_id=cid,\n                model=client_model,\n                lora_layers=client_lora_layers,\n                train_loader=train_loader,\n                test_loader=test_loader,\n                device=device,\n                lr=args.client_lr,\n                method=args.method,\n                warmup_steps=0,\n                warmup_lr=args.client_lr,\n            )\n        )\n\n    clients_per_round = args.clients_per_round or args.num_clients\n    best_accuracy = 0.0\n    layer_names = list(server.layer_names)\n    if hasattr(server, "client_layer_counts"):\n        client_max_layers = {cid: int(server.client_layer_counts.get(cid, len(layer_names))) for cid in range(args.num_clients)}\n    else:\n        client_max_layers = {cid: len(layer_names) for cid in range(args.num_clients)}\n\n    is_fedharp_family = method in {"fedharp", "fedharp_r", "fedharp_g", "fedharp_l", "fedharp_a", "fedanon"}\n    b_train_every = int(args.b_train_every) if (is_fedharp_family and method != "fedharp_l") else 0\n\n    for round_num in range(1, args.num_rounds + 1):\n        round_start_time = time.time()\n\n        selected_clients = np.random.choice(\n            args.num_clients, size=min(clients_per_round, args.num_clients), replace=False\n        )\n        needs_b = method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora"}\n        if method == "fedra":\n            server.current_round = int(round_num)\n            allocation_map = _build_fedra_allocation_map(\n                layer_names=layer_names,\n                selected_clients=selected_clients,\n                client_max_layers=client_max_layers,\n                seed=int(args.seed) + int(round_num) * 10007,\n                num_clients=args.num_clients,\n            )\n        elif method == "flora":\n            allocation_map = {cid: set(layer_names) for cid in range(int(args.num_clients))}\n        else:\n            allocation_map = server.start_round()\n\n        global_A = server.get_global_A_matrices()\n        global_B = server.get_global_B_matrices() if needs_b else None\n        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)\n        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0\n\n        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}\n        client_updates_B: Dict[int, Dict[str, torch.Tensor]] = {}\n        client_metrics_ab = []\n\n        for client_id in selected_clients:\n            cid = int(client_id)\n            client = clients[cid]\n            allocated_layers = allocation_map[cid]\n\n            client.receive_global_matrices(global_A, global_B, method=args.method)\n            if method == "flora":\n                client.flora_merge_and_reset(target_rank=args.lora_rank)\n\n            if b_train_every > 0 and is_fedharp_family and method != "fedharp_l":\n                stale_map = server.staleness.get(int(cid), {})\n                b_only_layers = {name for name, tau in stale_map.items() if int(tau) == int(b_train_every)}\n                if len(b_only_layers) > 0:\n                    new_b_sensitivity = client.train_B_only(b_only_layers, num_epochs=int(args.b_num_epochs))\n                    server.update_client_sensitivities(int(cid), new_b_sensitivity)\n\n            if torch.cuda.is_available() and device.type == "cuda":\n                torch.cuda.reset_peak_memory_stats(device)\n            t0 = time.time()\n            delta_A, delta_B, b_sensitivity = client.local_train(\n                allocated_layers=allocated_layers,\n                num_epochs=args.num_epochs,\n                method=args.method,\n            )\n            train_time_sec = time.time() - t0\n            if torch.cuda.is_available() and device.type == "cuda":\n                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)\n            else:\n                peak_mem_gb = 0.0\n\n            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)\n            uplink_B_mb = 0.0\n            if needs_b:\n                if delta_B is None:\n                    raise ValueError(f"{method} 需要上传 B")\n                uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)\n            comm_mb = downlink_A_mb + downlink_B_mb + uplink_A_mb + uplink_B_mb\n\n            train_samples = len(client.train_loader.dataset)\n            trainable_numel = _ab_trainable_param_numel(client.lora_layers, allocated_layers)\n            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12\n\n            client_metrics_ab.append(\n                {\n                    "round": round_num,\n                    "client": int(cid),\n                    "phase": "ab_train",\n                    "compute_tflops": float(compute_tflops),\n                    "peak_mem_gb": float(peak_mem_gb),\n                    "comm_mb": float(comm_mb),\n                    "train_time_sec": float(train_time_sec),\n                    "downlink_mb": float(downlink_A_mb + downlink_B_mb),\n                    "uplink_mb": float(uplink_A_mb + uplink_B_mb),\n                }\n            )\n\n            client_updates[cid] = delta_A\n            if needs_b:\n                client_updates_B[cid] = delta_B\n            server.update_client_sensitivities(cid, b_sensitivity)\n\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        "round={round}, type=client_metrics, phase={phase}, client={client}, "\n                        "compute_tflops={compute_tflops:.6f}, peak_mem_gb={peak_mem_gb:.6f}, "\n                        "comm_mb={comm_mb:.6f}, train_time_sec={train_time_sec:.6f}, "\n                        "downlink_mb={downlink_mb:.6f}, uplink_mb={uplink_mb:.6f}\\n".format(**client_metrics_ab[-1])\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n        if len(client_metrics_ab) > 0:\n            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics_ab]))\n            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics_ab]))\n            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics_ab]))\n            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        f"round={round_num}, type=round_metrics, phase=ab_train, "\n                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "\n                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\\n"\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n        if needs_b:\n            server.aggregate_matrices(\n                client_updates_A=client_updates,\n                client_updates_B=client_updates_B,\n                client_sample_counts=client_sample_counts,\n                allocation_map=allocation_map,\n            )\n        else:\n            server.aggregate_A_matrices(\n                client_updates=client_updates,\n                client_sample_counts=client_sample_counts,\n                allocation_map=allocation_map,\n            )\n\n        if round_num % args.eval_every == 0 or round_num == args.num_rounds:\n            total_weighted_correct = 0.0\n            total_weighted_loss = 0.0\n            total_samples = 0\n            eval_A = server.get_global_A_matrices()\n            eval_B = server.get_global_B_matrices() if needs_b else None\n            for cid in range(args.num_clients):\n                client = clients[cid]\n                client.receive_global_matrices(eval_A, eval_B, method=args.method)\n                if client.test_loader is None or len(client.test_loader.dataset) == 0:\n                    continue\n                results = client.evaluate()\n                acc = float(results["accuracy"])\n                loss = float(results["loss"])\n                n = len(client.test_loader.dataset)\n                total_weighted_correct += (acc / 100.0) * n\n                total_weighted_loss += loss * n\n                total_samples += n\n\n            if total_samples > 0:\n                global_acc = 100.0 * total_weighted_correct / total_samples\n                global_loss = total_weighted_loss / total_samples\n            else:\n                global_acc = 0.0\n                global_loss = 0.0\n\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(f"round={round_num}, type=personalized, accuracy={global_acc:.4f}, loss={global_loss:.6f}\\n")\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n            if global_acc > best_accuracy:\n                best_accuracy = global_acc\n\n        _ = time.time() - round_start_time\n\n    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")\n    print(f"Log saved to: {os.path.abspath(log_path)}")\n\n\nif __name__ == "__main__":\n    main()\n',
    'flora': '"""\nFLoRA (Federated LoRA) 联邦学习模拟主程序。\n使用 Stacking -> Merge -> Reset 的训练流程，不修改 main.py。\n"""\n\nimport argparse\nimport os\nimport time\nfrom typing import Dict, Set\n\nimport numpy as np\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader\n\nfrom client import FedHarpClient\nfrom dataset import CIFAR10Dataset, CIFAR100Dataset\nfrom models import create_client_model, create_vit_model\nfrom server import FedHarpServer\nfrom utils import create_directories, get_device, print_model_info, set_seed\n\n\ndef parse_args(argv=None):\n    parser = argparse.ArgumentParser(description="FLoRA: Federated LoRA (Stacking & Merging)")\n\n    parser.add_argument("--data_dir", type=str, default="./data")\n    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])\n    parser.add_argument("--alpha", type=float, default=0.5)\n    parser.add_argument("--data_fraction", type=float, default=1.0)\n\n    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")\n    parser.add_argument("--pretrained", action="store_true")\n    parser.add_argument("--pretrained_source", type=str, default="modelscope", choices=["modelscope", "local"])\n    parser.add_argument("--pretrained_path", type=str, default=None)\n    parser.add_argument("--ms_model_id", type=str, default=None)\n    parser.add_argument("--ms_revision", type=str, default=None)\n    parser.add_argument("--ms_cache_dir", type=str, default=None)\n    parser.add_argument("--ms_checkpoint_file", type=str, default=None)\n    parser.add_argument("--hf_endpoint", type=str, default=None)\n\n    parser.add_argument("--lora_rank", type=int, default=8)\n    parser.add_argument("--lora_alpha", type=float, default=16.0)\n    parser.add_argument("--lora_dropout", type=float, default=0.0)\n\n    parser.add_argument("--num_clients", type=int, default=10)\n    parser.add_argument("--num_rounds", type=int, default=1)\n    parser.add_argument("--clients_per_round", type=int, default=None)\n\n    parser.add_argument("--batch_size", type=int, default=128)\n    parser.add_argument("--client_lr", type=float, default=0.01)\n    parser.add_argument("--num_epochs", type=int, default=1)\n\n    parser.add_argument("--eval_every", type=int, default=1)\n    parser.add_argument("--test_batch_size", type=int, default=100)\n\n    parser.add_argument("--seed", type=int, default=42)\n    parser.add_argument("--device", type=str, default="cuda")\n    parser.add_argument("--save_checkpoints", action="store_true")\n    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")\n\n    return parser.parse_args(args=argv)\n\n\ndef evaluate_global_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:\n    model.eval()\n    test_loss = 0.0\n    correct = 0\n    total = 0\n    criterion = nn.CrossEntropyLoss()\n    with torch.no_grad():\n        for data, target in test_loader:\n            data, target = data.to(device), target.to(device)\n            output = model(data)\n            loss = criterion(output, target)\n            test_loss += float(loss.item())\n            pred = output.argmax(dim=1)\n            correct += int(pred.eq(target).sum().item())\n            total += int(target.size(0))\n    accuracy = 100.0 * correct / total if total > 0 else 0.0\n    avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0\n    return {"loss": avg_loss, "accuracy": accuracy}\n\n\ndef _tensor_nbytes(t: torch.Tensor) -> int:\n    return int(t.numel() * t.element_size())\n\n\ndef _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:\n    total = 0\n    for _, v in d.items():\n        total += _tensor_nbytes(v)\n    return total\n\n\ndef _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:\n    n = 0\n    for name in allocated_layers:\n        layer = client_lora_layers.get(name)\n        if layer is None:\n            continue\n        n += int(layer.lora_A.numel())\n        n += int(layer.lora_B.numel())\n    return n\n\n\ndef main(argv=None):\n    args = parse_args(argv)\n\n    if args.hf_endpoint:\n        os.environ["HF_ENDPOINT"] = args.hf_endpoint\n\n    set_seed(args.seed)\n\n    if args.device:\n        device = torch.device(args.device)\n    else:\n        device = get_device()\n\n    if args.save_checkpoints:\n        create_directories(args.checkpoint_dir)\n\n    safe_method = "flora"\n    safe_dataset = str(args.dataset).replace("/", "_").replace(" ", "_")\n    safe_model = str(args.model_name).replace("/", "_").replace(" ", "_")\n    safe_alpha = str(args.alpha).replace("/", "_").replace(" ", "_")\n    log_filename = f"Traininglog-{safe_method}-alpha{safe_alpha}-{safe_dataset}-{safe_model}.txt"\n    log_path = os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)\n    try:\n        with open(log_path, "w", encoding="utf-8") as f:\n            f.write("Training configuration (arguments):\\n")\n            for k, v in sorted(vars(args).items()):\n                f.write(f"{k} = {v}\\n")\n            f.write("\\nRound logs:\\n")\n    except Exception as e:\n        print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n    if args.dataset == "cifar100":\n        dataset = CIFAR100Dataset(\n            data_dir=args.data_dir,\n            alpha=args.alpha,\n            num_clients=args.num_clients,\n            seed=args.seed,\n        )\n        num_classes = 100\n    else:\n        dataset = CIFAR10Dataset(\n            data_dir=args.data_dir,\n            alpha=args.alpha,\n            num_clients=args.num_clients,\n            seed=args.seed,\n        )\n        num_classes = 10\n\n    global_test_loader = dataset.get_test_dataloader(batch_size=args.test_batch_size)\n    if hasattr(dataset, "train_dataset") and global_test_loader.dataset is getattr(dataset, "train_dataset"):\n        raise ValueError("global_test_loader 指向了训练集，存在数据泄露风险")\n\n    global_model, global_lora_layers = create_vit_model(\n        model_name=args.model_name,\n        num_classes=num_classes,\n        pretrained=args.pretrained,\n        pretrained_source=args.pretrained_source,\n        pretrained_checkpoint_path=args.pretrained_path,\n        modelscope_model_id=args.ms_model_id,\n        modelscope_revision=args.ms_revision,\n        modelscope_cache_dir=args.ms_cache_dir,\n        modelscope_checkpoint_file=args.ms_checkpoint_file,\n        use_peft=False,\n        lora_rank=args.lora_rank,\n        lora_alpha=args.lora_alpha,\n        lora_dropout=args.lora_dropout,\n    )\n    global_model.to(device)\n    print_model_info(global_model, global_lora_layers)\n\n    server = FedHarpServer(\n        model=global_model,\n        lora_layers=global_lora_layers,\n        num_clients=args.num_clients,\n        allocation_ratio=1.0,\n        aggregation_lr=1.0,\n        struct_to_data_rounds=1,\n        seed=args.seed,\n        method="flora",\n    )\n\n    clients = []\n    print(f"正在初始化 {args.num_clients} 个客户端...")\n    for client_id in range(args.num_clients):\n        train_loader = dataset.get_client_dataloader(\n            client_id,\n            batch_size=args.batch_size,\n            shuffle=True,\n            split="train",\n            fraction=args.data_fraction,\n        )\n        test_loader = dataset.get_client_dataloader(\n            client_id,\n            batch_size=args.test_batch_size,\n            shuffle=False,\n            split="test",\n            fraction=1.0,\n        )\n        client_model, client_lora_layers = create_client_model(global_model, global_lora_layers)\n        client_model.to(device)\n        client = FedHarpClient(\n            client_id=client_id,\n            model=client_model,\n            lora_layers=client_lora_layers,\n            train_loader=train_loader,\n            test_loader=test_loader,\n            device=device,\n            lr=args.client_lr,\n            method="flora",\n        )\n        clients.append(client)\n\n    clients_per_round = args.clients_per_round or args.num_clients\n    all_layers = set(server.layer_names)\n\n    for round_num in range(1, args.num_rounds + 1):\n        round_start_time = time.time()\n        selected_clients = np.random.choice(\n            args.num_clients, size=min(clients_per_round, args.num_clients), replace=False\n        )\n\n        global_A = server.get_global_A_matrices()\n        global_B = server.get_global_B_matrices()\n\n        downlink_A_mb = _dict_tensor_nbytes(global_A) / (1024.0 * 1024.0)\n        downlink_B_mb = _dict_tensor_nbytes(global_B) / (1024.0 * 1024.0)\n\n        client_updates_A = {}\n        client_updates_B = {}\n        client_metrics = []\n\n        for client_id in selected_clients:\n            client = clients[int(client_id)]\n            client.set_A_matrices(global_A)\n            client.set_B_matrices(global_B)\n            client.flora_merge_and_reset(target_rank=args.lora_rank)\n\n            if torch.cuda.is_available() and device.type == "cuda":\n                torch.cuda.reset_peak_memory_stats(device)\n            t0 = time.time()\n            A_i, B_i, _ = client.local_train(\n                allocated_layers=all_layers,\n                num_epochs=args.num_epochs,\n                method="flora",\n            )\n            train_time_sec = time.time() - t0\n            if torch.cuda.is_available() and device.type == "cuda":\n                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)\n            else:\n                peak_mem_gb = 0.0\n\n            client_updates_A[int(client_id)] = A_i\n            if B_i is None:\n                raise ValueError("FLoRA 训练必须上传 B 矩阵")\n            client_updates_B[int(client_id)] = B_i\n\n            uplink_A_mb = _dict_tensor_nbytes(A_i) / (1024.0 * 1024.0)\n            uplink_B_mb = _dict_tensor_nbytes(B_i) / (1024.0 * 1024.0)\n            comm_mb = downlink_A_mb + downlink_B_mb + uplink_A_mb + uplink_B_mb\n\n            train_samples = len(client.train_loader.dataset)\n            trainable_numel = _ab_trainable_param_numel(client.lora_layers, all_layers)\n            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12\n\n            client_metrics.append(\n                {\n                    "round": int(round_num),\n                    "client": int(client_id),\n                    "phase": "ab_train",\n                    "compute_tflops": float(compute_tflops),\n                    "peak_mem_gb": float(peak_mem_gb),\n                    "comm_mb": float(comm_mb),\n                    "train_time_sec": float(train_time_sec),\n                    "downlink_mb": float(downlink_A_mb + downlink_B_mb),\n                    "uplink_mb": float(uplink_A_mb + uplink_B_mb),\n                }\n            )\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        "round={round}, type=client_metrics, phase={phase}, client={client}, "\n                        "compute_tflops={compute_tflops:.6f}, peak_mem_gb={peak_mem_gb:.6f}, "\n                        "comm_mb={comm_mb:.6f}, train_time_sec={train_time_sec:.6f}, "\n                        "downlink_mb={downlink_mb:.6f}, uplink_mb={uplink_mb:.6f}\\n".format(\n                            **client_metrics[-1]\n                        )\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n        server.aggregate_flora_stacking(client_updates_A, client_updates_B)\n\n        if len(client_metrics) > 0:\n            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics]))\n            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics]))\n            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics]))\n            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics]))\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        f"round={round_num}, type=round_metrics, phase=ab_train, "\n                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "\n                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\\n"\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n        if args.eval_every > 0 and round_num % args.eval_every == 0:\n            results = evaluate_global_model(global_model, global_test_loader, device=device)\n            elapsed = time.time() - round_start_time\n            print(\n                f"[Round {round_num}] acc={results[\'accuracy\']:.2f}% loss={results[\'loss\']:.4f} time={elapsed:.2f}s"\n            )\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        f"round={round_num}, type=post_communication, "\n                        f"accuracy={results[\'accuracy\']:.4f}, loss={results[\'loss\']:.6f}\\n"\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n\nif __name__ == "__main__":\n    main()\n',
    'fedra': '"""\nFedRA 对比实验入口：\n- 每轮为每个客户端随机选择“层数 + 层集合”\n- 被分配的层同时训练 LoRA A 与 LoRA B\n- 上传/聚合 LoRA A 与 LoRA B\n- 评估使用各客户端本地 test split（加权汇总）\n"""\n\nimport argparse\nimport os\nimport time\nfrom typing import Dict, List, Set\n\nimport numpy as np\ntry:\n    import torch\n    import torch.nn as nn\n    from torch.utils.data import DataLoader\nexcept ModuleNotFoundError as e:\n    raise SystemExit("Missing dependency \'torch\'. Install with: pip install -r requirements.txt") from e\n\nfrom client import FedHarpClient\nfrom dataset import CIFAR10Dataset, CIFAR100Dataset\nfrom models import create_client_model, create_vit_model\nfrom server import FedHarpServer\nfrom utils import create_directories, get_device, print_model_info, set_seed\n\n\ndef parse_args(argv=None):\n    parser = argparse.ArgumentParser(description="FedRA: Random Layer Allocation (Train A+B)")\n\n    parser.add_argument("--data_dir", type=str, default="./data")\n    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])\n    parser.add_argument("--alpha", type=float, default=0.5)\n    parser.add_argument("--client_splits", type=str, default="0.9,0.1")\n    parser.add_argument("--data_fraction", type=float, default=1.0)\n\n    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")\n    parser.add_argument("--pretrained", action="store_true")\n    parser.add_argument("--pretrained_source", type=str, default="modelscope", choices=["modelscope", "local"])\n    parser.add_argument("--pretrained_path", type=str, default=None)\n    parser.add_argument("--ms_model_id", type=str, default=None)\n    parser.add_argument("--ms_revision", type=str, default=None)\n    parser.add_argument("--ms_cache_dir", type=str, default=None)\n    parser.add_argument("--ms_checkpoint_file", type=str, default=None)\n    parser.add_argument("--hf_endpoint", type=str, default=None)\n\n    parser.add_argument("--lora_rank", type=int, default=8)\n    parser.add_argument("--lora_alpha", type=float, default=16.0)\n    parser.add_argument("--lora_dropout", type=float, default=0.0)\n\n    parser.add_argument("--num_clients", type=int, default=10)\n    parser.add_argument("--num_rounds", type=int, default=1)\n    parser.add_argument("--clients_per_round", type=int, default=None)\n    parser.add_argument("--heterogeneity_type", type=str, default="6-3-1", choices=["uniform", "6-3-1", "1-1-1"])\n    parser.add_argument("--allocation_ratio", type=float, default=0.5)\n    parser.add_argument("--aggregation_lr", type=float, default=1.0)\n\n    parser.add_argument("--batch_size", type=int, default=128)\n    parser.add_argument("--client_lr", type=float, default=0.01)\n    parser.add_argument("--num_epochs", type=int, default=1)\n\n    parser.add_argument("--eval_every", type=int, default=1)\n    parser.add_argument("--test_batch_size", type=int, default=100)\n\n    parser.add_argument("--seed", type=int, default=42)\n    parser.add_argument("--device", type=str, default="cuda")\n    parser.add_argument("--save_checkpoints", action="store_true")\n    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")\n\n    return parser.parse_args(args=argv)\n\n\ndef generate_client_resources(num_clients: int, heterogeneity_type: str, base_ratio: float) -> List[float]:\n    if heterogeneity_type == "uniform":\n        return [base_ratio] * num_clients\n    if heterogeneity_type == "6-3-1":\n        n_low = int(0.6 * num_clients)\n        n_mid = int(0.3 * num_clients)\n        n_high = num_clients - n_low - n_mid\n        ratios: List[float] = []\n        ratios.extend([base_ratio] * n_low)\n        ratios.extend([min(1.0, base_ratio * 1.5)] * n_mid)\n        ratios.extend([1.0] * n_high)\n        np.random.shuffle(ratios)\n        return ratios\n    if heterogeneity_type == "1-1-1":\n        n_low = int(num_clients // 3)\n        n_mid = int(num_clients // 3)\n        n_high = int(num_clients) - int(n_low) - int(n_mid)\n        ratios: List[float] = []\n        ratios.extend([base_ratio] * int(n_low))\n        ratios.extend([min(1.0, base_ratio * 1.5)] * int(n_mid))\n        ratios.extend([1.0] * int(n_high))\n        np.random.shuffle(ratios)\n        return ratios\n    return [base_ratio] * num_clients\n\n\ndef _tensor_nbytes(t: torch.Tensor) -> int:\n    return int(t.numel() * t.element_size())\n\n\ndef _dict_tensor_nbytes(d: Dict[str, torch.Tensor]) -> int:\n    total = 0\n    for _, v in d.items():\n        total += _tensor_nbytes(v)\n    return total\n\n\ndef _delta_A_nbytes(delta_A: Dict[str, torch.Tensor]) -> int:\n    return _dict_tensor_nbytes(delta_A)\n\ndef _delta_B_nbytes(delta_B: Dict[str, torch.Tensor]) -> int:\n    return _dict_tensor_nbytes(delta_B)\n\n\ndef _A_matrices_nbytes(A_matrices: Dict[str, torch.Tensor]) -> int:\n    return _dict_tensor_nbytes(A_matrices)\n\n\ndef _ab_trainable_param_numel(client_lora_layers: Dict[str, nn.Module], allocated_layers: Set[str]) -> int:\n    n = 0\n    for name in allocated_layers:\n        layer = client_lora_layers.get(name)\n        if layer is None:\n            continue\n        n += int(layer.lora_A.numel())\n        n += int(layer.lora_B.numel())\n    return n\n\n\ndef _build_fedra_allocation_map(\n    *,\n    layer_names: List[str],\n    selected_clients: np.ndarray,\n    client_max_layers: Dict[int, int],\n    seed: int,\n    num_clients: int,\n) -> Dict[int, Set[str]]:\n    rng = np.random.RandomState(int(seed))\n    suffixes = (".query", ".value", ".q_proj", ".v_proj")\n    group_to_layers: Dict[str, List[str]] = {}\n    for name in layer_names:\n        group_id = name\n        for suf in suffixes:\n            if name.endswith(suf):\n                group_id = name[: -len(suf)]\n                break\n        group_to_layers.setdefault(group_id, []).append(name)\n    group_ids = sorted(group_to_layers.keys())\n\n    allocation_map: Dict[int, Set[str]] = {cid: set() for cid in range(int(num_clients))}\n    for cid in selected_clients:\n        cid_int = int(cid)\n        budget = int(client_max_layers.get(cid_int, max(1, len(layer_names))))\n        budget = max(1, min(budget, len(layer_names)))\n        target_budget = int(rng.randint(1, budget + 1))\n\n        perm = rng.permutation(group_ids)\n        chosen_groups: List[str] = []\n        used = 0\n        for gid in perm:\n            members = group_to_layers.get(str(gid), [])\n            if len(members) == 0:\n                continue\n            if used + len(members) > target_budget:\n                continue\n            chosen_groups.append(str(gid))\n            used += len(members)\n            if used >= target_budget:\n                break\n\n        if len(chosen_groups) == 0:\n            smallest = min(group_ids, key=lambda g: len(group_to_layers.get(str(g), [])))\n            chosen_groups = [str(smallest)]\n\n        allocated: Set[str] = set()\n        for gid in chosen_groups:\n            allocated.update(group_to_layers.get(str(gid), []))\n        allocation_map[cid_int] = allocated\n    return allocation_map\n\n\ndef main(argv=None):\n    args = parse_args(argv)\n\n    if args.hf_endpoint:\n        os.environ["HF_ENDPOINT"] = args.hf_endpoint\n\n    set_seed(args.seed)\n\n    if args.device:\n        device = torch.device(args.device)\n    else:\n        device = get_device()\n\n    if args.save_checkpoints:\n        create_directories(args.checkpoint_dir)\n\n    safe_method = "fedra"\n    safe_dataset = str(args.dataset).replace("/", "_").replace(" ", "_")\n    safe_model = str(args.model_name).replace("/", "_").replace(" ", "_")\n    safe_alpha = str(args.alpha).replace("/", "_").replace(" ", "_")\n    safe_heterogeneity_type = str(args.heterogeneity_type).replace("/", "_").replace(" ", "_")\n    log_filename = f"Traininglog-{safe_method}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_heterogeneity_type}.txt"\n    log_path = os.path.join(args.checkpoint_dir if args.save_checkpoints else ".", log_filename)\n    try:\n        with open(log_path, "w", encoding="utf-8") as f:\n            f.write("Training configuration (arguments):\\n")\n            for k, v in sorted(vars(args).items()):\n                f.write(f"{k} = {v}\\n")\n            f.write("\\nRound logs:\\n")\n    except Exception as e:\n        print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n    try:\n        splits = [float(x) for x in args.client_splits.split(",")]\n    except Exception:\n        splits = [0.9, 0.1]\n\n    if args.dataset == "cifar100":\n        dataset = CIFAR100Dataset(\n            data_dir=args.data_dir,\n            alpha=args.alpha,\n            num_clients=args.num_clients,\n            seed=args.seed,\n            splits=splits,\n        )\n        num_classes = 100\n    else:\n        dataset = CIFAR10Dataset(\n            data_dir=args.data_dir,\n            alpha=args.alpha,\n            num_clients=args.num_clients,\n            seed=args.seed,\n            splits=splits,\n        )\n        num_classes = 10\n\n    global_test_loader = dataset.get_test_dataloader(batch_size=args.test_batch_size)\n    if hasattr(dataset, "train_dataset") and global_test_loader.dataset is getattr(dataset, "train_dataset"):\n        raise ValueError("global_test_loader 指向了训练集，存在数据泄露风险")\n\n    global_model, global_lora_layers = create_vit_model(\n        model_name=args.model_name,\n        num_classes=num_classes,\n        pretrained=args.pretrained,\n        pretrained_source=args.pretrained_source,\n        pretrained_checkpoint_path=args.pretrained_path,\n        modelscope_model_id=args.ms_model_id,\n        modelscope_revision=args.ms_revision,\n        modelscope_cache_dir=args.ms_cache_dir,\n        modelscope_checkpoint_file=args.ms_checkpoint_file,\n        lora_rank=args.lora_rank,\n        lora_alpha=args.lora_alpha,\n        lora_dropout=args.lora_dropout,\n    )\n    global_model.to(device)\n    print_model_info(global_model, global_lora_layers)\n\n    client_capacities = generate_client_resources(args.num_clients, args.heterogeneity_type, args.allocation_ratio)\n    num_total_layers = len(global_lora_layers)\n    client_max_layers: Dict[int, int] = {\n        cid: max(1, int(num_total_layers * float(cap))) for cid, cap in enumerate(client_capacities)\n    }\n\n    server = FedHarpServer(\n        model=global_model,\n        lora_layers=global_lora_layers,\n        num_clients=args.num_clients,\n        allocation_ratio=args.allocation_ratio,\n        aggregation_lr=args.aggregation_lr,\n        struct_to_data_rounds=1,\n        seed=args.seed,\n        method="fedra",\n    )\n\n    clients: List[FedHarpClient] = []\n    client_sample_counts: Dict[int, int] = {}\n\n    for client_id in range(args.num_clients):\n        train_loader = dataset.get_client_dataloader(\n            client_id,\n            batch_size=args.batch_size,\n            shuffle=True,\n            split="train",\n            fraction=args.data_fraction,\n        )\n        test_loader = dataset.get_client_dataloader(\n            client_id,\n            batch_size=args.test_batch_size,\n            shuffle=False,\n            split="test",\n            fraction=1.0,\n        )\n        if (\n            hasattr(train_loader, "dataset")\n            and hasattr(test_loader, "dataset")\n            and hasattr(train_loader.dataset, "dataset")\n            and hasattr(test_loader.dataset, "dataset")\n            and train_loader.dataset.dataset is test_loader.dataset.dataset\n            and hasattr(train_loader.dataset, "indices")\n            and hasattr(test_loader.dataset, "indices")\n        ):\n            train_ids = set(int(x) for x in train_loader.dataset.indices)\n            test_ids = set(int(x) for x in test_loader.dataset.indices)\n            overlap = train_ids.intersection(test_ids)\n            if len(overlap) > 0:\n                raise ValueError(f"Client {client_id} 的 train/test 索引有重叠，存在数据泄露风险")\n        client_sample_counts[client_id] = len(train_loader.dataset)\n\n        client_model, client_lora_layers = create_client_model(global_model, global_lora_layers)\n        client_model.to(device)\n\n        clients.append(\n            FedHarpClient(\n                client_id=client_id,\n                model=client_model,\n                lora_layers=client_lora_layers,\n                train_loader=train_loader,\n                test_loader=test_loader,\n                device=device,\n                lr=args.client_lr,\n                method="fedra",\n            )\n        )\n\n    clients_per_round = args.clients_per_round or args.num_clients\n    layer_names = list(server.layer_names)\n\n    best_accuracy = 0.0\n    for round_num in range(1, args.num_rounds + 1):\n        round_start_time = time.time()\n        server.current_round = int(round_num)\n\n        selected_clients = np.random.choice(\n            args.num_clients, size=min(clients_per_round, args.num_clients), replace=False\n        )\n\n        allocation_map = _build_fedra_allocation_map(\n            layer_names=layer_names,\n            selected_clients=selected_clients,\n            client_max_layers=client_max_layers,\n            seed=args.seed + round_num * 10007,\n            num_clients=args.num_clients,\n        )\n\n        global_A = server.get_global_A_matrices()\n        global_B = server.get_global_B_matrices()\n        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)\n        downlink_B_mb = _A_matrices_nbytes(global_B) / (1024.0 * 1024.0)\n\n        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}\n        client_updates_B: Dict[int, Dict[str, torch.Tensor]] = {}\n        client_metrics_ab = []\n\n        for client_id in selected_clients:\n            cid = int(client_id)\n            client = clients[cid]\n            allocated_layers = allocation_map[cid]\n\n            client.receive_global_matrices(global_A, global_B, method="fedra")\n\n            if torch.cuda.is_available() and device.type == "cuda":\n                torch.cuda.reset_peak_memory_stats(device)\n            t0 = time.time()\n            delta_A, delta_B, _ = client.local_train(\n                allocated_layers=allocated_layers,\n                num_epochs=args.num_epochs,\n                method="fedra",\n            )\n            train_time_sec = time.time() - t0\n            if torch.cuda.is_available() and device.type == "cuda":\n                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)\n            else:\n                peak_mem_gb = 0.0\n\n            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)\n            if delta_B is None:\n                raise ValueError("fedra 需要上传 delta_B")\n            uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)\n            comm_mb = downlink_A_mb + downlink_B_mb + uplink_A_mb + uplink_B_mb\n\n            train_samples = len(client.train_loader.dataset)\n            trainable_numel = _ab_trainable_param_numel(client.lora_layers, allocated_layers)\n            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12\n\n            client_metrics_ab.append(\n                {\n                    "round": round_num,\n                    "client": int(cid),\n                    "phase": "ab_train",\n                    "compute_tflops": float(compute_tflops),\n                    "peak_mem_gb": float(peak_mem_gb),\n                    "comm_mb": float(comm_mb),\n                    "train_time_sec": float(train_time_sec),\n                    "downlink_mb": float(downlink_A_mb + downlink_B_mb),\n                    "uplink_mb": float(uplink_A_mb + uplink_B_mb),\n                }\n            )\n            client_updates[cid] = delta_A\n            client_updates_B[cid] = delta_B\n\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        "round={round}, type=client_metrics, phase={phase}, client={client}, "\n                        "compute_tflops={compute_tflops:.6f}, peak_mem_gb={peak_mem_gb:.6f}, "\n                        "comm_mb={comm_mb:.6f}, train_time_sec={train_time_sec:.6f}, "\n                        "downlink_mb={downlink_mb:.6f}, uplink_mb={uplink_mb:.6f}\\n".format(**client_metrics_ab[-1])\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n        if len(client_metrics_ab) > 0:\n            avg_compute = float(np.mean([m["compute_tflops"] for m in client_metrics_ab]))\n            avg_mem = float(np.mean([m["peak_mem_gb"] for m in client_metrics_ab]))\n            avg_comm = float(np.mean([m["comm_mb"] for m in client_metrics_ab]))\n            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        f"round={round_num}, type=round_metrics, phase=ab_train, "\n                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "\n                        f"avg_comm_mb={avg_comm:.6f}, avg_train_time_sec={avg_time:.6f}\\n"\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n        server.aggregate_matrices(\n            client_updates_A=client_updates,\n            client_updates_B=client_updates_B,\n            client_sample_counts=client_sample_counts,\n            allocation_map=allocation_map,\n        )\n\n        if round_num % args.eval_every == 0 or round_num == args.num_rounds:\n            total_weighted_correct = 0.0\n            total_samples = 0\n            total_weighted_loss = 0.0\n            eval_A = server.get_global_A_matrices()\n            eval_B = server.get_global_B_matrices()\n\n            for cid in range(args.num_clients):\n                client = clients[cid]\n                client.receive_global_matrices(eval_A, eval_B, method="fedra")\n                if client.test_loader is None or len(client.test_loader.dataset) == 0:\n                    continue\n                results = client.evaluate()\n                acc = float(results["accuracy"])\n                loss = float(results["loss"])\n                num_samples = len(client.test_loader.dataset)\n                total_weighted_correct += (acc / 100.0) * num_samples\n                total_weighted_loss += loss * num_samples\n                total_samples += num_samples\n\n            if total_samples > 0:\n                global_acc = 100.0 * total_weighted_correct / total_samples\n                global_loss = total_weighted_loss / total_samples\n            else:\n                global_acc = 0.0\n                global_loss = 0.0\n\n            try:\n                with open(log_path, "a", encoding="utf-8") as f:\n                    f.write(\n                        f"round={round_num}, type=personalized, "\n                        f"accuracy={global_acc:.4f}, loss={global_loss:.6f}\\n"\n                    )\n            except Exception as e:\n                print(f"[警告] 无法写入日志文件 {log_path}: {e}")\n\n            if global_acc > best_accuracy:\n                best_accuracy = global_acc\n\n        if args.save_checkpoints and round_num % 10 == 0:\n            pass\n\n        _ = time.time() - round_start_time\n\n    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")\n    print(f"Log saved to: {os.path.abspath(log_path)}")\n\n\nif __name__ == "__main__":\n    main()\n',
    'qwen': 'from __future__ import annotations\n\nimport argparse\nimport datetime as _dt\nimport json\nimport logging\nimport os\nimport re\nimport sys\nimport time\nfrom dataclasses import dataclass\nfrom typing import Dict, List, Optional, Sequence, Set, Tuple\n\nimport numpy as np\ntry:\n    import torch\n    import torch.nn as nn\n    from torch.utils.data import DataLoader, Dataset\nexcept ModuleNotFoundError:\n    torch = None\n    nn = None\n    DataLoader = None\n\n    class Dataset:\n        pass\n\ndef parse_args(argv: Optional[Sequence[str]] = None):\n    p = argparse.ArgumentParser(description="Qwen3-4B + GSM8K 中文: FedHARP/FedRA 联邦微调")\n    p.add_argument("--run_mode", type=str, default="train", choices=["train", "zero_shot"])\n    p.add_argument("--method", type=str, default="fedharp", choices=["fedharp", "fed_harp", "fedra", "fed_ra", "fed_Ra", "FedRA", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora"])\n    p.add_argument("--model_path", type=str, default="./Qwen3-4B_test/Qwen/Qwen3-4B copy")\n    p.add_argument("--data_path", type=str, default="./data/gsm8k/GSM8K_zh_train.json")\n    p.add_argument("--test_data_path", type=str, default="./data/gsm8k/GSM8K_zh_test.json")\n    p.add_argument("--fewshot_prompt", type=str, default="")\n    p.add_argument("--chat_format", type=str, default="chatml", choices=["chatml", "raw"])\n    p.add_argument("--system_prompt", type=str, default="")\n    p.add_argument("--eval_ratio", type=float, default=0.1)\n    p.add_argument("--eval_batch_size", type=int, default=16)\n    p.add_argument("--eval_max_new_tokens", type=int, default=512)\n    p.add_argument("--eval_every_rounds", type=int, default=2)\n    p.add_argument("--train_variant", type=str, default="both", choices=["zh", "en", "both"])\n    p.add_argument("--test_variant", type=str, default="both", choices=["zh", "en", "both"])\n    p.add_argument("--dirichlet_alpha", type=float, default=0.5)\n\n    p.add_argument("--num_clients", type=int, default=10)\n    p.add_argument("--num_rounds", type=int, default=10)\n    p.add_argument("--clients_per_round", type=int, default=None)\n\n    p.add_argument("--allocation_ratio", type=float, default=0.5)\n    p.add_argument("--allocation_scheme", type=str, default="tiered_631", choices=["tiered_631", "uniform"])\n    p.add_argument("--aggregation_lr", type=float, default=1.0)\n    p.add_argument("--struct_to_data_rounds", type=int, default=8)\n\n    p.add_argument("--lora_rank", type=int, default=8)\n    p.add_argument("--lora_alpha", type=float, default=16.0)\n    p.add_argument("--lora_dropout", type=float, default=0.0)\n    p.add_argument("--lora_targets", type=str, default="c_attn")\n    p.add_argument("--lora_qv_only", action=argparse.BooleanOptionalAction, default=True)\n\n    p.add_argument("--batch_size", type=int, default=4)\n    p.add_argument("--grad_accum_steps", type=int, default=8)\n    p.add_argument("--num_epochs", type=int, default=1)\n    p.add_argument("--client_lr", type=float, default=5e-4)\n    p.add_argument("--max_length", type=int, default=512)\n    p.add_argument("--max_train_samples", type=int, default=None)\n    p.add_argument("--train_on_inputs", action=argparse.BooleanOptionalAction, default=False)\n\n    p.add_argument("--seed", type=int, default=42)\n    p.add_argument("--device", type=str, default="cuda")\n    p.add_argument("--gradient_checkpointing", action="store_true")\n    p.add_argument("--log_file", type=str, default=None)\n    p.add_argument("--log_dir", type=str, default=".")\n    p.add_argument("--log_interval", type=int, default=10)\n    p.add_argument("--metrics_file", type=str, default=None)\n    return p.parse_args(args=argv)\n\n\ndef _normalize_method(method: str) -> str:\n    m = str(method).strip()\n    m = m.replace("_", "").lower()\n    if m in {"fedharp", "fedanon"}:\n        return "fedharp"\n    if m in {"fedra"}:\n        return "fedra"\n    return m\n\n\ndef _setup_logging(*, log_dir: str, log_file: Optional[str]) -> Tuple[logging.Logger, str]:\n    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")\n    if log_file is None or str(log_file).strip() == "":\n        log_file = f"qwenmain_train_{ts}.log"\n\n    log_dir = str(log_dir) if log_dir is not None else "."\n    os.makedirs(log_dir, exist_ok=True)\n    log_path = os.path.abspath(os.path.join(log_dir, str(log_file)))\n\n    logger = logging.getLogger("qwenmain")\n    logger.setLevel(logging.INFO)\n    logger.propagate = False\n\n    for h in list(logger.handlers):\n        logger.removeHandler(h)\n\n    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")\n\n    fh = logging.FileHandler(log_path, encoding="utf-8")\n    fh.setLevel(logging.INFO)\n    fh.setFormatter(fmt)\n    logger.addHandler(fh)\n\n    sh = logging.StreamHandler(stream=sys.stdout)\n    sh.setLevel(logging.INFO)\n    sh.setFormatter(fmt)\n    logger.addHandler(sh)\n\n    return logger, log_path\n\n\ndef _tensor_bytes(t: torch.Tensor) -> int:\n    if not isinstance(t, torch.Tensor):\n        return 0\n    return int(t.numel()) * int(t.element_size())\n\n\ndef _dict_tensor_bytes(d: Optional[Dict[str, torch.Tensor]]) -> int:\n    if not d:\n        return 0\n    return int(sum(_tensor_bytes(v) for v in d.values() if isinstance(v, torch.Tensor)))\n\n\ndef _append_metrics_jsonl(path: str, row: Dict[str, object]):\n    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)\n    with open(path, "a", encoding="utf-8") as f:\n        f.write(json.dumps(row, ensure_ascii=False) + "\\n")\n\n\ndef _evaluate_on_samples(\n    *,\n    model: nn.Module,\n    tokenizer,\n    samples: List[_Sample],\n    chat_format: str,\n    batch_size: int,\n    max_length: int,\n    train_on_inputs: bool,\n    device: torch.device,\n) -> float:\n    if len(samples) == 0:\n        return 0.0\n    model.eval()\n    ds = _GSM8KDataset(samples)\n    loader = DataLoader(\n        ds,\n        batch_size=int(batch_size),\n        shuffle=False,\n        collate_fn=lambda b: _collate_causal_lm(\n            b,\n            tokenizer=tokenizer,\n            max_length=int(max_length),\n            device=device,\n            chat_format=chat_format,\n            train_on_inputs=bool(train_on_inputs),\n        ),\n    )\n    loss_sum = 0.0\n    n = 0\n    with torch.no_grad():\n        for batch in loader:\n            out = model(**batch)\n            loss = getattr(out, "loss", None)\n            if loss is None:\n                raise ValueError("模型输出不包含 loss")\n            loss_t = loss.detach()\n            if not torch.isfinite(loss_t).all():\n                continue\n            loss_sum += float(loss_t.item())\n            n += 1\n    return float(loss_sum / float(max(1, n)))\n\n\n_GSM8K_ANS_RE = re.compile(r"####\\s*(\\-?[0-9\\.\\,]+)")\n\n\ndef _read_fewshot_prompt(prompt_path: str) -> str:\n    p = str(prompt_path).strip() if prompt_path is not None else ""\n    if p == "":\n        return ""\n    if os.path.exists(p):\n        with open(p, "r", encoding="utf-8") as f:\n            return f.read()\n    return ""\n\n\ndef _join_fewshot_prompt(*, fewshot_prompt: str, prompt: str) -> str:\n    few = str(fewshot_prompt) if fewshot_prompt is not None else ""\n    if few.strip() == "":\n        return str(prompt)\n    few = few.rstrip()\n    p = str(prompt).lstrip()\n    return few + "\\n" + p\n\n\ndef _truncate_gsm8k_completion(text: str) -> str:\n    s = str(text) if text is not None else ""\n    if s == "":\n        return s\n    cut_tokens = ["<|endoftext|>", "\\n\\n\\n", "\\n\\n", "Question:"]\n    for tok in cut_tokens:\n        if tok in s:\n            s = s.split(tok, 1)[0]\n    return s.strip()\n\n\ndef _extract_gsm8k_gold_answer(text: str) -> Optional[str]:\n    s = str(text) if text is not None else ""\n    if s == "":\n        return None\n    m = _GSM8K_ANS_RE.search(s)\n    if m is not None:\n        cand = m.group(1).strip().replace(",", "")\n        return _canonicalize_number_str(cand)\n    nums = re.findall(r"[-+]?\\d+(?:\\.\\d+)?", s.replace(",", ""))\n    if len(nums) == 0:\n        return None\n    return _canonicalize_number_str(nums[-1])\n\n\ndef _extract_gsm8k_pred_answer(text: str) -> Optional[str]:\n    s = str(text) if text is not None else ""\n    if s == "":\n        return None\n    m = _GSM8K_ANS_RE.search(s)\n    if m is not None:\n        cand = m.group(1).strip().replace(",", "")\n        return _canonicalize_number_str(cand)\n    nums = re.findall(r"[-+]?\\d+(?:\\.\\d+)?", s.replace(",", ""))\n    if len(nums) == 0:\n        return None\n    return _canonicalize_number_str(nums[-1])\n\n\ndef _canonicalize_number_str(num: str) -> Optional[str]:\n    if num is None:\n        return None\n    s = str(num).strip()\n    if len(s) == 0:\n        return None\n    if s[0] == "+":\n        s = s[1:]\n    sign = ""\n    if s.startswith("-"):\n        sign = "-"\n        s = s[1:]\n    if len(s) == 0:\n        return None\n    if "." in s:\n        int_part, frac_part = s.split(".", 1)\n        frac_part = frac_part.rstrip("0")\n        if frac_part == "":\n            s = int_part\n        else:\n            s = int_part + "." + frac_part\n    s = s.lstrip("0") or "0"\n    if s == "0":\n        return "0"\n    return sign + s if sign else s\n\n\ndef _evaluate_gsm8k_accuracy(\n    *,\n    model: nn.Module,\n    tokenizer,\n    samples: List[_Sample],\n    chat_format: str,\n    max_length: int,\n    max_new_tokens: int,\n    device: torch.device,\n) -> Tuple[float, int, int]:\n    total = int(len(samples))\n    if total == 0:\n        return 0.0, 0, 0\n\n    model.eval()\n    correct = 0\n\n    eos_id = getattr(tokenizer, "eos_token_id", None)\n    if eos_id is None:\n        eos_id = getattr(tokenizer, "eod_id", None)\n    pad_id = getattr(tokenizer, "pad_token_id", None)\n    if pad_id is None:\n        pad_id = eos_id if eos_id is not None else 0\n\n    eos_token_id = eos_id\n    if str(chat_format).lower() == "chatml":\n        im_end_id = getattr(tokenizer, "im_end_id", None)\n        if im_end_id is not None:\n            eos_token_id = int(im_end_id)\n\n    min_prompt_tokens = 16\n    eff_max_new_tokens = int(max_new_tokens)\n    if int(max_length) > 1:\n        eff_max_new_tokens = min(eff_max_new_tokens, max(1, int(max_length) - int(min_prompt_tokens)))\n    prompt_max = int(max(1, int(max_length) - int(eff_max_new_tokens)))\n    with torch.no_grad():\n        for s in samples:\n            prompt_token_ids = getattr(s, "prompt_token_ids", None)\n            if prompt_token_ids is not None:\n                prompt_ids = list(prompt_token_ids)[-prompt_max:]\n            else:\n                prompt_ids = tokenizer(str(s.prompt), add_special_tokens=True, truncation=True, max_length=prompt_max)["input_ids"]\n\n            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)\n            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)\n\n            gen = model.generate(\n                input_ids=input_ids,\n                attention_mask=attention_mask,\n                do_sample=False,\n                num_beams=1,\n                max_new_tokens=int(eff_max_new_tokens),\n                pad_token_id=int(pad_id),\n                eos_token_id=int(eos_token_id) if eos_token_id is not None else None,\n                use_cache=False,\n            )\n            gen_ids = gen[0].tolist()\n            new_ids = gen_ids[len(prompt_ids) :]\n            pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)\n            pred_text = _truncate_gsm8k_completion(pred_text)\n\n            pred_ans = _extract_gsm8k_pred_answer(pred_text)\n            gold_only = getattr(s, "answer_only", None)\n            gold_ans = _extract_gsm8k_gold_answer(gold_only) if gold_only else _extract_gsm8k_gold_answer(s.answer)\n            if pred_ans is not None and gold_ans is not None and pred_ans == gold_ans:\n                correct += 1\n\n    acc = float(correct) / float(max(1, total))\n    return acc, int(correct), int(total)\n\n\ndef _read_gsm8k_zh_json(path: str) -> List[Tuple[str, str]]:\n    with open(path, "r", encoding="utf-8") as f:\n        obj = json.load(f)\n    if not isinstance(obj, list):\n        raise ValueError(f"gsm8k json 必须是 list: {path}")\n\n    pairs: List[Tuple[str, str]] = []\n    for row in obj:\n        if not isinstance(row, dict):\n            continue\n        q = row.get("question_zh") or row.get("question")\n        a = row.get("answer_zh") or row.get("answer")\n        if not q or not a:\n            continue\n        pairs.append((str(q), str(a)))\n    if len(pairs) == 0:\n        raise ValueError(f"未从 gsm8k 读取到样本: {path}")\n    return pairs\n\n\ndef _read_gsm8k_rows(path: str) -> List[Dict[str, object]]:\n    with open(path, "r", encoding="utf-8") as f:\n        obj = json.load(f)\n    if not isinstance(obj, list):\n        raise ValueError(f"gsm8k json 必须是 list: {path}")\n    rows: List[Dict[str, object]] = []\n    for row in obj:\n        if not isinstance(row, dict):\n            continue\n        q = row.get("question")\n        q_zh = row.get("question_zh")\n        a = row.get("answer")\n        a_zh = row.get("answer_zh")\n        a_only = row.get("answer_only")\n        if not q and not q_zh:\n            continue\n        if not a and not a_zh:\n            continue\n        rows.append(\n            {\n                "question": q,\n                "question_zh": q_zh,\n                "answer": a,\n                "answer_zh": a_zh,\n                "answer_only": a_only,\n            }\n        )\n    if len(rows) == 0:\n        raise ValueError(f"未从 gsm8k 读取到样本: {path}")\n    return rows\n\n\ndef _build_samples_from_rows(*, rows: List[Dict[str, object]], variant: str, chat_format: str) -> List[_Sample]:\n    out: List[_Sample] = []\n    v = str(variant).lower().strip()\n    for r in rows:\n        q = r.get("question")\n        q_zh = r.get("question_zh")\n        a = r.get("answer")\n        a_zh = r.get("answer_zh")\n        a_only = r.get("answer_only")\n\n        def add_one(lang: str):\n            if lang == "zh":\n                qq = q_zh or q\n                aa = a_zh or a\n            else:\n                qq = q or q_zh\n                aa = a or a_zh\n            if not qq or not aa:\n                return\n            prompt = _build_train_prompt(question=str(qq), lang=str(lang))\n            out.append(_Sample(prompt=str(prompt), answer=str(aa), answer_only=(str(a_only) if a_only else None), lang=str(lang)))\n\n        if v == "zh":\n            add_one("zh")\n        elif v == "en":\n            add_one("en")\n        else:\n            add_one("zh")\n            add_one("en")\n    if len(out) == 0:\n        raise ValueError(f"variant={variant} 未构造出任何样本")\n    return out\n\n\ndef _subsample_samples(samples: List[_Sample], ratio: float, rng: np.random.RandomState) -> List[_Sample]:\n    if samples is None or len(samples) == 0:\n        return []\n    r = float(ratio) if ratio is not None else 1.0\n    if r >= 1.0:\n        return list(samples)\n    if r <= 0.0:\n        return []\n    n = int(np.ceil(len(samples) * r))\n    n = int(max(1, min(len(samples), n)))\n    idx = np.arange(len(samples))\n    rng.shuffle(idx)\n    take = idx[:n].tolist()\n    return [samples[i] for i in take]\n\n\ndef _split_clients_iid(samples: List[_Sample], num_clients: int) -> List[List[_Sample]]:\n    per_client = int(np.ceil(len(samples) / float(max(1, num_clients))))\n    splits: List[List[_Sample]] = []\n    for cid in range(int(num_clients)):\n        start = cid * per_client\n        end = min(len(samples), (cid + 1) * per_client)\n        splits.append(list(samples[start:end]))\n    return splits\n\n\ndef _split_clients_dirichlet(\n    *,\n    samples: List[_Sample],\n    num_clients: int,\n    alpha: float,\n    seed: int,\n) -> List[List[_Sample]]:\n    rng = np.random.RandomState(int(seed))\n    idx_zh = [i for i, s in enumerate(samples) if str(getattr(s, "lang", "zh")).lower() == "zh"]\n    idx_en = [i for i, s in enumerate(samples) if str(getattr(s, "lang", "zh")).lower() != "zh"]\n\n    rng.shuffle(idx_zh)\n    rng.shuffle(idx_en)\n\n    splits: List[List[_Sample]] = [[] for _ in range(int(num_clients))]\n\n    def assign(indices: List[int]):\n        if len(indices) == 0:\n            return\n        p = rng.dirichlet([float(alpha)] * int(num_clients))\n        counts = rng.multinomial(len(indices), pvals=p)\n        off = 0\n        for cid, c in enumerate(counts.tolist()):\n            if c <= 0:\n                continue\n            take = indices[off : off + int(c)]\n            off += int(c)\n            splits[int(cid)].extend([samples[i] for i in take])\n\n    assign(idx_zh)\n    assign(idx_en)\n\n    sizes = [len(s) for s in splits]\n    for cid in range(int(num_clients)):\n        if sizes[cid] > 0:\n            continue\n        src = int(np.argmax(sizes))\n        if sizes[src] <= 1:\n            continue\n        splits[cid].append(splits[src].pop())\n        sizes[cid] += 1\n        sizes[src] -= 1\n\n    return splits\n\n\ndef _ensure_qwen_safetensors_index(model_dir: str):\n    model_dir = str(model_dir)\n    index_path = os.path.join(model_dir, "model.safetensors.index.json")\n    if os.path.isfile(index_path):\n        return\n\n    shard_files = []\n    for fn in os.listdir(model_dir):\n        low = fn.lower()\n        if low.endswith(".safetensors") and "model-" in low and "-of-" in low:\n            shard_files.append(fn)\n    shard_files.sort()\n\n    if len(shard_files) == 0:\n        return\n\n    weight_map: Dict[str, str] = {}\n    total_size = 0\n    for fn in shard_files:\n        fpath = os.path.join(model_dir, fn)\n        try:\n            fsize = int(os.path.getsize(fpath))\n        except Exception as e:\n            raise SystemExit(f"无法读取权重文件大小: {fpath}: {e}") from e\n        total_size += fsize\n\n        try:\n            import struct\n\n            with open(fpath, "rb") as f:\n                header_len = struct.unpack("<Q", f.read(8))[0]\n                header_bytes = f.read(int(header_len))\n            header_obj = json.loads(header_bytes.decode("utf-8"))\n        except Exception as e:\n            raise SystemExit(f"无法解析 safetensors header: {fpath}: {e}") from e\n\n        data_section_size = fsize - (8 + int(header_len))\n        if data_section_size <= 0:\n            raise SystemExit(f"safetensors 文件异常(无数据区): {fpath}")\n\n        bad_offsets: List[Tuple[str, int, int]] = []\n        for k, v in header_obj.items():\n            if k == "__metadata__":\n                continue\n            if not isinstance(v, dict):\n                continue\n            offs = v.get("data_offsets")\n            if not isinstance(offs, (list, tuple)) or len(offs) != 2:\n                continue\n            s, e = int(offs[0]), int(offs[1])\n            if s < 0 or e < 0 or e < s or e > data_section_size:\n                bad_offsets.append((str(k), s, e))\n                if len(bad_offsets) >= 5:\n                    break\n            weight_map[str(k)] = str(fn)\n\n        if len(bad_offsets) > 0:\n            example = "; ".join([f"{name}[{s},{e}]" for name, s, e in bad_offsets[:3]])\n            need = max(e for _, _, e in bad_offsets) + (8 + int(header_len))\n            raise SystemExit(\n                f"检测到损坏/不完整的 safetensors 分片: {fpath}. "\n                f"文件大小={fsize}，header 需要至少覆盖到 {need} 字节。例: {example}"\n            )\n\n    if len(weight_map) == 0:\n        return\n\n    payload = {"metadata": {"total_size": int(total_size)}, "weight_map": weight_map}\n    with open(index_path, "w", encoding="utf-8") as f:\n        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)\n\n\n@dataclass\nclass _Sample:\n    prompt: str\n    answer: str\n    answer_only: Optional[str] = None\n    lang: str = "zh"\n    prompt_token_ids: Optional[List[int]] = None\n\n\nclass _GSM8KDataset(Dataset):\n    def __init__(self, samples: List[_Sample]):\n        self.samples = samples\n\n    def __len__(self):\n        return len(self.samples)\n\n    def __getitem__(self, idx: int) -> _Sample:\n        return self.samples[idx]\n\n\ndef _build_train_prompt(*, question: str, lang: str) -> str:\n    q = str(question) if question is not None else ""\n    return f"Question: {q}\\nLet\'s think step by step:\\n"\n\n\ndef _strip_raw_prompt_to_question(text: str) -> str:\n    s = str(text) if text is not None else ""\n    lines = [ln.strip() for ln in s.splitlines() if str(ln).strip() != ""]\n    for ln in reversed(lines):\n        if ln.startswith("Question:"):\n            return ln[len("Question:") :].strip()\n        if ln.startswith("问题："):\n            return ln.split("问题：", 1)[-1].strip()\n        if ln.startswith("问题:"):\n            return ln.split("问题:", 1)[-1].strip()\n    return s.strip()\n\n\ndef _build_zero_shot_raw_prompt(*, question: str, lang: str) -> str:\n    q = str(question) if question is not None else ""\n    return f"Question: {q}\\nPlease reason step by step, and on the last line output only: #### <final answer>.\\nAnswer:"\n\n\ndef _list_tokenizer_files(model_dir: str) -> List[str]:\n    model_dir = str(model_dir)\n    names = [\n        "tokenizer.json",\n        "tokenizer_config.json",\n        "special_tokens_map.json",\n        "vocab.json",\n        "tokenizer.model",\n        "spiece.model",\n        "vocab.txt",\n        "merges.txt",\n    ]\n    out: List[str] = []\n    for n in names:\n        if os.path.isfile(os.path.join(model_dir, n)):\n            out.append(n)\n    return out\n\n\ndef _resolve_local_model_dir(model_path: str) -> str:\n    p = os.path.expanduser(str(model_path))\n    if os.path.isdir(p):\n        return p\n    if os.path.isfile(p):\n        parent = os.path.dirname(p)\n        if os.path.isdir(parent):\n            return parent\n\n    parent = os.path.dirname(p)\n    base = os.path.basename(p)\n    if base == "" or not os.path.isdir(parent):\n        return p\n\n    base_key = base.replace("-", "").replace("_", "").lower()\n    candidates: List[Tuple[int, str]] = []\n    try:\n        for name in os.listdir(parent):\n            full = os.path.join(parent, name)\n            if not os.path.isdir(full):\n                continue\n            if name == base:\n                candidates.append((0, full))\n                continue\n            if name.startswith(base):\n                candidates.append((1, full))\n                continue\n            name_key = name.replace("-", "").replace("_", "").lower()\n            if base_key != "" and base_key in name_key:\n                candidates.append((2, full))\n    except Exception:\n        return p\n\n    if len(candidates) == 0:\n        return p\n\n    def _is_model_dir(d: str) -> bool:\n        if not os.path.isfile(os.path.join(d, "config.json")):\n            return False\n        tok_files = _list_tokenizer_files(d)\n        return len(tok_files) > 0\n\n    candidates.sort(key=lambda x: x[0])\n    for _, cand in candidates:\n        if _is_model_dir(cand):\n            return cand\n    return candidates[0][1]\n\n\ndef _encode_text(tokenizer, text: str) -> List[int]:\n    try:\n        return list(tokenizer.encode(text, allowed_special=set()))\n    except TypeError:\n        return list(tokenizer.encode(text))\n\n\ndef _make_chatml_prompt_tokens(*, tokenizer, query: str, system: str) -> List[int]:\n    im_start_id = getattr(tokenizer, "im_start_id", None)\n    im_end_id = getattr(tokenizer, "im_end_id", None)\n    if im_start_id is None or im_end_id is None:\n        raise ValueError("chatml 需要 tokenizer.im_start_id / tokenizer.im_end_id")\n\n    nl_tokens = _encode_text(tokenizer, "\\n")\n    system_part = _encode_text(tokenizer, "system") + nl_tokens + _encode_text(tokenizer, system)\n    user_part = _encode_text(tokenizer, "user") + nl_tokens + _encode_text(tokenizer, query)\n    assistant_part = _encode_text(tokenizer, "assistant") + nl_tokens\n\n    tokens: List[int] = []\n    tokens += [int(im_start_id)] + system_part + [int(im_end_id)]\n    tokens += nl_tokens + [int(im_start_id)] + user_part + [int(im_end_id)]\n    tokens += nl_tokens + [int(im_start_id)] + assistant_part\n    return tokens\n\n\ndef _collate_causal_lm(\n    batch: List[_Sample],\n    *,\n    tokenizer,\n    max_length: int,\n    device: torch.device,\n    chat_format: str,\n    train_on_inputs: bool,\n):\n    input_ids_list: List[torch.Tensor] = []\n    attention_mask_list: List[torch.Tensor] = []\n    labels_list: List[torch.Tensor] = []\n\n    eos_id = getattr(tokenizer, "eos_token_id", None)\n    if eos_id is None:\n        eos_id = getattr(tokenizer, "eod_id", None)\n    pad_id = getattr(tokenizer, "pad_token_id", None)\n    if pad_id is None:\n        pad_id = eos_id if eos_id is not None else 0\n    im_end_id = getattr(tokenizer, "im_end_id", None)\n\n    for s in batch:\n        prompt_token_ids = getattr(s, "prompt_token_ids", None)\n        if prompt_token_ids is not None:\n            prompt_ids_full = list(prompt_token_ids)\n        else:\n            prompt_ids_full = tokenizer(s.prompt, add_special_tokens=True, truncation=False)["input_ids"]\n\n        answer_ids_full = tokenizer(s.answer, add_special_tokens=False, truncation=False)["input_ids"]\n        answer_ids = list(answer_ids_full)\n        if str(chat_format).lower() == "chatml" and im_end_id is not None:\n            if len(answer_ids) == 0 or int(answer_ids[-1]) != int(im_end_id):\n                answer_ids = list(answer_ids) + [int(im_end_id)]\n        elif eos_id is not None:\n            if len(answer_ids) == 0 or int(answer_ids[-1]) != int(eos_id):\n                answer_ids = list(answer_ids) + [int(eos_id)]\n        if len(answer_ids) == 0:\n            answer_ids = [int(pad_id)]\n\n        max_answer_len = int(max(1, int(max_length) - 1))\n        answer_ids = list(answer_ids)[:max_answer_len]\n        prompt_max = int(max(0, int(max_length) - len(answer_ids)))\n        prompt_ids = list(prompt_ids_full)[-prompt_max:] if prompt_max > 0 else []\n\n        input_ids = list(prompt_ids) + list(answer_ids)\n        if bool(train_on_inputs):\n            labels = list(input_ids)\n        else:\n            labels = [-100] * len(prompt_ids) + list(answer_ids)\n\n        input_ids_t = torch.tensor(input_ids, dtype=torch.long)\n        labels_t = torch.tensor(labels, dtype=torch.long)\n        attn_t = torch.ones_like(input_ids_t, dtype=torch.long)\n\n        input_ids_list.append(input_ids_t)\n        labels_list.append(labels_t)\n        attention_mask_list.append(attn_t)\n\n    max_len = max(int(t.shape[0]) for t in input_ids_list)\n\n    def pad_to(t: torch.Tensor, value: int):\n        if int(t.shape[0]) >= max_len:\n            return t\n        pad = torch.full((max_len - int(t.shape[0]),), int(value), dtype=t.dtype)\n        return torch.cat([t, pad], dim=0)\n\n    input_ids = torch.stack([pad_to(t, pad_id) for t in input_ids_list], dim=0).to(device)\n    attention_mask = torch.stack([pad_to(t, 0) for t in attention_mask_list], dim=0).to(device)\n    labels = torch.stack([pad_to(t, -100) for t in labels_list], dim=0).to(device)\n\n    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}\n\n\ndef _extract_peft_lora_layers(model: nn.Module, adapter_name: str = "default"):\n    from models import _extract_peft_lora_layers as _impl\n\n    return _impl(model, adapter_name=adapter_name)\n\n\ndef _setup_layer_trainability(lora_layers: Dict[str, object], allocated_layers: Set[str]):\n    for name, layer in lora_layers.items():\n        if name in allocated_layers:\n            layer.unfreeze_all()\n        else:\n            layer.freeze_all()\n\n\ndef _collect_trainable_params(model: nn.Module, lora_layers: Dict[str, object]) -> List[torch.nn.Parameter]:\n    trainable: List[torch.nn.Parameter] = []\n    seen = set()\n    for layer in lora_layers.values():\n        for p in layer.get_trainable_params():\n            pid = id(p)\n            if pid in seen:\n                continue\n            seen.add(pid)\n            trainable.append(p)\n    for p in model.parameters():\n        if not p.requires_grad:\n            continue\n        pid = id(p)\n        if pid in seen:\n            continue\n        seen.add(pid)\n        trainable.append(p)\n    return trainable\n\n\ndef _get_all_A(lora_layers: Dict[str, object]) -> Dict[str, torch.Tensor]:\n    return {name: layer.get_A() for name, layer in lora_layers.items()}\n\n\ndef _get_all_B(lora_layers: Dict[str, object]) -> Dict[str, torch.Tensor]:\n    return {name: layer.get_B() for name, layer in lora_layers.items()}\n\n\ndef _set_all_A(lora_layers: Dict[str, object], A: Dict[str, torch.Tensor]):\n    for name, t in A.items():\n        if name in lora_layers:\n            lora_layers[name].set_A(t)\n\n\ndef _set_all_B(lora_layers: Dict[str, object], B: Dict[str, torch.Tensor]):\n    for name, t in B.items():\n        if name in lora_layers:\n            lora_layers[name].set_B(t)\n\n\ndef _train_one_client_round(\n    *,\n    model: nn.Module,\n    lora_layers: Dict[str, object],\n    dataloader: DataLoader,\n    allocated_layers: Set[str],\n    method: str,\n    lr: float,\n    num_epochs: int,\n    grad_accum_steps: int,\n    max_steps: Optional[int],\n    device: torch.device,\n    client_B_state: Optional[Dict[str, torch.Tensor]],\n    client_id: Optional[int] = None,\n    round_num: Optional[int] = None,\n    logger: Optional[logging.Logger] = None,\n    log_interval: int = 10,\n) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], Dict[str, float], Optional[Dict[str, torch.Tensor]], float]:\n    method_l = str(method).lower()\n    is_ffa_lora = method_l in {"ffa-lora", "ffalora", "ffa_lora"}\n    upload_B = method_l == "fedra" or is_ffa_lora\n\n    if client_B_state is not None and not upload_B:\n        _set_all_B(lora_layers, client_B_state)\n\n    A_before = _get_all_A(lora_layers)\n    B_before = _get_all_B(lora_layers) if upload_B else None\n\n    if is_ffa_lora:\n        for name, layer in lora_layers.items():\n            if name in allocated_layers:\n                layer.freeze_A()\n                layer.unfreeze_B()\n            else:\n                layer.freeze_all()\n    else:\n        _setup_layer_trainability(lora_layers, allocated_layers)\n\n    trainable_params = _collect_trainable_params(model, lora_layers)\n    if len(trainable_params) == 0:\n        return {}, None, {name: 0.0 for name in lora_layers.keys()}, client_B_state, 0.0\n\n    opt = torch.optim.AdamW(trainable_params, lr=float(lr))\n\n    grad_norm_sq_sum: Dict[str, float] = {name: 0.0 for name in lora_layers.keys()}\n    grad_steps = 0\n\n    model.train()\n    opt.zero_grad(set_to_none=True)\n    step = 0\n    loss_sum = 0.0\n    loss_count = 0\n\n    if logger is not None:\n        try:\n            num_batches = len(dataloader)\n        except Exception:\n            num_batches = None\n        logger.info(\n            f"round={round_num} client={client_id} start epochs={int(num_epochs)} "\n            f"batches={num_batches} lr={float(lr)} grad_accum={int(grad_accum_steps)} "\n            f"allocated_layers={len(allocated_layers)} method={method_l}"\n        )\n\n    for _ in range(int(num_epochs)):\n        for batch in dataloader:\n            out = model(**batch)\n            loss = getattr(out, "loss", None)\n            if loss is None:\n                raise ValueError("模型输出不包含 loss")\n            loss_v = float(loss.detach().item())\n            loss_sum += loss_v\n            loss_count += 1\n\n            if logger is not None and (loss_count == 1 or (loss_count % int(max(1, log_interval)) == 0)):\n                avg_loss = loss_sum / float(max(1, loss_count))\n                logger.info(\n                    f"round={round_num} client={client_id} step={loss_count} loss={loss_v:.6f} avg_loss={avg_loss:.6f}"\n                )\n            (loss / float(max(1, grad_accum_steps))).backward()\n\n            for name, layer in lora_layers.items():\n                grad_b = getattr(layer, "lora_B", None)\n                grad_b = None if grad_b is None else grad_b.grad\n                if grad_b is None:\n                    continue\n                n = torch.norm(grad_b.detach(), p=2).item()\n                grad_norm_sq_sum[name] += float(n) * float(n)\n            grad_steps += 1\n\n            if (step + 1) % int(max(1, grad_accum_steps)) == 0:\n                opt.step()\n                opt.zero_grad(set_to_none=True)\n\n            step += 1\n            if max_steps is not None and int(step) >= int(max_steps):\n                break\n        if max_steps is not None and int(step) >= int(max_steps):\n            break\n\n    if logger is not None:\n        avg_loss = loss_sum / float(max(1, loss_count))\n        logger.info(f"round={round_num} client={client_id} done steps={loss_count} avg_loss={avg_loss:.6f}")\n\n    A_after = _get_all_A(lora_layers)\n    B_after = _get_all_B(lora_layers) if upload_B else None\n\n    delta_A: Dict[str, torch.Tensor] = {}\n    delta_B: Optional[Dict[str, torch.Tensor]] = None\n\n    if not is_ffa_lora:\n        for name in allocated_layers:\n            if name in A_before and name in A_after:\n                delta_A[name] = A_after[name] - A_before[name]\n\n    if upload_B and B_before is not None and B_after is not None:\n        delta_B = {}\n        for name in allocated_layers:\n            if name in B_before and name in B_after:\n                delta_B[name] = B_after[name] - B_before[name]\n\n    if grad_steps > 0:\n        b_sensitivity = {name: float((grad_norm_sq_sum[name] / float(grad_steps)) ** 0.5) for name in grad_norm_sq_sum.keys()}\n    else:\n        b_sensitivity = {name: 0.0 for name in lora_layers.keys()}\n\n    new_client_B_state = client_B_state\n    if not upload_B:\n        new_client_B_state = _get_all_B(lora_layers)\n\n    avg_loss = float(loss_sum / float(max(1, loss_count)))\n    return delta_A, delta_B, b_sensitivity, new_client_B_state, avg_loss\n\n\ndef main(argv: Optional[Sequence[str]] = None):\n    args = parse_args(argv)\n    logger, log_path = _setup_logging(log_dir=str(args.log_dir), log_file=args.log_file)\n    logger.info(f"log_file={log_path}")\n    metrics_path = os.path.abspath(os.path.join(str(args.log_dir), str(args.metrics_file))) if args.metrics_file else os.path.abspath(log_path.replace(".log", "_metrics.jsonl"))\n    logger.info(f"metrics_file={metrics_path}")\n    logger.info(f"args={vars(args)}")\n    if torch is None:\n        raise SystemExit("Missing dependency \'torch\'. Install with: pip install -r requirements.txt")\n    from utils import get_device, set_seed\n    set_seed(int(args.seed))\n    if str(args.device).lower() == "cpu":\n        device = torch.device("cpu")\n        logger.info("device=cpu")\n    else:\n        device = get_device()\n        logger.info(f"device={device}")\n\n    run_mode = str(getattr(args, "run_mode", "train")).strip().lower()\n\n    try:\n        import google.protobuf\n    except ModuleNotFoundError as e:\n        raise SystemExit("Missing dependency \'protobuf\'. Install with: pip install \'protobuf<5\'") from e\n\n    try:\n        from transformers import AutoModelForCausalLM, AutoTokenizer\n    except ModuleNotFoundError as e:\n        raise SystemExit("Missing dependency \'transformers\'. Install with: pip install -r requirements.txt") from e\n\n    model_path = _resolve_local_model_dir(str(args.model_path))\n    if model_path != str(args.model_path):\n        logger.info(f"model_path_resolved_from={args.model_path} model_path={model_path}")\n    if os.path.isdir(model_path):\n        _ensure_qwen_safetensors_index(model_path)\n    else:\n        logger.info(f"model_path_not_dir={model_path}")\n\n    rng = np.random.RandomState(int(args.seed))\n    chat_format = str(args.chat_format).lower()\n    test_rows = _read_gsm8k_rows(str(args.test_data_path))\n    rng_test = np.random.RandomState(int(args.seed) + 1)\n    rng_test.shuffle(test_rows)\n    test_samples = _build_samples_from_rows(rows=test_rows, variant=str(args.test_variant), chat_format=chat_format)\n    fewshot_prompt = _read_fewshot_prompt(str(getattr(args, "fewshot_prompt", "")))\n    if fewshot_prompt.strip() != "":\n        for s in test_samples:\n            s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=str(getattr(s, "prompt", "")))\n\n    if run_mode == "train":\n        method = _normalize_method(args.method)\n        train_rows = _read_gsm8k_rows(str(args.data_path))\n        rng.shuffle(train_rows)\n        if args.max_train_samples is not None:\n            train_rows = train_rows[: int(args.max_train_samples)]\n\n        samples = _build_samples_from_rows(rows=train_rows, variant=str(args.train_variant), chat_format=chat_format)\n        rng.shuffle(samples)\n\n        num_clients = int(args.num_clients)\n        alpha = getattr(args, "dirichlet_alpha", None)\n        if alpha is not None and float(alpha) > 0.0:\n            client_splits = _split_clients_dirichlet(\n                samples=samples, num_clients=num_clients, alpha=float(alpha), seed=int(args.seed) + 11\n            )\n            client_test_splits = _split_clients_dirichlet(\n                samples=test_samples, num_clients=num_clients, alpha=float(alpha), seed=int(args.seed) + 29\n            )\n            logger.info(\n                f"data_split=dirichlet alpha={float(alpha)} train_variant={args.train_variant} test_variant={args.test_variant}"\n            )\n        else:\n            client_splits = _split_clients_iid(samples, num_clients)\n            client_test_splits = _split_clients_iid(test_samples, num_clients)\n            logger.info(f"data_split=iid train_variant={args.train_variant} test_variant={args.test_variant}")\n\n        for cid in range(num_clients):\n            tr = client_splits[cid]\n            te = client_test_splits[cid]\n            tr_zh = sum(1 for s in tr if str(getattr(s, "lang", "zh")).lower() == "zh")\n            te_zh = sum(1 for s in te if str(getattr(s, "lang", "zh")).lower() == "zh")\n            logger.info(\n                f"client={cid} train_samples={len(tr)} (zh={tr_zh} en={len(tr)-tr_zh}) "\n                f"test_samples={len(te)} (zh={te_zh} en={len(te)-te_zh})"\n            )\n    elif run_mode == "zero_shot":\n        method = "zero_shot"\n        logger.info(f"run_mode=zero_shot test_variant={args.test_variant} test_samples={len(test_samples)}")\n    else:\n        raise ValueError(f"未知 run_mode={run_mode}, 仅支持 train/zero_shot")\n\n    if os.path.isdir(model_path):\n        tok_files = _list_tokenizer_files(model_path)\n        has_any_tokenizer = any(\n            n in tok_files for n in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model", "spiece.model", "vocab.txt"]\n        )\n        if not has_any_tokenizer:\n            files_preview = ", ".join(tok_files) if len(tok_files) > 0 else "(空)"\n            raise SystemExit(\n                "模型目录缺少 tokenizer 相关文件，无法加载分词器。\\n"\n                f"model_path={model_path}\\n"\n                f"已存在文件: {files_preview}\\n"\n                "请重新下载完整模型到该目录，或把 tokenizer.json / tokenizer.model / vocab.json 等文件拷贝进来。\\n"\n                "如果你是从 ModelScope 下载，建议运行：\\n"\n                f"  python download_qwen3_4b_modelscope.py --model_id Qwen/Qwen3-4B --out_dir {model_path} --verbose\\n"\n                "并确保 transformers 版本足够新(建议 >=4.51.0)。"\n            )\n\n    try:\n        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)\n    except TypeError as e:\n        msg = str(e)\n        if "vocab_file" in msg and "NoneType" in msg:\n            tok_files = _list_tokenizer_files(model_path) if os.path.isdir(model_path) else []\n            files_preview = ", ".join(tok_files) if len(tok_files) > 0 else "(空)"\n            raise SystemExit(\n                "加载 tokenizer 失败：疑似缺少 vocab.json/tokenizer 文件，导致 vocab_file=None。\\n"\n                f"model_path={model_path}\\n"\n                f"已存在文件: {files_preview}\\n"\n                "请补齐 tokenizer 文件或重新下载完整模型。"\n            ) from e\n        raise\n    if getattr(tokenizer, "pad_token_id", None) is None:\n        fallback_pad_id = getattr(tokenizer, "eod_id", None)\n        if fallback_pad_id is None:\n            fallback_pad_id = getattr(tokenizer, "eos_token_id", None)\n        if fallback_pad_id is not None:\n            try:\n                tokenizer.pad_token_id = int(fallback_pad_id)\n            except Exception:\n                pass\n\n    dtype = (\n        torch.bfloat16\n        if (device.type == "cuda" and torch.cuda.is_bf16_supported())\n        else torch.float16\n        if device.type == "cuda"\n        else torch.float32\n    )\n    model_load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}\n    model = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)\n    model.to(device)\n\n    if bool(getattr(args, "gradient_checkpointing", False)):\n        try:\n            model.gradient_checkpointing_enable()\n        except Exception:\n            pass\n    try:\n        if hasattr(model, "config"):\n            model.config.use_cache = False\n    except Exception:\n        pass\n\n    if run_mode == "zero_shot":\n        system_prompt = str(getattr(args, "system_prompt", "")).strip()\n        if system_prompt == "":\n            system_prompt = (\n                "你是一个擅长数学推理的助手。请逐步推理，并在最后一行只输出：#### <最终答案>。\\n"\n                "You are a math reasoning assistant. Reason step by step, and on the last line output only: #### <final answer>."\n            )\n\n        effective_chat_format = chat_format\n        if chat_format == "chatml":\n            try:\n                for s in test_samples:\n                    s.prompt_token_ids = _make_chatml_prompt_tokens(\n                        tokenizer=tokenizer, query=str(s.prompt), system=system_prompt\n                    )\n            except Exception:\n                effective_chat_format = "raw"\n                for s in test_samples:\n                    q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n                    raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n                    s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n        else:\n            for s in test_samples:\n                q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n                raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n                s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n\n        eval_rng = np.random.RandomState(int(args.seed) + 1)\n        eval_samples = _subsample_samples(test_samples, float(getattr(args, "eval_ratio", 1.0)), eval_rng)\n        eval_acc, eval_correct, eval_total = _evaluate_gsm8k_accuracy(\n            model=model,\n            tokenizer=tokenizer,\n            samples=eval_samples,\n            chat_format=effective_chat_format,\n            max_length=int(args.max_length),\n            max_new_tokens=int(getattr(args, "eval_max_new_tokens", 128)),\n            device=device,\n        )\n        logger.info(f"zero_shot overall acc={eval_acc:.4f} correct={eval_correct} total={eval_total} samples={len(eval_samples)}")\n\n        eval_zh = [s for s in eval_samples if str(getattr(s, "lang", "zh")).lower() == "zh"]\n        eval_en = [s for s in eval_samples if str(getattr(s, "lang", "zh")).lower() != "zh"]\n        if len(eval_zh) > 0 and len(eval_zh) != len(eval_samples):\n            a, c, t = _evaluate_gsm8k_accuracy(\n                model=model,\n                tokenizer=tokenizer,\n                samples=eval_zh,\n                chat_format=effective_chat_format,\n                max_length=int(args.max_length),\n                max_new_tokens=int(getattr(args, "eval_max_new_tokens", 128)),\n                device=device,\n            )\n            logger.info(f"zero_shot zh acc={a:.4f} correct={c} total={t} samples={len(eval_zh)}")\n        if len(eval_en) > 0 and len(eval_en) != len(eval_samples):\n            a, c, t = _evaluate_gsm8k_accuracy(\n                model=model,\n                tokenizer=tokenizer,\n                samples=eval_en,\n                chat_format=effective_chat_format,\n                max_length=int(args.max_length),\n                max_new_tokens=int(getattr(args, "eval_max_new_tokens", 128)),\n                device=device,\n            )\n            logger.info(f"zero_shot en acc={a:.4f} correct={c} total={t} samples={len(eval_en)}")\n        return\n\n    try:\n        from peft import LoraConfig, get_peft_model, TaskType\n    except ModuleNotFoundError as e:\n        raise SystemExit("Missing dependency \'peft\'. Install with: pip install -r requirements.txt") from e\n\n    target_modules = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]\n    peft_config = LoraConfig(\n        r=int(args.lora_rank),\n        lora_alpha=float(args.lora_alpha),\n        lora_dropout=float(args.lora_dropout),\n        target_modules=target_modules,\n        bias="none",\n        task_type=TaskType.CAUSAL_LM,\n    )\n    model = get_peft_model(model, peft_config)\n    if bool(getattr(args, "lora_qv_only", False)):\n        try:\n            from models import _patch_peft_qkv_qv_only\n\n            patched = 0\n            for name, module in model.named_modules():\n                low = str(name).lower()\n                if low.endswith("c_attn") or ".c_attn" in low:\n                    _patch_peft_qkv_qv_only(module)\n                    if getattr(module, "_qv_only_patched", False):\n                        patched += 1\n            logger.info(f"lora_qv_only_patched={patched}")\n        except Exception as e:\n            logger.info(f"lora_qv_only_patch_failed err={type(e).__name__}:{e}")\n    try:\n        if hasattr(model, "config"):\n            model.config.use_cache = False\n    except Exception:\n        pass\n    try:\n        if hasattr(model, "base_model") and hasattr(model.base_model, "config"):\n            model.base_model.config.use_cache = False\n    except Exception:\n        pass\n    try:\n        if hasattr(model, "get_base_model") and callable(model.get_base_model):\n            bm = model.get_base_model()\n            if hasattr(bm, "config"):\n                bm.config.use_cache = False\n    except Exception:\n        pass\n    try:\n        if hasattr(model, "generation_config") and model.generation_config is not None:\n            model.generation_config.use_cache = False\n    except Exception:\n        pass\n\n    lora_layers = _extract_peft_lora_layers(model, adapter_name="default")\n    if len(lora_layers) == 0:\n        raise ValueError(f"未找到任何 LoRA 层，请检查 lora_targets={args.lora_targets}")\n\n    try:\n        cfg = getattr(model, "config", None)\n        if cfg is None and hasattr(model, "base_model"):\n            cfg = getattr(getattr(model, "base_model"), "config", None)\n        if cfg is not None:\n            n_layers = getattr(cfg, "num_hidden_layers", None)\n            hidden_size = getattr(cfg, "hidden_size", None)\n            logger.info(f"model_num_hidden_layers={n_layers} model_hidden_size={hidden_size}")\n    except Exception:\n        pass\n\n    if chat_format == "chatml":\n        system_prompt = str(args.system_prompt)\n        for s in samples:\n            s.prompt_token_ids = _make_chatml_prompt_tokens(tokenizer=tokenizer, query=str(s.prompt), system=system_prompt)\n        for s in test_samples:\n            s.prompt_token_ids = _make_chatml_prompt_tokens(tokenizer=tokenizer, query=str(s.prompt), system=system_prompt)\n\n    from server import FedHarpServer\n    server = FedHarpServer(\n        model=model,\n        lora_layers=lora_layers,\n        num_clients=num_clients,\n        allocation_ratio=float(args.allocation_ratio),\n        aggregation_lr=float(args.aggregation_lr),\n        struct_to_data_rounds=int(args.struct_to_data_rounds),\n        seed=int(args.seed),\n        method=method,\n    )\n\n    num_total_layers = len(server.layer_names)\n    logger.info(f"lora_layer_total={num_total_layers} allocation_ratio={float(args.allocation_ratio)}")\n    if num_total_layers > 0:\n        preview = ",".join(server.layer_names[: min(12, num_total_layers)])\n        logger.info(f"lora_layer_examples={preview}")\n    scheme = str(getattr(args, "allocation_scheme", "uniform")).lower().strip()\n    if scheme == "tiered_631":\n        perm = rng.permutation(num_clients).tolist()\n        n_low = int(round(num_clients * 0.6))\n        n_mid = int(round(num_clients * 0.3))\n        n_low = max(0, min(num_clients, n_low))\n        n_mid = max(0, min(num_clients - n_low, n_mid))\n        n_high = int(num_clients - n_low - n_mid)\n        low_set = set(int(x) for x in perm[:n_low])\n        mid_set = set(int(x) for x in perm[n_low : n_low + n_mid])\n        high_set = set(int(x) for x in perm[n_low + n_mid :])\n\n        def budget_ratio(cid: int) -> float:\n            if cid in high_set:\n                return 1.0\n            if cid in mid_set:\n                return 0.75\n            return 0.5\n\n        for cid in range(num_clients):\n            r = float(budget_ratio(int(cid)))\n            server.client_layer_counts[int(cid)] = max(1, int(np.ceil(float(num_total_layers) * r)))\n        logger.info(\n            f"allocation_scheme=tiered_631 clients_low={len(low_set)} clients_mid={len(mid_set)} clients_high={len(high_set)}"\n        )\n    else:\n        for cid in range(num_clients):\n            server.client_layer_counts[cid] = max(1, int(num_total_layers * float(args.allocation_ratio)))\n\n    client_sample_counts: Dict[int, int] = {cid: len(client_splits[cid]) for cid in range(num_clients)}\n    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n\n    clients_per_round = int(args.clients_per_round) if args.clients_per_round is not None else num_clients\n\n    for round_num in range(1, int(args.num_rounds) + 1):\n        t_round = time.time()\n        logger.info(f"round={round_num} start")\n        allocation_map = server.start_round()\n        selected = np.arange(num_clients)\n        if clients_per_round < num_clients:\n            selected = rng.choice(num_clients, size=clients_per_round, replace=False)\n\n        global_A = server.get_global_A_matrices()\n        global_B = server.get_global_B_matrices() if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"} else None\n\n        client_updates_A: Dict[int, Dict[str, torch.Tensor]] = {}\n        client_updates_B: Dict[int, Dict[str, torch.Tensor]] = {}\n        client_train_metrics: Dict[int, Dict[str, object]] = {}\n\n        for cid in selected:\n            cid_int = int(cid)\n            allocated_layers = allocation_map.get(cid_int, set())\n            logger.info(\n                f"round={round_num} client={cid_int} allocation layers={len(allocated_layers)}/{num_total_layers} "\n                f"budget={int(server.client_layer_counts.get(cid_int, 0))}"\n            )\n            down_bytes = _dict_tensor_bytes(global_A) + (_dict_tensor_bytes(global_B) if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"} else 0)\n            _set_all_A(lora_layers, global_A)\n            if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"}:\n                if global_B is None:\n                    raise ValueError("fedra 需要 global_B")\n                _set_all_B(lora_layers, global_B)\n\n            ds = _GSM8KDataset(client_splits[cid_int])\n            loader = DataLoader(\n                ds,\n                batch_size=int(args.batch_size),\n                shuffle=True,\n                collate_fn=lambda b: _collate_causal_lm(\n                    b,\n                    tokenizer=tokenizer,\n                    max_length=int(args.max_length),\n                    device=device,\n                    chat_format=chat_format,\n                    train_on_inputs=bool(getattr(args, "train_on_inputs", False)),\n                ),\n            )\n\n            if device.type == "cuda":\n                try:\n                    torch.cuda.reset_peak_memory_stats(device=device)\n                except Exception:\n                    pass\n                try:\n                    torch.cuda.synchronize(device=device)\n                except Exception:\n                    pass\n            t_client_train = time.time()\n            delta_A, delta_B, b_sense, new_B_state, train_avg_loss = _train_one_client_round(\n                model=model,\n                lora_layers=lora_layers,\n                dataloader=loader,\n                allocated_layers=set(allocated_layers),\n                method=method,\n                lr=float(args.client_lr),\n                num_epochs=int(args.num_epochs),\n                grad_accum_steps=int(args.grad_accum_steps),\n                max_steps=None,\n                device=device,\n                client_B_state=client_B_states.get(cid_int),\n                client_id=cid_int,\n                round_num=round_num,\n                logger=logger,\n                log_interval=int(args.log_interval),\n            )\n            if device.type == "cuda":\n                try:\n                    torch.cuda.synchronize(device=device)\n                except Exception:\n                    pass\n            train_time_sec = float(time.time() - t_client_train)\n            upload_bytes = _dict_tensor_bytes(delta_A) + (_dict_tensor_bytes(delta_B) if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"} else 0)\n            peak_alloc = int(torch.cuda.max_memory_allocated(device=device)) if device.type == "cuda" else 0\n            peak_reserved = int(torch.cuda.max_memory_reserved(device=device)) if device.type == "cuda" else 0\n            client_train_metrics[cid_int] = {\n                "train_time_sec": train_time_sec,\n                "train_avg_loss": float(train_avg_loss),\n                "upload_bytes": int(upload_bytes),\n                "download_bytes": int(down_bytes),\n                "train_peak_alloc_bytes": int(peak_alloc),\n                "train_peak_reserved_bytes": int(peak_reserved),\n            }\n            logger.info(\n                f"round={round_num} client={cid_int} train_done time_sec={train_time_sec:.2f} "\n                f"upload_bytes={upload_bytes} download_bytes={down_bytes} peak_alloc={peak_alloc} peak_reserved={peak_reserved}"\n            )\n            client_updates_A[cid_int] = delta_A\n            if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"}:\n                if delta_B is None:\n                    raise ValueError("fedra 需要 delta_B")\n                client_updates_B[cid_int] = delta_B\n            if new_B_state is not None and method not in {"fedra", "ffa-lora", "ffalora", "ffa_lora"}:\n                client_B_states[cid_int] = new_B_state\n            server.update_client_sensitivities(cid_int, b_sense)\n\n        server.aggregate_matrices(\n            client_updates_A=client_updates_A,\n            client_updates_B=(client_updates_B if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"} else None),\n            client_sample_counts=client_sample_counts,\n            allocation_map=allocation_map,\n        )\n\n        global_A_after = server.get_global_A_matrices()\n        global_B_after = server.get_global_B_matrices() if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"} else None\n\n        eval_every_rounds = int(getattr(args, "eval_every_rounds", 5) or 5)\n        if eval_every_rounds < 1:\n            eval_every_rounds = 1\n        should_eval = (round_num % eval_every_rounds == 0) or (round_num == int(args.num_rounds))\n        if not should_eval:\n            dt = time.time() - t_round\n            logger.info(f"round={round_num} eval_skip eval_every_rounds={eval_every_rounds}")\n            logger.info(f"round={round_num} done method={method} selected={len(selected)} time_sec={dt:.2f}")\n            continue\n\n        eval_ratio = float(getattr(args, "eval_ratio", 0.1))\n        round_eval_correct = 0\n        round_eval_total = 0\n        for cid_int in range(num_clients):\n            split = client_test_splits[cid_int]\n            if len(split) == 0:\n                eval_subset: List[_Sample] = []\n            else:\n                k = int(max(1, int(len(split) * eval_ratio)))\n                k = min(k, len(split))\n                rng_eval = np.random.RandomState(int(args.seed) + 1000 * int(round_num) + int(cid_int))\n                idx = rng_eval.choice(len(split), size=k, replace=False)\n                eval_subset = [split[int(i)] for i in idx]\n\n            _set_all_A(lora_layers, global_A_after)\n            if method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"}:\n                if global_B_after is None:\n                    raise ValueError("fedra 需要 global_B")\n                _set_all_B(lora_layers, global_B_after)\n            else:\n                b_state = client_B_states.get(cid_int)\n                if b_state is not None:\n                    _set_all_B(lora_layers, b_state)\n\n            if device.type == "cuda":\n                try:\n                    torch.cuda.reset_peak_memory_stats(device=device)\n                except Exception:\n                    pass\n                try:\n                    torch.cuda.synchronize(device=device)\n                except Exception:\n                    pass\n            t_eval = time.time()\n            eval_avg_loss = _evaluate_on_samples(\n                model=model,\n                tokenizer=tokenizer,\n                samples=eval_subset,\n                chat_format=chat_format,\n                batch_size=int(args.eval_batch_size),\n                max_length=int(args.max_length),\n                train_on_inputs=bool(getattr(args, "train_on_inputs", False)),\n                device=device,\n            )\n            eval_acc, eval_correct, eval_total = _evaluate_gsm8k_accuracy(\n                model=model,\n                tokenizer=tokenizer,\n                samples=eval_subset,\n                chat_format=chat_format,\n                max_length=int(args.max_length),\n                max_new_tokens=int(getattr(args, "eval_max_new_tokens", 64)),\n                device=device,\n            )\n            if device.type == "cuda":\n                try:\n                    torch.cuda.synchronize(device=device)\n                except Exception:\n                    pass\n            eval_time_sec = float(time.time() - t_eval)\n            eval_peak_alloc = int(torch.cuda.max_memory_allocated(device=device)) if device.type == "cuda" else 0\n            eval_peak_reserved = int(torch.cuda.max_memory_reserved(device=device)) if device.type == "cuda" else 0\n\n            train_m = client_train_metrics.get(cid_int, {})\n            row: Dict[str, object] = {\n                "round": int(round_num),\n                "client": int(cid_int),\n                "method": str(method),\n                "train_time_sec": float(train_m.get("train_time_sec", 0.0)),\n                "train_avg_loss": float(train_m.get("train_avg_loss", 0.0)),\n                "eval_time_sec": float(eval_time_sec),\n                "eval_avg_loss": float(eval_avg_loss),\n                "eval_acc": float(eval_acc),\n                "eval_correct": int(eval_correct),\n                "eval_total": int(eval_total),\n                "eval_max_new_tokens": int(getattr(args, "eval_max_new_tokens", 64)),\n                "upload_bytes": int(train_m.get("upload_bytes", 0)),\n                "download_bytes": int(train_m.get("download_bytes", 0)),\n                "train_peak_alloc_bytes": int(train_m.get("train_peak_alloc_bytes", 0)),\n                "train_peak_reserved_bytes": int(train_m.get("train_peak_reserved_bytes", 0)),\n                "eval_peak_alloc_bytes": int(eval_peak_alloc),\n                "eval_peak_reserved_bytes": int(eval_peak_reserved),\n                "eval_samples": int(len(eval_subset)),\n                "client_train_samples": int(client_sample_counts.get(cid_int, 0)),\n                "client_test_samples": int(len(split)),\n            }\n            _append_metrics_jsonl(metrics_path, row)\n            logger.info(\n                f"round={round_num} client={cid_int} eval_done samples={len(eval_subset)} "\n                f"loss={eval_avg_loss:.6f} acc={eval_acc:.4f} "\n                f"time_sec={eval_time_sec:.2f} peak_alloc={eval_peak_alloc} peak_reserved={eval_peak_reserved}"\n            )\n            round_eval_correct += int(eval_correct)\n            round_eval_total += int(eval_total)\n\n        round_eval_acc = float(round_eval_correct) / float(max(1, round_eval_total))\n        _append_metrics_jsonl(\n            metrics_path,\n            {\n                "round": int(round_num),\n                "client": -1,\n                "method": str(method),\n                "row_type": "round_summary",\n                "eval_acc_weighted": float(round_eval_acc),\n                "eval_correct": int(round_eval_correct),\n                "eval_total": int(round_eval_total),\n            },\n        )\n        logger.info(\n            f"round={round_num} eval_summary weighted_acc={round_eval_acc:.4f} "\n            f"correct={round_eval_correct} total={round_eval_total}"\n        )\n\n        dt = time.time() - t_round\n        logger.info(f"round={round_num} done method={method} selected={len(selected)} time_sec={dt:.2f}")\n\n\nif __name__ == "__main__":\n    main()\n',
}


def _patch_embedded_task_sources_for_personalized_eval():
    """Keep embedded NLP/GSM8K entries aligned with the unified personalized eval path."""
    def replace_once_checked(source: str, old: str, new: str, label: str) -> str:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"embedded source patch {label!r} expected exactly one match, found {count}"
            )
        return source.replace(old, new, 1)

    roberta = _EMBEDDED_TASK_SOURCES.get("roberta")
    if roberta:
        # The local Roberta directory contains both safetensors and PyTorch
        # weights, but its safetensors metadata is incomplete for some
        # transformers releases.  Force the compatible PyTorch checkpoint.
        roberta = roberta.replace(
            "            weights_only=False,",
            "            use_safetensors=False,\n            weights_only=False,",
            1,
        )
        roberta = roberta.replace(
            "model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=int(num_labels))",
            "model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=int(num_labels), use_safetensors=False)",
            1,
        )
        roberta = roberta.replace(
            '            num_labels=int(num_labels), \\n            weights_only=False,\\n',
            '            num_labels=int(num_labels), \\n            use_safetensors=False,\\n            weights_only=False,\\n',
            1,
        )
        roberta = roberta.replace(
            '    p.add_argument("--alpha", type=float, default=0.5)\n',
            '    p.add_argument("--data_partition", type=str, default="dirichlet", choices=["dirichlet", "uniform"])\n'
            '    p.add_argument("--alpha", type=float, default=0.5)\n',
            1,
        )
        roberta = roberta.replace(
            'def _dirichlet_partition(\n',
            'def _uniform_partition(\n'
            '    labels: np.ndarray,\n'
            '    num_clients: int,\n'
            '    seed: int,\n'
            '):\n'
            '    rng = np.random.RandomState(int(seed))\n'
            '    client_indices: List[List[int]] = [[] for _ in range(int(num_clients))]\n'
            '    for cls in np.unique(labels):\n'
            '        cls_idx = np.where(labels == cls)[0]\n'
            '        rng.shuffle(cls_idx)\n'
            '        for cid, chunk in enumerate(np.array_split(cls_idx, int(num_clients))):\n'
            '            client_indices[cid].extend(int(x) for x in chunk.tolist())\n'
            '    for cid in range(int(num_clients)):\n'
            '        rng.shuffle(client_indices[cid])\n'
            '    return client_indices\n\n\n'
            'def _dirichlet_partition(\n',
            1,
        )
        roberta = roberta.replace(
            '    client_indices = _dirichlet_partition(\n'
            '        train_labels,\n'
            '        num_clients=args.num_clients,\n'
            '        alpha=args.alpha,\n'
            '        seed=args.seed,\n'
            '    )\n',
            '    data_partition = str(getattr(args, "data_partition", "dirichlet")).lower().strip()\n'
            '    if data_partition == "uniform":\n'
            '        client_indices = _uniform_partition(\n'
            '            train_labels,\n'
            '            num_clients=args.num_clients,\n'
            '            seed=args.seed,\n'
            '        )\n'
            '        print("Data partition: uniform (stratified IID; alpha ignored)")\n'
            '    else:\n'
            '        client_indices = _dirichlet_partition(\n'
            '            train_labels,\n'
            '            num_clients=args.num_clients,\n'
            '            alpha=args.alpha,\n'
            '            seed=args.seed,\n'
            '        )\n'
            '        print(f"Data partition: dirichlet alpha={args.alpha}")\n',
            1,
        )
        roberta = roberta.replace(
            '    safe_heterogeneity = _safe_token(args.heterogeneity_type)\n'
            '    log_filename = f"Traininglog-{safe_method}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_heterogeneity}.txt"\n',
            '    safe_heterogeneity = _safe_token(args.heterogeneity_type)\n'
            '    safe_partition = _safe_token(getattr(args, "data_partition", "dirichlet"))\n'
            '    log_filename = f"Traininglog-{safe_method}-{safe_partition}-alpha{safe_alpha}-{safe_dataset}-{safe_model}-{safe_heterogeneity}.txt"\n',
            1,
        )
        roberta = roberta.replace(
            '            "flex_lora",\n        ],',
            '            "flex_lora",\n'
            '            "fedit",\n'
            '            "fed-it",\n'
            '            "fed_it",\n'
            '        ],',
        )
        roberta = roberta.replace(
            'needs_b = method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora"}',
            'needs_b = method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}',
        )
        roberta = roberta.replace(
            '    p.add_argument("--eval_on", type=str, default="val", choices=["val", "client_holdout"])\n\n'
            '    p.add_argument("--seed", type=int, default=42)\n',
            '    p.add_argument("--eval_on", type=str, default="val", choices=["val", "client_holdout"])\n'
            '    p.add_argument("--eval_scope", type=str, default="personalized", choices=["global_test", "personalized"])\n'
            '    p.add_argument("--fedharp_matrix_mode", type=str, default="a", choices=["a", "b", "ab", "adaptive", "auto"])\n'
            '    p.add_argument("--adaptive_matrix_threshold", type=float, default=1.25)\n\n'
            '    p.add_argument("--allocation_temperature", type=float, default=1.0)\n'
            '    p.add_argument("--coverage_prior_weight", type=float, default=0.15)\n'
            '    p.add_argument("--sample_weight_power", type=float, default=0.5)\n\n'
            '    p.add_argument("--seed", type=int, default=42)\n',
        )
        roberta = roberta.replace(
            '    p.add_argument("--lora_targets", type=str, default="query,value")\n\n'
            '    p.add_argument("--batch_size", type=int, default=16)\n',
            '    p.add_argument("--lora_targets", type=str, default="query,value")\n'
            '    p.add_argument("--flora_rank_policy", type=str, default="capacity", choices=["capacity", "uniform"])\n'
            '    p.add_argument("--flora_min_rank", type=int, default=1)\n'
            '    p.add_argument("--flora_local_ranks", type=str, default=None)\n\n'
            '    p.add_argument("--batch_size", type=int, default=16)\n',
            1,
        )
        roberta = roberta.replace(
            'def _build_fedra_allocation_map(\n',
            'def _build_flora_client_ranks(\n'
            '    client_capacities: List[float],\n'
            '    max_rank: int,\n'
            '    policy: str = "capacity",\n'
            '    min_rank: int = 1,\n'
            '    explicit_ranks: Optional[str] = None,\n'
            ') -> List[int]:\n'
            '    num_clients = len(client_capacities)\n'
            '    max_rank = max(1, int(max_rank))\n'
            '    min_rank = max(1, min(int(min_rank), max_rank))\n'
            '    if explicit_ranks is not None and str(explicit_ranks).strip() != "":\n'
            '        ranks = [int(x.strip()) for x in str(explicit_ranks).split(",") if x.strip()]\n'
            '        if len(ranks) == 1:\n'
            '            ranks = ranks * num_clients\n'
            '        if len(ranks) != num_clients:\n'
            '            raise ValueError(f"flora_local_ranks expects 1 or {num_clients} values, got {len(ranks)}")\n'
            '        return [max(min_rank, min(max_rank, int(rank))) for rank in ranks]\n'
            '    if str(policy).lower().strip() == "uniform":\n'
            '        return [max_rank] * num_clients\n'
            '    return [\n'
            '        max(min_rank, min(max_rank, int(round(max_rank * float(capacity)))))\n'
            '        for capacity in client_capacities\n'
            '    ]\n\n\n'
            'def _build_fedra_allocation_map(\n',
            1,
        )
        roberta = roberta.replace(
            '    method = str(args.method).lower()\n'
            '    server = RobertaFedHarpServer(\n',
            '    method = str(args.method).lower()\n'
            '    flora_client_ranks = _build_flora_client_ranks(\n'
            '        client_capacities=client_capacities,\n'
            '        max_rank=args.lora_rank,\n'
            '        policy=getattr(args, "flora_rank_policy", "capacity"),\n'
            '        min_rank=getattr(args, "flora_min_rank", 1),\n'
            '        explicit_ranks=getattr(args, "flora_local_ranks", None),\n'
            '    )\n'
            '    if method == "flora":\n'
            '        print(\n'
            '            f"FLoRA heterogeneous ranks: policy={args.flora_rank_policy}, "\n'
            '            f"mean={float(np.mean(flora_client_ranks)):.2f}, "\n'
            '            f"min={min(flora_client_ranks)}, max={max(flora_client_ranks)}, "\n'
            '            f"ranks={flora_client_ranks}"\n'
            '        )\n'
            '    server = RobertaFedHarpServer(\n',
            1,
        )
        roberta = roberta.replace(
            '        client_model, client_lora_layers = _clone_model_and_layers(global_model, global_lora_layers)\n'
            '        client_model.to(device)\n'
            '        client_sample_counts[cid] = len(train_ds)\n',
            '        client_model, client_lora_layers = _clone_model_and_layers(global_model, global_lora_layers)\n'
            '        client_model.to(device)\n'
            '        if method == "flora":\n'
            '            for layer in client_lora_layers.values():\n'
            '                layer.reset_lora(rank=flora_client_ranks[cid])\n'
            '        client_sample_counts[cid] = len(train_ds)\n',
            1,
        )
        roberta = roberta.replace(
            '            if method == "flora":\n'
            '                client.flora_merge_and_reset(target_rank=args.lora_rank)\n',
            '            if method == "flora":\n'
            '                client.flora_merge_and_reset(target_rank=flora_client_ranks[cid])\n',
            1,
        )
        roberta = roberta.replace(
            '        method=args.method,\n'
            '    )\n'
            '    if hasattr(server, "client_layer_counts"):\n',
            '        method=args.method,\n'
            '        fedharp_matrix_mode=("adaptive" if str(getattr(args, "fedharp_matrix_mode", "a")).lower() == "auto" else getattr(args, "fedharp_matrix_mode", "a")),\n'
            '        allocation_temperature=float(args.allocation_temperature),\n'
            '        coverage_prior_weight=float(args.coverage_prior_weight),\n'
            '        sample_weight_power=float(args.sample_weight_power),\n'
            '    )\n'
            '    if hasattr(server, "client_layer_counts"):\n',
            1,
        )
        roberta = roberta.replace(
            '        else:\n'
            '            test_ds = SST2ClientDataset(encoded_val, list(range(len(encoded_val.labels))))\n',
            '        else:\n'
            '            val_chunks = np.array_split(np.arange(len(encoded_val.labels)), int(args.num_clients))\n'
            '            val_idx = [int(x) for x in val_chunks[cid].tolist()]\n'
            '            if len(val_idx) == 0:\n'
            '                val_idx = list(range(len(encoded_val.labels)))\n'
            '            test_ds = SST2ClientDataset(encoded_val, val_idx)\n',
        )
        roberta = roberta.replace(
            '    b_train_every = int(args.b_train_every) if (is_fedharp_family and method != "fedharp_l") else 0\n\n    for round_num in range(1, args.num_rounds + 1):',
            '    fedharp_matrix_mode = str(getattr(args, "fedharp_matrix_mode", "a")).lower().replace("-", "_")\n'
            '    if fedharp_matrix_mode == "auto":\n'
            '        fedharp_matrix_mode = "adaptive"\n'
            '    if fedharp_matrix_mode not in {"a", "b", "ab", "adaptive"}:\n'
            '        fedharp_matrix_mode = "adaptive"\n'
            '    fedharp_upload_B = is_fedharp_family and fedharp_matrix_mode in {"b", "ab", "adaptive"}\n'
            '    b_train_every = int(args.b_train_every) if (is_fedharp_family and method != "fedharp_l") else 0\n'
            '    client_personal_A_states = {cid: clients[cid].get_all_A_matrices() for cid in range(args.num_clients)}\n'
            '    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}\n'
            '    fedharp_adaptive_B_sync_layers: Set[str] = set()\n\n'
            '    for round_num in range(1, args.num_rounds + 1):',
        )
        roberta = roberta.replace(
            '        needs_b = method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}\n',
            '        aggregate_b_method = fedharp_upload_B or method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}\n'
            '        download_b_method = method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} or (is_fedharp_family and fedharp_matrix_mode in {"b", "ab"})\n',
        )
        roberta = roberta.replace(
            '        global_A = server.get_global_A_matrices()\n'
            '        global_B = server.get_global_B_matrices() if needs_b else None\n'
            '        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)\n'
            '        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0\n',
            '        global_A = server.get_global_A_matrices()\n'
            '        global_B_full = server.get_global_B_matrices() if (aggregate_b_method or download_b_method) else None\n'
            '        global_B = global_B_full if download_b_method else None\n'
            '        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)\n',
        )
        roberta = roberta.replace(
            '            client.receive_global_matrices(global_A, global_B, method=args.method)\n',
            '            client_global_B = global_B\n'
            '            if (\n'
            '                is_fedharp_family\n'
            '                and fedharp_matrix_mode == "adaptive"\n'
            '                and global_B_full is not None\n'
            '                and len(fedharp_adaptive_B_sync_layers) > 0\n'
            '            ):\n'
            '                sparse_layers = set(allocated_layers).intersection(fedharp_adaptive_B_sync_layers)\n'
            '                client_global_B = {name: global_B_full[name] for name in sparse_layers if name in global_B_full}\n'
            '            downlink_B_mb = (_A_matrices_nbytes(client_global_B) / (1024.0 * 1024.0)) if client_global_B is not None else 0.0\n'
            '            client.receive_global_matrices(global_A, client_global_B, method=args.method)\n',
        )
        roberta = roberta.replace(
            '            if b_train_every > 0 and is_fedharp_family and method != "fedharp_l":\n',
            '            if b_train_every > 0 and is_fedharp_family and method != "fedharp_l" and fedharp_matrix_mode != "adaptive":\n',
        )
        roberta = roberta.replace(
            '            if needs_b:\n'
            '                if delta_B is None:\n'
            '                    raise ValueError(f"{method} 需要上传 B")\n'
            '                uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)\n',
            '            if aggregate_b_method:\n'
            '                if delta_B is None:\n'
            '                    raise ValueError(f"{method} 需要上传 B")\n'
            '                uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)\n',
        )
        roberta = roberta.replace(
            '            client_updates[cid] = delta_A\n'
            '            if needs_b:\n'
            '                client_updates_B[cid] = delta_B\n'
            '            server.update_client_sensitivities(cid, b_sensitivity)\n',
            '            client_updates[cid] = delta_A\n'
            '            if aggregate_b_method:\n'
            '                client_updates_B[cid] = delta_B\n'
            '                if is_fedharp_family and fedharp_matrix_mode == "adaptive" and delta_B is not None:\n'
            '                    fedharp_adaptive_B_sync_layers.update(delta_B.keys())\n'
            '            server.update_client_sensitivities(cid, b_sensitivity)\n'
            '            if (\n'
            '                b_train_every > 0\n'
            '                and is_fedharp_family\n'
            '                and method != "fedharp_l"\n'
            '                and fedharp_matrix_mode == "adaptive"\n'
            '                and int(round_num) % max(1, int(b_train_every)) == 0\n'
            '            ):\n'
            '                factor_choices = getattr(client, "last_factor_choices", {})\n'
            '                if len(factor_choices) > 0:\n'
            '                    cfc_sensitivity = client.train_complementary_factors(\n'
            '                        factor_choices,\n'
            '                        num_epochs=int(args.b_num_epochs),\n'
            '                    )\n'
            '                    server.update_client_sensitivities(cid, cfc_sensitivity)\n',
        )
        roberta = roberta.replace(
            '        if needs_b:\n'
            '            server.aggregate_matrices(\n'
            '                client_updates_A=client_updates,\n'
            '                client_updates_B=client_updates_B,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n'
            '        else:\n'
            '            server.aggregate_A_matrices(\n'
            '                client_updates=client_updates,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n',
            '        if aggregate_b_method:\n'
            '            server.aggregate_matrices(\n'
            '                client_updates_A=client_updates,\n'
            '                client_updates_B=client_updates_B,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n'
            '        else:\n'
            '            server.aggregate_A_matrices(\n'
            '                client_updates=client_updates,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n',
        )
        roberta = roberta.replace(
            '            global_eval_B = server.get_global_B_matrices() if needs_b else None\n',
            '            global_eval_B = server.get_global_B_matrices() if aggregate_b_method else None\n',
        )
        roberta = roberta.replace(
            '                method=args.method,\n'
            '            )\n'
            '            train_time_sec = time.time() - t0\n',
            '                method=args.method,\n'
            '                matrix_update_mode=fedharp_matrix_mode,\n'
            '                adaptive_matrix_threshold=float(getattr(args, "adaptive_matrix_threshold", 1.25)),\n'
            '            )\n'
            '            train_time_sec = time.time() - t0\n',
            1,
        )
        roberta = roberta.replace(
            '            eval_A = server.get_global_A_matrices()\n'
            '            eval_B = server.get_global_B_matrices() if needs_b else None\n'
            '            for cid in range(args.num_clients):\n'
            '                client = clients[cid]\n'
            '                client.receive_global_matrices(eval_A, eval_B, method=args.method)\n',
            '            eval_scope = str(getattr(args, "eval_scope", "global_test")).lower().strip()\n'
            '            global_eval_A = server.get_global_A_matrices()\n'
            '            global_eval_B = server.get_global_B_matrices() if needs_b else None\n'
            '            for cid in range(args.num_clients):\n'
            '                client = clients[cid]\n'
            '                if eval_scope == "personalized":\n'
            '                    eval_A = client_personal_A_states.get(cid, global_eval_A)\n'
            '                    eval_B = client_personal_B_states.get(cid, server.get_global_B_matrices())\n'
            '                    client.set_A_matrices(eval_A)\n'
            '                    client.set_B_matrices(eval_B)\n'
            '                else:\n'
            '                    client.set_A_matrices(global_eval_A)\n'
            '                    if global_eval_B is not None:\n'
            '                        client.set_B_matrices(global_eval_B)\n'
            '                    else:\n'
            '                        client.set_B_matrices(client_personal_B_states.get(cid, server.get_global_B_matrices()))\n',
        )
        roberta = roberta.replace(
            '            global_eval_B = server.get_global_B_matrices() if needs_b else None\n',
            '            global_eval_B = server.get_global_B_matrices() if aggregate_b_method else None\n',
        )
        roberta = roberta.replace(
            '                    f.write(f"round={round_num}, type=personalized, accuracy={global_acc:.4f}, loss={global_loss:.6f}\\n")\n',
            '                    f.write(f"round={round_num}, type={eval_scope}, accuracy={global_acc:.4f}, loss={global_loss:.6f}\\n")\n',
        )
        roberta = roberta.replace(
            '    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}\n'
            '    fedharp_adaptive_B_sync_layers: Set[str] = set()\n\n',
            '    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}\n\n',
        )
        roberta = roberta.replace(
            '        aggregate_b_method = fedharp_upload_B or method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}\n'
            '        download_b_method = method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} or (is_fedharp_family and fedharp_matrix_mode in {"b", "ab"})\n',
            '        needs_b = fedharp_upload_B or method in {"fedhello", "fedra", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}\n',
        )
        roberta = roberta.replace(
            '        global_A = server.get_global_A_matrices()\n'
            '        global_B_full = server.get_global_B_matrices() if (aggregate_b_method or download_b_method) else None\n'
            '        global_B = global_B_full if download_b_method else None\n'
            '        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)\n',
            '        global_A = server.get_global_A_matrices()\n'
            '        global_B = server.get_global_B_matrices() if needs_b else None\n'
            '        downlink_A_mb = _A_matrices_nbytes(global_A) / (1024.0 * 1024.0)\n'
            '        downlink_B_mb = (_A_matrices_nbytes(global_B) / (1024.0 * 1024.0)) if global_B is not None else 0.0\n',
        )
        roberta = roberta.replace(
            '            client_global_B = global_B\n'
            '            if (\n'
            '                is_fedharp_family\n'
            '                and fedharp_matrix_mode == "adaptive"\n'
            '                and global_B_full is not None\n'
            '                and len(fedharp_adaptive_B_sync_layers) > 0\n'
            '            ):\n'
            '                sparse_layers = set(allocated_layers).intersection(fedharp_adaptive_B_sync_layers)\n'
            '                client_global_B = {name: global_B_full[name] for name in sparse_layers if name in global_B_full}\n'
            '            downlink_B_mb = (_A_matrices_nbytes(client_global_B) / (1024.0 * 1024.0)) if client_global_B is not None else 0.0\n'
            '            client.receive_global_matrices(global_A, client_global_B, method=args.method)\n',
            '            client.receive_global_matrices(global_A, global_B, method=args.method)\n',
        )
        roberta = roberta.replace(
            '            if b_train_every > 0 and is_fedharp_family and method != "fedharp_l" and fedharp_matrix_mode != "adaptive":\n',
            '            if b_train_every > 0 and is_fedharp_family and method != "fedharp_l":\n',
        )
        roberta = roberta.replace(
            '            if aggregate_b_method:\n'
            '                if delta_B is None:\n'
            '                    raise ValueError(f"{method} 需要上传 B")\n'
            '                uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)\n',
            '            if needs_b:\n'
            '                if delta_B is None:\n'
            '                    raise ValueError(f"{method} 需要上传 B")\n'
            '                uplink_B_mb = _delta_B_nbytes(delta_B) / (1024.0 * 1024.0)\n',
        )
        roberta = roberta.replace(
            '            client_updates[cid] = delta_A\n'
            '            if aggregate_b_method:\n'
            '                client_updates_B[cid] = delta_B\n'
            '                if is_fedharp_family and fedharp_matrix_mode == "adaptive" and delta_B is not None:\n'
            '                    fedharp_adaptive_B_sync_layers.update(delta_B.keys())\n'
            '            server.update_client_sensitivities(cid, b_sensitivity)\n'
            '            if (\n'
            '                b_train_every > 0\n'
            '                and is_fedharp_family\n'
            '                and method != "fedharp_l"\n'
            '                and fedharp_matrix_mode == "adaptive"\n'
            '                and int(round_num) % max(1, int(b_train_every)) == 0\n'
            '            ):\n'
            '                factor_choices = getattr(client, "last_factor_choices", {})\n'
            '                if len(factor_choices) > 0:\n'
            '                    cfc_sensitivity = client.train_complementary_factors(\n'
            '                        factor_choices,\n'
            '                        num_epochs=int(args.b_num_epochs),\n'
            '                    )\n'
            '                    server.update_client_sensitivities(cid, cfc_sensitivity)\n',
            '            client_updates[cid] = delta_A\n'
            '            if needs_b:\n'
            '                client_updates_B[cid] = delta_B\n'
            '            server.update_client_sensitivities(cid, b_sensitivity)\n',
        )
        roberta = roberta.replace(
            '        if aggregate_b_method:\n'
            '            server.aggregate_matrices(\n'
            '                client_updates_A=client_updates,\n'
            '                client_updates_B=client_updates_B,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n'
            '        else:\n'
            '            server.aggregate_A_matrices(\n'
            '                client_updates=client_updates,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n',
            '        if needs_b:\n'
            '            server.aggregate_matrices(\n'
            '                client_updates_A=client_updates,\n'
            '                client_updates_B=client_updates_B,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n'
            '        else:\n'
            '            server.aggregate_A_matrices(\n'
            '                client_updates=client_updates,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n',
        )
        roberta = roberta.replace(
            '            global_eval_B = server.get_global_B_matrices() if aggregate_b_method else None\n',
            '            global_eval_B = server.get_global_B_matrices() if needs_b else None\n',
        )
        roberta = roberta.replace(
            '    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")\n'
            '    print(f"Log saved to: {os.path.abspath(log_path)}")\n',
            '    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")\n'
            '    if (\n'
            '        hasattr(server, "uses_partial_layer_allocation")\n'
            '        and server.uses_partial_layer_allocation()\n'
            '    ):\n'
            '        server.print_underallocated_coverage_stats()\n'
            '    print(f"Log saved to: {os.path.abspath(log_path)}")\n',
        )
        roberta = roberta.replace(
            '            compute_tflops = (3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)) / 1e12\n\n'
            '            client_metrics_ab.append(\n',
            '            compute_flops = 3.0 * float(trainable_numel) * float(train_samples) * float(args.num_epochs)\n'
            '            compute_tflops = compute_flops / 1e12\n'
            '            flops_per_sec = compute_flops / max(1e-9, float(train_time_sec))\n'
            '            first_client_layer = next(iter(client.lora_layers.values()), None)\n'
            '            active_lora_rank = int(first_client_layer.lora_A.shape[0]) if first_client_layer is not None else 0\n\n'
            '            client_metrics_ab.append(\n',
            1,
        )
        roberta = roberta.replace(
            '                    "phase": "ab_train",\n'
            '                    "compute_flops": float(compute_flops),\n'
            '                    "compute_tflops": float(compute_tflops),\n'
            '                    "flops_per_sec": float(flops_per_sec),\n',
            '                    "phase": "ab_train",\n'
            '                    "lora_rank": int(active_lora_rank),\n'
            '                    "compute_tflops": float(compute_tflops),\n',
            1,
        )
        roberta = roberta.replace(
            '                        "round={round}, type=client_metrics, phase={phase}, client={client}, "\n'
            '                        "compute_flops={compute_flops:.3f}, compute_tflops={compute_tflops:.6f}, "\n'
            '                        "flops_per_sec={flops_per_sec:.3f}, peak_mem_gb={peak_mem_gb:.6f}, "\n',
            '                        "round={round}, type=client_metrics, phase={phase}, client={client}, "\n'
            '                        "lora_rank={lora_rank}, "\n'
            '                        "compute_tflops={compute_tflops:.6f}, peak_mem_gb={peak_mem_gb:.6f}, "\n',
            1,
        )
        roberta = roberta.replace(
            '            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))\n'
            '            try:\n',
            '            avg_time = float(np.mean([m["train_time_sec"] for m in client_metrics_ab]))\n'
            '            ranks = [int(m["lora_rank"]) for m in client_metrics_ab]\n'
            '            try:\n',
            1,
        )
        roberta = roberta.replace(
            '                        f"round={round_num}, type=round_metrics, phase=ab_train, "\n'
            '                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "\n',
            '                        f"round={round_num}, type=round_metrics, phase=ab_train, "\n'
            '                        f"avg_lora_rank={float(np.mean(ranks)):.4f}, min_lora_rank={min(ranks)}, max_lora_rank={max(ranks)}, "\n'
            '                        f"avg_compute_tflops={avg_compute:.6f}, avg_peak_mem_gb={avg_mem:.6f}, "\n',
            1,
        )
        roberta = roberta.replace(
            '                    "phase": "ab_train",\n'
            '                    "compute_tflops": float(compute_tflops),\n',
            '                    "phase": "ab_train",\n'
            '                    "compute_flops": float(compute_flops),\n'
            '                    "compute_tflops": float(compute_tflops),\n'
            '                    "flops_per_sec": float(flops_per_sec),\n',
            1,
        )
        roberta = roberta.replace(
            '                        "lora_rank={lora_rank}, "\n'
            '                        "compute_tflops={compute_tflops:.6f}, peak_mem_gb={peak_mem_gb:.6f}, "\n',
            '                        "lora_rank={lora_rank}, "\n'
            '                        "compute_flops={compute_flops:.3f}, compute_tflops={compute_tflops:.6f}, "\n'
            '                        "flops_per_sec={flops_per_sec:.3f}, peak_mem_gb={peak_mem_gb:.6f}, "\n',
            1,
        )
        # The earlier lora_rank dict-field replace runs BEFORE the replace
        # that inserts compute_flops into the metrics dict, so its old_str
        # never matches (dead code) and the round-statistics block above
        # then raises KeyError('lora_rank').  Insert the field here, after
        # the dict has reached its final shape.  active_lora_rank is defined
        # by the declaration replace further up the chain.
        roberta = roberta.replace(
            '                    "phase": "ab_train",\n'
            '                    "compute_flops": float(compute_flops),\n',
            '                    "phase": "ab_train",\n'
            '                    "lora_rank": int(active_lora_rank),\n'
            '                    "compute_flops": float(compute_flops),\n',
            1,
        )
        # Heterogeneous-rank baselines use a shared rank-R server adapter and
        # ordinary weighted FedAvg after zero-padding local rank-r_k updates.
        roberta = roberta.replace(
            '    p.add_argument("--flora_local_ranks", type=str, default=None)\n\n'
            '    p.add_argument("--batch_size", type=int, default=16)\n',
            '    p.add_argument("--flora_local_ranks", type=str, default=None)\n'
            '    p.add_argument("--hetero_rank_policy", type=str, default=None, choices=["capacity", "uniform"])\n'
            '    p.add_argument("--hetero_min_rank", type=int, default=None)\n'
            '    p.add_argument("--hetero_local_ranks", type=str, default=None)\n\n'
            '    p.add_argument("--batch_size", type=int, default=16)\n',
            1,
        )
        roberta = roberta.replace(
            '    method = str(args.method).lower()\n'
            '    flora_client_ranks = _build_flora_client_ranks(\n'
            '        client_capacities=client_capacities,\n'
            '        max_rank=args.lora_rank,\n'
            '        policy=getattr(args, "flora_rank_policy", "capacity"),\n'
            '        min_rank=getattr(args, "flora_min_rank", 1),\n'
            '        explicit_ranks=getattr(args, "flora_local_ranks", None),\n'
            '    )\n'
            '    if method == "flora":\n'
            '        print(\n'
            '            f"FLoRA heterogeneous ranks: policy={args.flora_rank_policy}, "\n'
            '            f"mean={float(np.mean(flora_client_ranks)):.2f}, "\n'
            '            f"min={min(flora_client_ranks)}, max={max(flora_client_ranks)}, "\n'
            '            f"ranks={flora_client_ranks}"\n'
            '        )\n'
            '    server = RobertaFedHarpServer(\n',
            '    method = str(args.method).lower()\n'
            '    rank_heterogeneity_methods = {\n'
            '        "flora", "flexlora", "flex-lora", "flex_lora",\n'
            '        "fedsalora", "fedsa-lora", "fedsa_lora",\n'
            '        "ffa-lora", "ffalora", "ffa_lora", "fedit", "fed-it", "fed_it",\n'
            '    }\n'
            '    zero_padded_rank_methods = rank_heterogeneity_methods.difference(\n'
            '        {"flora", "flexlora", "flex-lora", "flex_lora"}\n'
            '    )\n'
            '    rank_policy = getattr(args, "hetero_rank_policy", None) or getattr(args, "flora_rank_policy", "capacity")\n'
            '    rank_min_rank = getattr(args, "hetero_min_rank", None)\n'
            '    if rank_min_rank is None:\n'
            '        rank_min_rank = getattr(args, "flora_min_rank", 1)\n'
            '    rank_local_ranks = getattr(args, "hetero_local_ranks", None)\n'
            '    if rank_local_ranks is None:\n'
            '        rank_local_ranks = getattr(args, "flora_local_ranks", None)\n'
            '    if method != "flora" and not str(rank_local_ranks or "").strip():\n'
            '        rank_local_ranks = str(int(args.lora_rank))\n'
            '    flora_client_ranks = _build_flora_client_ranks(\n'
            '        client_capacities=client_capacities,\n'
            '        max_rank=args.lora_rank,\n'
            '        policy=rank_policy,\n'
            '        min_rank=rank_min_rank,\n'
            '        explicit_ranks=rank_local_ranks,\n'
            '    )\n'
            '    if method in rank_heterogeneity_methods:\n'
            '        rank_mode = "heterogeneous_rank" if len(set(flora_client_ranks)) > 1 else "fixed_rank"\n'
            '        if method == "flora":\n'
            '            aggregation = "stacking"\n'
            '        elif method in {"flexlora", "flex-lora", "flex_lora"}:\n'
            '            aggregation = "svd_full_update_average"\n'
            '        elif rank_mode == "heterogeneous_rank":\n'
            '            aggregation = "zero_padded_weighted_average"\n'
            '        else:\n'
            '            aggregation = "weighted_average"\n'
            '        print(\n'
            '            f"{method} rank mode={rank_mode}: policy={rank_policy}, "\n'
            '            f"aggregation={aggregation}, mean={float(np.mean(flora_client_ranks)):.2f}, "\n'
            '            f"min={min(flora_client_ranks)}, max={max(flora_client_ranks)}, "\n'
            '            f"ranks={flora_client_ranks}"\n'
            '        )\n'
            '        try:\n'
            '            with open(log_path, "a", encoding="utf-8") as f:\n'
            '                for cid, (capacity, rank) in enumerate(zip(client_capacities, flora_client_ranks)):\n'
            '                    f.write(\n'
            '                        f"type=rank_assignment, mode={rank_mode}, method={method}, aggregation={aggregation}, "\n'
            '                        f"client={cid}, capacity={float(capacity):.4f}, rank={int(rank)}\\n"\n'
            '                    )\n'
            '        except Exception as e:\n'
            '            print(f"[warning] failed to write heterogeneous rank assignment: {e}")\n'
            '    server = RobertaFedHarpServer(\n',
            1,
        )
        roberta = roberta.replace(
            '        client_model, client_lora_layers = _clone_model_and_layers(global_model, global_lora_layers)\n'
            '        client_model.to(device)\n'
            '        if method == "flora":\n'
            '            for layer in client_lora_layers.values():\n'
            '                layer.reset_lora(rank=flora_client_ranks[cid])\n'
            '        client_sample_counts[cid] = len(train_ds)\n',
            '        client_model, client_lora_layers = _clone_model_and_layers(global_model, global_lora_layers)\n'
            '        client_model.to(device)\n'
            '        if method in rank_heterogeneity_methods:\n'
            '            for layer in client_lora_layers.values():\n'
            '                layer.reset_lora(rank=flora_client_ranks[cid])\n'
            '        client_sample_counts[cid] = len(train_ds)\n',
            1,
        )
        roberta = roberta.replace(
            '            client.receive_global_matrices(global_A, global_B, method=args.method)\n'
            '            if method == "flora":\n'
            '                client.flora_merge_and_reset(target_rank=flora_client_ranks[cid])\n',
            '            client_global_A = global_A\n'
            '            client_global_B = global_B\n'
            '            if method in zero_padded_rank_methods:\n'
            '                client_global_A, client_global_B = client.prepare_global_matrices_for_rank(\n'
            '                    global_A, global_B, target_rank=flora_client_ranks[cid]\n'
            '                )\n'
            '            client.receive_global_matrices(client_global_A, client_global_B, method=args.method)\n'
            '            if method == "flora":\n'
            '                client.flora_merge_and_reset(target_rank=flora_client_ranks[cid])\n'
            '            client_downlink_A_mb = _A_matrices_nbytes(client_global_A) / (1024.0 * 1024.0)\n'
            '            client_downlink_B_mb = (\n'
            '                _A_matrices_nbytes(client_global_B) / (1024.0 * 1024.0)\n'
            '                if client_global_B is not None else 0.0\n'
            '            )\n',
            1,
        )
        roberta = roberta.replace(
            '            comm_mb = downlink_A_mb + downlink_B_mb + uplink_A_mb + uplink_B_mb\n',
            '            comm_mb = client_downlink_A_mb + client_downlink_B_mb + uplink_A_mb + uplink_B_mb\n',
            1,
        )
        roberta = roberta.replace(
            '                    "downlink_mb": float(downlink_A_mb + downlink_B_mb),\n',
            '                    "downlink_mb": float(client_downlink_A_mb + client_downlink_B_mb),\n',
            1,
        )
        roberta = roberta.replace(
            '                else:\n'
            '                    client.set_A_matrices(global_eval_A)\n'
            '                    if global_eval_B is not None:\n'
            '                        client.set_B_matrices(global_eval_B)\n'
            '                    else:\n'
            '                        client.set_B_matrices(client_personal_B_states.get(cid, server.get_global_B_matrices()))\n',
            '                else:\n'
            '                    if method in zero_padded_rank_methods:\n'
            '                        eval_A, eval_B = client.prepare_global_matrices_for_rank(\n'
            '                            global_eval_A, global_eval_B, target_rank=flora_client_ranks[cid]\n'
            '                        )\n'
            '                        client.receive_global_matrices(eval_A, eval_B, method=args.method)\n'
            '                    else:\n'
            '                        client.set_A_matrices(global_eval_A)\n'
            '                        if global_eval_B is not None:\n'
            '                            client.set_B_matrices(global_eval_B)\n'
            '                        else:\n'
            '                            client.set_B_matrices(client_personal_B_states.get(cid, server.get_global_B_matrices()))\n',
            1,
        )
        roberta = replace_once_checked(
            roberta,
            '            server.update_client_sensitivities(cid, b_sensitivity)\n\n'
            '            try:\n',
            '            server.update_client_sensitivities(cid, b_sensitivity)\n'
            '            client_personal_A_states[cid] = client.get_all_A_matrices()\n'
            '            client_personal_B_states[cid] = client.get_all_B_matrices()\n\n'
            '            try:\n',
            "roberta-save-personal-lora-state",
        )
        roberta = replace_once_checked(
            roberta,
            '                if eval_scope == "personalized":\n'
            '                    eval_A = client_personal_A_states.get(cid, global_eval_A)\n'
            '                    eval_B = client_personal_B_states.get(cid, server.get_global_B_matrices())\n'
            '                    client.set_A_matrices(eval_A)\n'
            '                    client.set_B_matrices(eval_B)\n'
            '                else:\n'
            '                    if method in zero_padded_rank_methods:\n'
            '                        eval_A, eval_B = client.prepare_global_matrices_for_rank(\n'
            '                            global_eval_A, global_eval_B, target_rank=flora_client_ranks[cid]\n'
            '                        )\n'
            '                        client.receive_global_matrices(eval_A, eval_B, method=args.method)\n'
            '                    else:\n'
            '                        client.set_A_matrices(global_eval_A)\n'
            '                        if global_eval_B is not None:\n'
            '                            client.set_B_matrices(global_eval_B)\n'
            '                        else:\n'
            '                            client.set_B_matrices(client_personal_B_states.get(cid, server.get_global_B_matrices()))\n',
            '                personal_A = client_personal_A_states.get(cid, client.get_all_A_matrices())\n'
            '                personal_B = client_personal_B_states.get(cid, client.get_all_B_matrices())\n'
            '                if is_fedharp_family and fedharp_matrix_mode == "a":\n'
            '                    eval_A_source, eval_B_source = global_eval_A, personal_B\n'
            '                elif is_fedharp_family and fedharp_matrix_mode == "b":\n'
            '                    eval_A_source, eval_B_source = personal_A, global_eval_B\n'
            '                elif is_fedharp_family:\n'
            '                    eval_A_source, eval_B_source = global_eval_A, global_eval_B\n'
            '                elif method in {"fedsalora", "fedsa-lora", "fedsa_lora"}:\n'
            '                    eval_A_source, eval_B_source = global_eval_A, personal_B\n'
            '                else:\n'
            '                    eval_A_source, eval_B_source = global_eval_A, global_eval_B\n'
            '                if method in zero_padded_rank_methods:\n'
            '                    eval_A, eval_B = client.prepare_global_matrices_for_rank(\n'
            '                        eval_A_source, eval_B_source, target_rank=flora_client_ranks[cid]\n'
            '                    )\n'
            '                else:\n'
            '                    eval_A, eval_B = eval_A_source, eval_B_source\n'
            '                client.set_A_matrices(eval_A)\n'
            '                if eval_B is not None:\n'
            '                    client.set_B_matrices(eval_B)\n',
            "roberta-method-consistent-personalized-eval",
        )
        roberta = replace_once_checked(
            roberta,
            '            global_eval_A = server.get_global_A_matrices()\n'
            '            global_eval_B = server.get_global_B_matrices() if needs_b else None\n'
            '            for cid in range(args.num_clients):\n',
            '            global_eval_A = server.get_global_A_matrices()\n'
            '            global_eval_B = server.get_global_B_matrices() if needs_b else None\n'
            '            if is_fedharp_family and fedharp_matrix_mode == "a":\n'
            '                eval_adapter_policy = "global_A+local_B"\n'
            '            elif is_fedharp_family and fedharp_matrix_mode == "b":\n'
            '                eval_adapter_policy = "local_A+global_B"\n'
            '            elif is_fedharp_family:\n'
            '                eval_adapter_policy = "global_A+global_B"\n'
            '            elif method in {"fedsalora", "fedsa-lora", "fedsa_lora"}:\n'
            '                eval_adapter_policy = "global_A+local_B"\n'
            '            else:\n'
            '                eval_adapter_policy = "global_aggregated_adapter"\n'
            '            print(f"Evaluation policy: adapter={eval_adapter_policy}, classifier=client_local")\n'
            '            try:\n'
            '                with open(log_path, "a", encoding="utf-8") as f:\n'
            '                    f.write(\n'
            '                        f"round={round_num}, type=eval_policy, "\n'
            '                        f"adapter={eval_adapter_policy}, classifier=client_local\\n"\n'
            '                    )\n'
            '            except Exception as e:\n'
            '                print(f"[warning] failed to write evaluation policy: {e}")\n'
            '            for cid in range(args.num_clients):\n',
            "roberta-eval-policy-log",
        )
        roberta = replace_once_checked(
            roberta,
            '    p.add_argument("--struct_to_data_rounds", type=int, default=20)\n\n',
            '    p.add_argument("--struct_to_data_rounds", type=int, default=20)\n'
            '    p.add_argument("--fedhello_rgd_rounds", type=int, default=2)\n'
            '    p.add_argument("--fedhello_fim_every", type=int, default=2)\n'
            '    p.add_argument("--fedhello_fim_proxy_size", type=int, default=50)\n'
            '    p.add_argument(\n'
            '        "--fedhello_geometry", type=str, default="bottleneck",\n'
            '        choices=["triangle", "inverted_triangle", "bottleneck", "uniform"],\n'
            '    )\n\n',
            "roberta-fedhello-cli",
        )
        roberta = replace_once_checked(
            roberta,
            'from utils import create_directories, get_device, set_seed\n',
            'from utils import create_directories, get_device, set_seed\n'
            'from mec.fedhello import empirical_fim_scores, reserve_proxy_indices, should_refresh_fim\n',
            "roberta-fedhello-imports",
        )
        roberta = replace_once_checked(
            roberta,
            '    encoded_train = _encode_texts(tokenizer, train_texts, train_labels.tolist(), max_length=args.max_length)\n'
            '    encoded_val = _encode_texts(tokenizer, val_texts, val_labels.tolist(), max_length=args.max_length)\n\n',
            '    encoded_train = _encode_texts(tokenizer, train_texts, train_labels.tolist(), max_length=args.max_length)\n'
            '    encoded_val = _encode_texts(tokenizer, val_texts, val_labels.tolist(), max_length=args.max_length)\n'
            '    fedhello_proxy_loader = None\n'
            '    evaluation_val_indices = list(range(len(encoded_val.labels)))\n'
            '    if str(args.method).lower() == "fedhello":\n'
            '        proxy_indices, evaluation_val_indices = reserve_proxy_indices(\n'
            '            num_examples=len(encoded_val.labels),\n'
            '            num_proxy=int(args.fedhello_fim_proxy_size),\n'
            '            seed=int(args.seed) + 314159,\n'
            '        )\n'
            '        proxy_ds = SST2ClientDataset(encoded_val, proxy_indices)\n'
            '        fedhello_proxy_loader = DataLoader(\n'
            '            proxy_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=True\n'
            '        )\n'
            '        print(\n'
            '            f"Fed-HeLLo proxy split: proxy={len(proxy_indices)}, "\n'
            '            f"validation={len(evaluation_val_indices)}, overlap=0"\n'
            '        )\n\n',
            "roberta-fedhello-proxy-split",
        )
        roberta = replace_once_checked(
            roberta,
            '            val_chunks = np.array_split(np.arange(len(encoded_val.labels)), int(args.num_clients))\n',
            '            val_chunks = np.array_split(np.asarray(evaluation_val_indices, dtype=np.int64), int(args.num_clients))\n',
            "roberta-fedhello-exclude-proxy-from-validation",
        )
        roberta = replace_once_checked(
            roberta,
            '    if hasattr(server, "client_layer_counts"):\n'
            '        for cid, cap in enumerate(client_capacities):\n'
            '            server.client_layer_counts[cid] = max(1, int(num_total_layers * float(cap)))\n\n'
            '    clients: List[TextFedHarpClient] = []\n',
            '    if hasattr(server, "client_layer_counts"):\n'
            '        for cid, cap in enumerate(client_capacities):\n'
            '            server.client_layer_counts[cid] = max(1, int(num_total_layers * float(cap)))\n'
            '    if method == "fedhello":\n'
            '        server.fedhello_rgd_rounds = max(0, int(args.fedhello_rgd_rounds))\n'
            '        server.fedhello_fim_every = max(1, int(args.fedhello_fim_every))\n'
            '        server.fedhello_geometry = str(args.fedhello_geometry)\n'
            '        print(\n'
            '            f"Fed-HeLLo-style HLA: geometry={server.fedhello_geometry}, "\n'
            '            f"rgd_rounds={server.fedhello_rgd_rounds}, "\n'
            '            f"fim_every={server.fedhello_fim_every}, "\n'
            '            f"fim_proxy_size={int(args.fedhello_fim_proxy_size)}"\n'
            '        )\n\n'
            '    clients: List[TextFedHarpClient] = []\n',
            "roberta-fedhello-server-config",
        )
        roberta = replace_once_checked(
            roberta,
            '        elif method == "flora":\n'
            '            allocation_map = {cid: set(layer_names) for cid in range(int(args.num_clients))}\n'
            '        else:\n'
            '            allocation_map = server.start_round()\n',
            '        elif method == "flora":\n'
            '            allocation_map = {cid: set(layer_names) for cid in range(int(args.num_clients))}\n'
            '        elif method == "fedhello":\n'
            '            refresh_fim = should_refresh_fim(\n'
            '                round_num=round_num,\n'
            '                rgd_rounds=int(args.fedhello_rgd_rounds),\n'
            '                fim_every=int(args.fedhello_fim_every),\n'
            '                has_scores=bool(server.fedhello_fim_scores),\n'
            '            )\n'
            '            fim_seen = 0\n'
            '            if refresh_fim:\n'
            '                if fedhello_proxy_loader is None:\n'
            '                    raise RuntimeError("Fed-HeLLo proxy loader was not initialized")\n'
            '                fim_scores, fim_seen = empirical_fim_scores(\n'
            '                    model=global_model,\n'
            '                    lora_layers=global_lora_layers,\n'
            '                    dataloader=fedhello_proxy_loader,\n'
            '                    device=device,\n'
            '                    max_samples=int(args.fedhello_fim_proxy_size),\n'
            '                )\n'
            '                server.set_fedhello_fim_scores(fim_scores, round_num=round_num)\n'
            '            allocation_map = server.start_round()\n'
            '            hla_phase = "fim" if bool(server.fedhello_fim_scores) else "rgd"\n'
            '            with open(log_path, "a", encoding="utf-8") as f:\n'
            '                f.write(\n'
            '                    f"round={round_num}, type=fedhello_hla, phase={hla_phase}, "\n'
            '                    f"geometry={server.fedhello_geometry}, refresh_fim={refresh_fim}, "\n'
            '                    f"fim_samples={fim_seen}, proxy_disjoint=True\\n"\n'
            '                )\n'
            '        else:\n'
            '            allocation_map = server.start_round()\n',
            "roberta-fedhello-fim-allocation",
        )
        roberta = replace_once_checked(
            roberta,
            '    p.add_argument(\n'
            '        "--fedhello_geometry", type=str, default="bottleneck",\n'
            '        choices=["triangle", "inverted_triangle", "bottleneck", "uniform"],\n'
            '    )\n\n',
            '    p.add_argument(\n'
            '        "--fedhello_geometry", type=str, default="bottleneck",\n'
            '        choices=["triangle", "inverted_triangle", "bottleneck", "uniform"],\n'
            '    )\n'
            '    p.add_argument("--mec", action="store_true")\n'
            '    p.add_argument("--mec_channel", type=str, default="correlated", choices=["constant", "rayleigh", "correlated"])\n'
            '    p.add_argument("--mec_channel_correlation", type=float, default=0.9)\n'
            '    p.add_argument("--mec_channel_gain_mean", type=float, default=1e-10)\n'
            '    p.add_argument("--mec_dropout_model", type=str, default="gilbert_elliott", choices=["none", "bernoulli", "gilbert_elliott"])\n'
            '    p.add_argument("--mec_dropout_rate", type=float, default=0.2)\n'
            '    p.add_argument("--mec_ge_good_to_bad", type=float, default=0.1)\n'
            '    p.add_argument("--mec_duration_hazard_rate", type=float, default=0.1)\n'
            '    p.add_argument("--mec_total_bw_mhz", type=float, default=5.0)\n'
            '    p.add_argument("--mec_power_max_dbm", type=float, default=23.0)\n'
            '    p.add_argument("--mec_deadline_s", type=float, default=10.0)\n'
            '    p.add_argument("--mec_output_dir", type=str, default=None)\n'
            '    p.add_argument("--mec_protocol_version", type=str, default="mec-v1.7.0-d044-fedhello-text-mec-smoke")\n'
            '    p.add_argument("--mec_code_version", type=str, default="mec-v1.7.0-d044-fedhello-text-mec-smoke")\n\n',
            "roberta-fedhello-mec-cli",
        )
        roberta = replace_once_checked(
            roberta,
            'from mec.fedhello import empirical_fim_scores, reserve_proxy_indices, should_refresh_fim\n',
            'from mec.fedhello import empirical_fim_scores, reserve_proxy_indices, should_refresh_fim\n'
            'from mec.fedhello_mec import FedHelloMecConfig, FedHelloMecController\n',
            "roberta-fedhello-mec-import",
        )
        roberta = replace_once_checked(
            roberta,
            '    best_accuracy = 0.0\n'
            '    layer_names = list(server.layer_names)\n',
            '    best_accuracy = 0.0\n'
            '    last_accuracy = 0.0\n'
            '    last_loss = 0.0\n'
            '    completed_mec_rounds = 0\n'
            '    layer_names = list(server.layer_names)\n',
            "roberta-fedhello-mec-result-state",
        )
        roberta = replace_once_checked(
            roberta,
            '    client_personal_A_states = {cid: clients[cid].get_all_A_matrices() for cid in range(args.num_clients)}\n'
            '    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}\n\n'
            '    for round_num in range(1, args.num_rounds + 1):\n',
            '    client_personal_A_states = {cid: clients[cid].get_all_A_matrices() for cid in range(args.num_clients)}\n'
            '    client_personal_B_states = {cid: clients[cid].get_all_B_matrices() for cid in range(args.num_clients)}\n\n'
            '    mec_controller = None\n'
            '    mec_output_dir = None\n'
            '    mec_metrics_path = None\n'
            '    mec_run_log_path = None\n'
            '    if bool(args.mec):\n'
            '        _mec_alias = {"fedsa-lora": "fedsalora", "fedsa_lora": "fedsalora", "FedSA-LoRA": "fedsalora", "ffalora": "ffa-lora", "ffa_lora": "ffa-lora"}\n'
            '        method_mec_base = _mec_alias.get(method, method)\n'
            '        _mec_text_methods = {"fedharp", "fedhello", "fedra", "flora", "fedsalora", "ffa-lora"}\n'
            '        if method_mec_base not in _mec_text_methods:\n'
            '            raise ValueError(f"Embedded RoBERTa MEC supports the six equal-fixed methods, got {method}")\n'
            '        _mec_matrix_mode = {"fedharp": "a", "fedra": "a", "fedsalora": "a", "ffa-lora": "b", "flora": "ab", "fedhello": "ab"}\n'
            '        if int(args.eval_every) != 1:\n'
            '            raise ValueError("Fed-HeLLo MEC audit requires --eval_every 1")\n'
            '        mec_output_dir = str(args.mec_output_dir or args.checkpoint_dir)\n'
            '        os.makedirs(mec_output_dir, exist_ok=True)\n'
            '        mec_metrics_path = os.path.join(mec_output_dir, "metrics.jsonl")\n'
            '        mec_run_log_path = os.path.join(mec_output_dir, "run.log")\n'
            '        for protected_name in ("metrics.jsonl", "config.json", "DONE", "FAILED"):\n'
            '            protected_path = os.path.join(mec_output_dir, protected_name)\n'
            '            if os.path.exists(protected_path):\n'
            '                raise FileExistsError(f"Refusing to overwrite existing MEC artifact: {protected_path}")\n'
            '        mec_cfg = FedHelloMecConfig(\n'
            '            num_clients=int(args.num_clients), num_rounds=int(args.num_rounds), seed=int(args.seed),\n'
            '            heterogeneity_type=str(args.heterogeneity_type), allocation_ratio=float(args.allocation_ratio),\n'
            '            channel_model=str(args.mec_channel), channel_correlation=float(args.mec_channel_correlation),\n'
            '            channel_gain_mean=float(args.mec_channel_gain_mean), dropout_model=str(args.mec_dropout_model),\n'
            '            dropout_rate=float(args.mec_dropout_rate), ge_good_to_bad=float(args.mec_ge_good_to_bad),\n'
            '            duration_hazard_rate=float(args.mec_duration_hazard_rate),\n'
            '            total_bandwidth_mhz=float(args.mec_total_bw_mhz), power_max_dbm=float(args.mec_power_max_dbm),\n'
            '            deadline_s=float(args.mec_deadline_s),\n'
            '        )\n'
            '        mec_controller = FedHelloMecController(mec_cfg)\n'
            '        mec_config_record = dict(vars(args))\n'
            '        mec_config_record.update({\n'
            '            "task": "roberta", "mec_method": method_mec_base + "_equal_fixed",\n'
            '            "network_policy": "equal_fixed", "matrix_mode": _mec_matrix_mode[method_mec_base],\n'
            '            "trace_hash": mec_controller.trace_hash,\n'
            '            "payload_source": "actual_delta_A_plus_delta_B_tensor_bytes",\n'
            '            "arrival_gate_affects_aggregation": True,\n'
            '            "energy_model": "model-based simulated",\n'
            '            "cli": list(argv or []),\n'
            '        })\n'
            '        with open(os.path.join(mec_output_dir, "config.json"), "w", encoding="utf-8") as f:\n'
            '            json.dump(mec_config_record, f, ensure_ascii=False, indent=2, sort_keys=True)\n'
            '        open(mec_metrics_path, "w", encoding="utf-8").close()\n'
            '        with open(mec_run_log_path, "w", encoding="utf-8") as f:\n'
            '            f.write(f"START method={method_mec_base} network=equal_fixed trace_hash={mec_controller.trace_hash}\\n")\n\n'
            '    for round_num in range(1, args.num_rounds + 1):\n',
            "roberta-fedhello-mec-init",
        )
        roberta = replace_once_checked(
            roberta,
            '        if needs_b:\n'
            '            server.aggregate_matrices(\n'
            '                client_updates_A=client_updates,\n'
            '                client_updates_B=client_updates_B,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n',
            '        mec_round = None\n'
            '        if mec_controller is not None:\n'
            '            payload_by_client = {\n'
            '                int(cid): int(_delta_A_nbytes(client_updates[int(cid)]))\n'
            '                + int(_delta_B_nbytes(client_updates_B.get(int(cid), {})))\n'
            '                for cid in selected_clients\n'
            '            }\n'
            '            mec_round = mec_controller.run_round(\n'
            '                round_num=round_num,\n'
            '                selected_client_ids=[int(cid) for cid in selected_clients],\n'
            '                payload_bytes=payload_by_client,\n'
            '                active_blocks={int(cid): len(allocation_map[int(cid)]) for cid in selected_clients},\n'
            '                train_samples={int(cid): int(client_sample_counts[int(cid)]) for cid in selected_clients},\n'
            '                wall_clock_s=float(time.time() - round_start_time),\n'
            '                download_bytes=float((downlink_A_mb + downlink_B_mb) * 1024.0 * 1024.0),\n'
            '            )\n'
            '            arrived_ids = set(int(cid) for cid in mec_round.arrived_client_ids)\n'
            '            client_updates = {cid: value for cid, value in client_updates.items() if int(cid) in arrived_ids}\n'
            '            client_updates_B = {cid: value for cid, value in client_updates_B.items() if int(cid) in arrived_ids}\n'
            '            if len(client_updates) == 0:\n'
            '                raise RuntimeError(f"round {round_num}: no {method_mec_base} update arrived; refusing empty aggregation")\n'
            '            with open(mec_run_log_path, "a", encoding="utf-8") as f:\n'
            '                f.write(\n'
            '                    f"round={round_num} scheduled={mec_round.diagnostics[\'scheduled_clients\']} "\n'
            '                    f"arrived={mec_round.diagnostics[\'arrived_clients\']} "\n'
            '                    f"base_drop={mec_round.diagnostics[\'base_dropout_clients\']} "\n'
            '                    f"duration_drop={mec_round.diagnostics[\'duration_dropout_clients\']} "\n'
            '                    f"deadline_miss={mec_round.diagnostics[\'deadline_miss_clients\']}\\n"\n'
            '                )\n\n'
            '        if needs_b:\n'
            '            server.aggregate_matrices(\n'
            '                client_updates_A=client_updates,\n'
            '                client_updates_B=client_updates_B,\n'
            '                client_sample_counts=client_sample_counts,\n'
            '                allocation_map=allocation_map,\n'
            '            )\n',
            "roberta-fedhello-mec-arrival-gate",
        )
        roberta = replace_once_checked(
            roberta,
            '            if global_acc > best_accuracy:\n'
            '                best_accuracy = global_acc\n\n',
            '            last_accuracy = float(global_acc)\n'
            '            last_loss = float(global_loss)\n'
            '            if global_acc > best_accuracy:\n'
            '                best_accuracy = global_acc\n'
            '            if mec_controller is not None:\n'
            '                if mec_round is None:\n'
            '                    raise RuntimeError("missing MEC round result")\n'
            '                summary = mec_controller.metrics.round_summary(\n'
            '                    round_num=round_num,\n'
            '                    accuracy_personalized=float(global_acc) / 100.0,\n'
            '                    accuracy_global=float(global_acc) / 100.0,\n'
            '                )\n'
            '                summary.update(mec_round.diagnostics)\n'
            '                summary.update({\n'
            '                    "dataset": str(args.dataset), "method": method_mec_base,\n'
            '                    "accuracy_global_is_proxy": True,\n'
            '                    "metric_split": "glue_validation",\n'
            '                    "selected_clients": [int(cid) for cid in selected_clients],\n'
            '                    "arrived_client_ids": [int(cid) for cid in mec_round.arrived_client_ids],\n'
            '                    "client_system_results": mec_round.client_results,\n'
            '                })\n'
            '                with open(mec_metrics_path, "a", encoding="utf-8") as f:\n'
            '                    f.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\\n")\n'
            '                completed_mec_rounds = int(round_num)\n\n',
            "roberta-fedhello-mec-round-summary",
        )
        roberta = replace_once_checked(
            roberta,
            '    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")\n',
            '    if mec_controller is not None:\n'
            '        stopping = {\n'
            '            "selection_split": "glue_validation", "test_used_for_selection": False,\n'
            '            "completed_rounds": int(completed_mec_rounds),\n'
            '            "maximum_rounds": int(args.num_rounds), "stopped_early": False,\n'
            '            "best_validation_accuracy": float(best_accuracy) / 100.0,\n'
            '        }\n'
            '        final_metrics = {\n'
            '            "evaluation_scope": "personalized_glue_validation",\n'
            '            "classifier_policy": "client_local", "dataset": str(args.dataset),\n'
            '            "completed_rounds": int(completed_mec_rounds),\n'
            '            "accuracy": float(last_accuracy) / 100.0, "loss": float(last_loss),\n'
            '            "energy_model": "model-based simulated",\n'
            '        }\n'
            '        with open(os.path.join(mec_output_dir, "STOPPING.json"), "w", encoding="utf-8") as f:\n'
            '            json.dump(stopping, f, ensure_ascii=False, indent=2, sort_keys=True)\n'
            '        with open(os.path.join(mec_output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:\n'
            '            json.dump(final_metrics, f, ensure_ascii=False, indent=2, sort_keys=True)\n'
            '        with open(os.path.join(mec_output_dir, "DONE"), "w", encoding="utf-8") as f:\n'
            '            f.write(f"completed rounds={completed_mec_rounds} rows={completed_mec_rounds} stopped_early=False\\n")\n'
            '        with open(mec_run_log_path, "a", encoding="utf-8") as f:\n'
            '            f.write(f"DONE completed_rounds={completed_mec_rounds}\\n")\n'
            '    print(f"Best local-weighted accuracy (personalized): {best_accuracy:.2f}%")\n',
            "roberta-fedhello-mec-finalize",
        )
        # ------------------------------------------------------------------
        # D052: FedSONATA-CORA text port (RIFT-LEF + CORA predictor/corrector).
        # ----------------------------------------------------------------------
        roberta = replace_once_checked(
            roberta,
            'from mec.fedhello_mec import FedHelloMecConfig, FedHelloMecController\n',
            'from mec.fedhello_mec import FedHelloMecConfig, FedHelloMecController\n'
            'from mec.fedsonata_text_mec import FedSonataTextMecConfig, FedSonataTextMecController\n',
            "roberta-fedsonata-import",
        )
        roberta = replace_once_checked(
            roberta,
            '    p.add_argument("--mec_code_version", type=str, default="mec-v1.7.0-d044-fedhello-text-mec-smoke")\n\n',
            '    p.add_argument("--mec_code_version", type=str, default="mec-v1.7.0-d044-fedhello-text-mec-smoke")\n'
            '    p.add_argument("--mec_text_factor_policy", type=str, default="equal_fixed", choices=["equal_fixed", "cora", "fedsonata_cora"])\n'
            '    p.add_argument("--rift_budget_ratio", type=float, default=0.35)\n'
            '    p.add_argument("--rift_ef_decay", type=float, default=0.5)\n'
            '    p.add_argument("--mec_d_min", type=float, default=0.5)\n'
            '    p.add_argument("--mec_d_max", type=float, default=10.0)\n'
            '    p.add_argument("--mec_deadline_grid_size", type=int, default=24)\n'
            '    p.add_argument("--mec_lambda_e", type=float, default=0.1)\n'
            '    p.add_argument("--mec_lambda_u", type=float, default=1.0)\n'
            '    p.add_argument("--mec_omega_net", type=float, default=1.0)\n'
            '    p.add_argument("--mec_reliability_beta", type=float, default=0.8)\n\n',
            "roberta-fedsonata-cli",
        )
        roberta = replace_once_checked(
            roberta,
            '        mec_cfg = FedHelloMecConfig(\n'
            '            num_clients=int(args.num_clients), num_rounds=int(args.num_rounds), seed=int(args.seed),\n'
            '            heterogeneity_type=str(args.heterogeneity_type), allocation_ratio=float(args.allocation_ratio),\n'
            '            channel_model=str(args.mec_channel), channel_correlation=float(args.mec_channel_correlation),\n'
            '            channel_gain_mean=float(args.mec_channel_gain_mean), dropout_model=str(args.mec_dropout_model),\n'
            '            dropout_rate=float(args.mec_dropout_rate), ge_good_to_bad=float(args.mec_ge_good_to_bad),\n'
            '            duration_hazard_rate=float(args.mec_duration_hazard_rate),\n'
            '            total_bandwidth_mhz=float(args.mec_total_bw_mhz), power_max_dbm=float(args.mec_power_max_dbm),\n'
            '            deadline_s=float(args.mec_deadline_s),\n'
            '        )\n'
            '        mec_controller = FedHelloMecController(mec_cfg)\n',
            '        _factor_policy = str(getattr(args, "mec_text_factor_policy", "equal_fixed")).lower()\n'
            '        if _factor_policy not in {"equal_fixed", "cora", "fedsonata_cora"}:\n'
            '            raise ValueError(f"unknown mec_text_factor_policy: {_factor_policy}")\n'
            '        if _factor_policy == "fedsonata_cora" and fedharp_matrix_mode != "ab":\n'
            '            raise ValueError("fedsonata_cora text policy requires --fedharp_matrix_mode ab")\n'
            '        if _factor_policy == "equal_fixed":\n'
            '            mec_cfg = FedHelloMecConfig(\n'
            '                num_clients=int(args.num_clients), num_rounds=int(args.num_rounds), seed=int(args.seed),\n'
            '                heterogeneity_type=str(args.heterogeneity_type), allocation_ratio=float(args.allocation_ratio),\n'
            '                channel_model=str(args.mec_channel), channel_correlation=float(args.mec_channel_correlation),\n'
            '                channel_gain_mean=float(args.mec_channel_gain_mean), dropout_model=str(args.mec_dropout_model),\n'
            '                dropout_rate=float(args.mec_dropout_rate), ge_good_to_bad=float(args.mec_ge_good_to_bad),\n'
            '                duration_hazard_rate=float(args.mec_duration_hazard_rate),\n'
            '                total_bandwidth_mhz=float(args.mec_total_bw_mhz), power_max_dbm=float(args.mec_power_max_dbm),\n'
            '                deadline_s=float(args.mec_deadline_s),\n'
            '            )\n'
            '            mec_controller = FedHelloMecController(mec_cfg)\n'
            '        else:\n'
            '            mec_cfg = FedSonataTextMecConfig(\n'
            '                num_clients=int(args.num_clients), num_rounds=int(args.num_rounds), seed=int(args.seed),\n'
            '                factor_policy=_factor_policy,\n'
            '                heterogeneity_type=str(args.heterogeneity_type), allocation_ratio=float(args.allocation_ratio),\n'
            '                channel_model=str(args.mec_channel), channel_correlation=float(args.mec_channel_correlation),\n'
            '                channel_gain_mean=float(args.mec_channel_gain_mean), dropout_model=str(args.mec_dropout_model),\n'
            '                dropout_rate=float(args.mec_dropout_rate), ge_good_to_bad=float(args.mec_ge_good_to_bad),\n'
            '                duration_hazard_rate=float(args.mec_duration_hazard_rate),\n'
            '                total_bandwidth_mhz=float(args.mec_total_bw_mhz), power_max_dbm=float(args.mec_power_max_dbm),\n'
            '                d_min_s=float(getattr(args, "mec_d_min", 0.5)), d_max_s=float(getattr(args, "mec_d_max", 10.0)),\n'
            '                deadline_grid_size=int(getattr(args, "mec_deadline_grid_size", 24)),\n'
            '                lambda_e=float(getattr(args, "mec_lambda_e", 0.1)), lambda_u=float(getattr(args, "mec_lambda_u", 1.0)),\n'
            '                omega_net=float(getattr(args, "mec_omega_net", 1.0)), reliability_beta=float(getattr(args, "mec_reliability_beta", 0.8)),\n'
            '                rift_budget_ratio=float(getattr(args, "rift_budget_ratio", 0.35)),\n'
            '                rift_ef_decay=float(getattr(args, "rift_ef_decay", 0.5)),\n'
            '            )\n'
            '            mec_controller = FedSonataTextMecController(mec_cfg)\n',
            "roberta-fedsonata-controller-dispatch",
        )
        roberta = replace_once_checked(
            roberta,
            '        mec_config_record.update({\n'
            '            "task": "roberta", "mec_method": method_mec_base + "_equal_fixed",\n'
            '            "network_policy": "equal_fixed", "matrix_mode": _mec_matrix_mode[method_mec_base],\n'
            '            "trace_hash": mec_controller.trace_hash,\n',
            '        _recorded_mec_method = (\n'
            '            "proposed_joint_rift_ef_cora_text" if _factor_policy == "fedsonata_cora"\n'
            '            else "proposed_joint_cora_text" if _factor_policy == "cora"\n'
            '            else method_mec_base + "_equal_fixed"\n'
            '        )\n'
            '        _recorded_network_policy = "equal_fixed" if _factor_policy == "equal_fixed" else "convex_joint_text"\n'
            '        _recorded_matrix_mode = (\n'
            '            fedharp_matrix_mode if _factor_policy in {"cora", "fedsonata_cora"}\n'
            '            else _mec_matrix_mode[method_mec_base]\n'
            '        )\n'
            '        mec_config_record.update({\n'
            '            "task": "roberta", "mec_method": _recorded_mec_method,\n'
            '            "network_policy": _recorded_network_policy, "matrix_mode": _recorded_matrix_mode,\n'
            '            "factor_policy": _factor_policy,\n'
            '            "trace_hash": mec_controller.trace_hash,\n',
            "roberta-fedsonata-config-record",
        )
        roberta = replace_once_checked(
            roberta,
            '            f.write(f"START method={method_mec_base} network=equal_fixed trace_hash={mec_controller.trace_hash}\\n")\n',
            '            f.write(f"START method={method_mec_base} policy={_factor_policy} network={_recorded_network_policy} trace_hash={mec_controller.trace_hash}\\n")\n',
            "roberta-fedsonata-start-line",
        )
        roberta = replace_once_checked(
            roberta,
            '        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}\n'
            '        client_updates_B: Dict[int, Dict[str, torch.Tensor]] = {}\n'
            '        client_metrics_ab = []\n\n'
            '        for client_id in selected_clients:\n',
            '        client_updates: Dict[int, Dict[str, torch.Tensor]] = {}\n'
            '        client_updates_B: Dict[int, Dict[str, torch.Tensor]] = {}\n'
            '        client_metrics_ab = []\n'
            '        rift_round_records: Dict[int, object] = {}\n'
            '        if isinstance(mec_controller, FedSonataTextMecController):\n'
            '            _pre_full_ab = {}\n'
            '            for _client_id in selected_clients:\n'
            '                _pcid = int(_client_id)\n'
            '                _players = allocation_map.get(_pcid, set())\n'
            '                if global_B is not None:\n'
            '                    _pre_full_ab[_pcid] = int(sum(\n'
            '                        _A_matrices_nbytes({name: global_A[name]})\n'
            '                        + _A_matrices_nbytes({name: global_B[name]})\n'
            '                        for name in _players\n'
            '                    ))\n'
            '                else:\n'
            '                    _pre_full_ab[_pcid] = int(sum(\n'
            '                        _A_matrices_nbytes({name: global_A[name]})\n'
            '                        for name in _players\n'
            '                    ))\n'
            '            mec_controller.pre_round(\n'
            '                round_num=round_num,\n'
            '                selected_client_ids=[int(cid) for cid in selected_clients],\n'
            '                full_ab_bytes=_pre_full_ab,\n'
            '                allocated_layers={int(cid): allocation_map.get(int(cid), set()) for cid in selected_clients},\n'
            '                train_samples={int(cid): int(client_sample_counts.get(int(cid), 0)) for cid in selected_clients},\n'
            '                layer_names=layer_names,\n'
            '            )\n\n'
            '        for client_id in selected_clients:\n',
            "roberta-fedsonata-pre-round",
        )
        roberta = replace_once_checked(
            roberta,
            '            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)\n',
            '            if isinstance(mec_controller, FedSonataTextMecController):\n'
            '                _a_after = client.get_all_A_matrices()\n'
            '                _b_after = client.get_all_B_matrices()\n'
            '                _sel_A, _sel_B, _rift_record = mec_controller.select_factors(\n'
            '                    round_num=round_num, client_id=cid,\n'
            '                    a_after=_a_after, b_after=_b_after,\n'
            '                    delta_a=delta_A, delta_b=(delta_B or {}),\n'
            '                    allocated_layers=set(allocated_layers),\n'
            '                )\n'
            '                delta_A = _sel_A\n'
            '                delta_B = _sel_B\n'
            '                if _rift_record is not None:\n'
            '                    rift_round_records[cid] = _rift_record\n'
            '            uplink_A_mb = _delta_A_nbytes(delta_A) / (1024.0 * 1024.0)\n',
            "roberta-fedsonata-select-factors",
        )
        roberta = replace_once_checked(
            roberta,
            '            mec_round = mec_controller.run_round(\n'
            '                round_num=round_num,\n'
            '                selected_client_ids=[int(cid) for cid in selected_clients],\n'
            '                payload_bytes=payload_by_client,\n'
            '                active_blocks={int(cid): len(allocation_map[int(cid)]) for cid in selected_clients},\n'
            '                train_samples={int(cid): int(client_sample_counts[int(cid)]) for cid in selected_clients},\n'
            '                wall_clock_s=float(time.time() - round_start_time),\n'
            '                download_bytes=float((downlink_A_mb + downlink_B_mb) * 1024.0 * 1024.0),\n'
            '            )\n',
            '            if isinstance(mec_controller, FedSonataTextMecController):\n'
            '                mec_round = mec_controller.run_round(\n'
            '                    round_num=round_num,\n'
            '                    selected_client_ids=[int(cid) for cid in selected_clients],\n'
            '                    payload_bytes=payload_by_client,\n'
            '                    active_blocks={int(cid): len(allocation_map[int(cid)]) for cid in selected_clients},\n'
            '                    train_samples={int(cid): int(client_sample_counts[int(cid)]) for cid in selected_clients},\n'
            '                    wall_clock_s=float(time.time() - round_start_time),\n'
            '                    download_bytes=float((downlink_A_mb + downlink_B_mb) * 1024.0 * 1024.0),\n'
            '                    all_layer_names=layer_names,\n'
            '                    rift_records=rift_round_records,\n'
            '                )\n'
            '            else:\n'
            '                mec_round = mec_controller.run_round(\n'
            '                    round_num=round_num,\n'
            '                    selected_client_ids=[int(cid) for cid in selected_clients],\n'
            '                    payload_bytes=payload_by_client,\n'
            '                    active_blocks={int(cid): len(allocation_map[int(cid)]) for cid in selected_clients},\n'
            '                    train_samples={int(cid): int(client_sample_counts[int(cid)]) for cid in selected_clients},\n'
            '                    wall_clock_s=float(time.time() - round_start_time),\n'
            '                    download_bytes=float((downlink_A_mb + downlink_B_mb) * 1024.0 * 1024.0),\n'
            '                )\n',
            "roberta-fedsonata-run-round",
        )
        if 'eval_A = client_personal_A_states.get(cid, global_eval_A)' in roberta:
            raise RuntimeError("stale personalized-A restore remains in embedded RoBERTa source")
        compile(roberta, "<embedded-roberta>", "exec")
        _EMBEDDED_TASK_SOURCES["roberta"] = roberta

    qwen = _EMBEDDED_TASK_SOURCES.get("qwen")
    if qwen:
        # Gemma 4 wraps multimodal projection layers in Gemma4ClippableLinear.
        # PEFT cannot inject into that wrapper, so restrict LoRA targets to the
        # language tower when the model exposes the Gemma 4 architecture.
        qwen = qwen.replace(
            '    target_modules = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]\n'
            '    peft_config = LoraConfig(\n',
            '    target_modules = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]\n'
            '    if "gemma4" in str(getattr(getattr(model, "config", None), "model_type", "")).lower():\n'
            '        target_modules = [\n'
            '            n for n, m in model.named_modules()\n'
            '            if (n.startswith("model.language_model.") or n.startswith("language_model."))\n'
            '            and any(n.endswith("." + t) for t in target_modules)\n'
            '            and isinstance(m, torch.nn.Linear)\n'
            '        ]\n'
            '        if not target_modules:\n'
            '            raise ValueError("No compatible Gemma 4 language-model LoRA targets found")\n'
            '    peft_config = LoraConfig(\n',
            1,
        )
        qwen = qwen.replace(
            '"flexlora", "flex-lora", "flex_lora"]',
            '"flexlora", "flex-lora", "flex_lora", '
            '"fedhello", "flora", "fedsalora", "fedsa-lora", "fedsa_lora", "FedSA-LoRA", '
            '"fedit", "fed-it", "fed_it"]',
        )
        qwen = qwen.replace(
            "Causal LM + GSM8K: FedHARP/FedRA/FFA-LoRA/FlexLoRA federated finetuning",
            "Causal LM + GSM8K: FedHARP and LoRA baselines federated finetuning",
        )
        qwen = qwen.replace(
            '    upload_B = method_l == "fedra" or is_ffa_lora\n',
            '    upload_B = method_l in {"fedra", "fedit", "fed-it", "fed_it"} or is_ffa_lora\n',
        )
        qwen = qwen.replace(
            '    upload_B = method_l in {"fedra", "fedit", "fed-it", "fed_it"} or is_ffa_lora\n',
            '    is_flexlora = method_l in {"flexlora", "flex-lora", "flex_lora"}\n'
            '    is_flora = method_l == "flora"\n'
            '    upload_B = method_l in {"fedra", "fedhello", "fedit", "fed-it", "fed_it"} or is_ffa_lora or is_flexlora or is_flora\n',
        )
        qwen = qwen.replace(
            '    if not is_ffa_lora:\n'
            '        for name in allocated_layers:\n'
            '            if name in A_before and name in A_after:\n'
            '                delta_A[name] = A_after[name] - A_before[name]\n'
            '\n'
            '    if upload_B and B_before is not None and B_after is not None:\n'
            '        delta_B = {}\n'
            '        for name in allocated_layers:\n'
            '            if name in B_before and name in B_after:\n'
            '                delta_B[name] = B_after[name] - B_before[name]\n',
            '    if is_flexlora or is_flora:\n'
            '        delta_B = {}\n'
            '        for name in allocated_layers:\n'
            '            if name in A_after and name in B_after:\n'
            '                delta_A[name] = A_after[name]\n'
            '                delta_B[name] = B_after[name]\n'
            '    else:\n'
            '        if not is_ffa_lora:\n'
            '            for name in allocated_layers:\n'
            '                if name in A_before and name in A_after:\n'
            '                    delta_A[name] = A_after[name] - A_before[name]\n'
            '\n'
            '        if upload_B and B_before is not None and B_after is not None:\n'
            '            delta_B = {}\n'
            '            for name in allocated_layers:\n'
            '                if name in B_before and name in B_after:\n'
            '                    delta_B[name] = B_after[name] - B_before[name]\n',
        )
        qwen = qwen.replace(
            "import argparse\n",
            "import argparse\n"
            "if not hasattr(argparse, \"BooleanOptionalAction\"):\n"
            "    class _BooleanOptionalAction(argparse.Action):\n"
            "        def __init__(self, option_strings, dest, default=None, **kwargs):\n"
            "            expanded = []\n"
            "            for opt in option_strings:\n"
            "                expanded.append(opt)\n"
            "                if opt.startswith(\"--\"):\n"
            "                    expanded.append(\"--no-\" + opt[2:])\n"
            "            super().__init__(expanded, dest, nargs=0, default=default, **kwargs)\n"
            "        def __call__(self, parser, namespace, values, option_string=None):\n"
            "            setattr(namespace, self.dest, not str(option_string).startswith(\"--no-\"))\n"
            "    argparse.BooleanOptionalAction = _BooleanOptionalAction\n",
            1,
        )
        qwen = replace_once_checked(
            qwen,
            'from dataclasses import dataclass\n',
            'from dataclasses import dataclass\n'
            'from mec.fedhello_mec import FedHelloMecConfig, FedHelloMecController\n',
            "gsm8k-mec-import",
        )
        qwen = replace_once_checked(
            qwen,
            '    p.add_argument("--metrics_file", type=str, default=None)\n'
            '    return p.parse_args(args=argv)\n',
            '    p.add_argument("--metrics_file", type=str, default=None)\n'
            '    p.add_argument("--mec", action=argparse.BooleanOptionalAction, default=False)\n'
            '    p.add_argument("--mec_channel", type=str, default="correlated", choices=["constant", "rayleigh", "correlated"])\n'
            '    p.add_argument("--mec_channel_correlation", type=float, default=0.9)\n'
            '    p.add_argument("--mec_channel_gain_mean", type=float, default=1e-10)\n'
            '    p.add_argument("--mec_dropout_model", type=str, default="gilbert_elliott", choices=["none", "bernoulli", "gilbert_elliott"])\n'
            '    p.add_argument("--mec_dropout_rate", type=float, default=0.2)\n'
            '    p.add_argument("--mec_ge_good_to_bad", type=float, default=0.1)\n'
            '    p.add_argument("--mec_duration_hazard_rate", type=float, default=0.1)\n'
            '    p.add_argument("--mec_total_bw_mhz", type=float, default=5.0)\n'
            '    p.add_argument("--mec_power_max_dbm", type=float, default=23.0)\n'
            '    p.add_argument("--mec_deadline_s", type=float, default=10.0)\n'
            '    p.add_argument("--mec_output_dir", type=str, default=None)\n'
            '    p.add_argument("--mec_protocol_version", type=str, default="mec-v1.7.0-d049-gsm8k")\n'
            '    p.add_argument("--mec_code_version", type=str, default="mec-v1.7.0-d049-gsm8k")\n'
            '    return p.parse_args(args=argv)\n',
            "gsm8k-mec-cli",
        )
        qwen = qwen.replace(
            'p = argparse.ArgumentParser(description="Qwen3-4B + GSM8K 中文: FedHARP/FedRA 联邦微调")',
            'p = argparse.ArgumentParser(description="Causal LM + GSM8K: FedHARP and LoRA baselines federated finetuning")',
        )
        qwen = qwen.replace(
            'p.add_argument("--model_path", type=str, default="./Qwen3-4B_test/Qwen/Qwen3-4B copy")',
            'p.add_argument("--model_path", type=str, default="./base_models/gemma-4-12B-it/")',
        )
        qwen = replace_once_checked(
            qwen,
            '    p.add_argument("--lora_qv_only", action=argparse.BooleanOptionalAction, default=True)\n',
            '    p.add_argument("--lora_qv_only", action=argparse.BooleanOptionalAction, default=True)\n'
            '    p.add_argument("--flora_rank_policy", type=str, default="capacity", choices=["capacity", "uniform"])\n'
            '    p.add_argument("--flora_min_rank", type=int, default=1)\n'
            '    p.add_argument("--flora_local_ranks", type=str, default=None)\n',
            "gsm8k-flora-rank-args",
        )
        qwen = replace_once_checked(
            qwen,
            '    p.add_argument("--eval_every_rounds", type=int, default=2)\n',
            '    p.add_argument("--eval_every_rounds", type=int, default=2)\n'
            '    p.add_argument("--zero_shot_eval_round", type=int, default=None)\n',
            "gsm8k-zero-shot-matched-round-arg",
        )
        qwen = replace_once_checked(
            qwen,
            '    p.add_argument("--zero_shot_eval_round", type=int, default=None)\n',
            '    p.add_argument("--zero_shot_eval_round", type=int, default=None)\n'
            '    p.add_argument("--zero_shot_report_subgroups", action=argparse.BooleanOptionalAction, default=True)\n',
            "gsm8k-zero-shot-subgroup-arg",
        )
        qwen = replace_once_checked(
            qwen,
            'def _setup_layer_trainability(lora_layers: Dict[str, object], allocated_layers: Set[str]):\n',
            'def _build_flora_client_ranks(\n'
            '    *, client_capacities: List[float], max_rank: int, policy: str,\n'
            '    min_rank: int, explicit_ranks: Optional[str],\n'
            ') -> List[int]:\n'
            '    num_clients = len(client_capacities)\n'
            '    max_rank = max(1, int(max_rank))\n'
            '    min_rank = max(1, min(int(min_rank), max_rank))\n'
            '    if explicit_ranks is not None and str(explicit_ranks).strip():\n'
            '        ranks = [int(x.strip()) for x in str(explicit_ranks).split(",") if x.strip()]\n'
            '        if len(ranks) == 1:\n'
            '            ranks = ranks * num_clients\n'
            '        if len(ranks) != num_clients:\n'
            '            raise ValueError(f"flora_local_ranks expects 1 or {num_clients} values, got {len(ranks)}")\n'
            '        return [max(min_rank, min(max_rank, int(rank))) for rank in ranks]\n'
            '    if str(policy).lower().strip() == "uniform":\n'
            '        return [max_rank] * num_clients\n'
            '    return [\n'
            '        max(min_rank, min(max_rank, int(round(max_rank * float(capacity)))))\n'
            '        for capacity in client_capacities\n'
            '    ]\n\n\n'
            'def _setup_layer_trainability(lora_layers: Dict[str, object], allocated_layers: Set[str]):\n',
            "gsm8k-flora-rank-builder",
        )
        qwen = qwen.replace(
            'p.add_argument("--chat_format", type=str, default="chatml", choices=["chatml", "raw"])',
            'p.add_argument("--chat_format", type=str, default="auto", choices=["chatml", "template", "auto", "raw"])',
        )
        qwen = qwen.replace(
            '    p.add_argument("--max_train_samples", type=int, default=None)\n'
            '    p.add_argument("--train_on_inputs", action=argparse.BooleanOptionalAction, default=False)\n',
            '    p.add_argument("--max_train_samples", type=int, default=None)\n'
            '    p.add_argument("--max_client_steps", type=int, default=None)\n'
            '    p.add_argument("--train_on_inputs", action=argparse.BooleanOptionalAction, default=False)\n',
        )
        qwen = qwen.replace(
            '    tokens += nl_tokens + [int(im_start_id)] + assistant_part\n'
            '    return tokens\n\n\n',
            '    tokens += nl_tokens + [int(im_start_id)] + assistant_part\n'
            '    return tokens\n\n\n'
            'def _coerce_token_id_list(ids, tokenizer=None) -> List[int]:\n'
            '    if isinstance(ids, dict):\n'
            '        ids = ids.get("input_ids", ids.get("input_ids_0", ids))\n'
            '    if hasattr(ids, "input_ids"):\n'
            '        ids = getattr(ids, "input_ids")\n'
            '    if isinstance(ids, str):\n'
            '        if tokenizer is None:\n'
            '            raise ValueError("token string returned but tokenizer is unavailable")\n'
            '        ids = tokenizer(ids, add_special_tokens=False, truncation=False)["input_ids"]\n'
            '    if hasattr(ids, "tolist"):\n'
            '        ids = ids.tolist()\n'
            '    if isinstance(ids, tuple):\n'
            '        ids = list(ids)\n'
            '    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):\n'
            '        ids = list(ids[0])\n'
            '    out: List[int] = []\n'
            '    for x in list(ids):\n'
            '        if isinstance(x, (list, tuple)):\n'
            '            out.extend(int(y) for y in x)\n'
            '        else:\n'
            '            out.append(int(x))\n'
            '    return out\n\n\n'
            'def _make_template_prompt_tokens(*, tokenizer, query: str, system: str) -> List[int]:\n'
            '    if not hasattr(tokenizer, "apply_chat_template"):\n'
            '        raise ValueError("tokenizer has no apply_chat_template")\n'
            '    messages = []\n'
            '    if str(system).strip() != "":\n'
            '        messages.append({"role": "system", "content": str(system)})\n'
            '    messages.append({"role": "user", "content": str(query)})\n'
            '    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)\n'
            '    return _coerce_token_id_list(ids, tokenizer=tokenizer)\n\n\n'
            'def _make_prompt_token_ids(*, tokenizer, query: str, system: str, chat_format: str) -> List[int]:\n'
            '    mode = str(chat_format).lower().strip()\n'
            '    if mode == "chatml":\n'
            '        return _make_chatml_prompt_tokens(tokenizer=tokenizer, query=query, system=system)\n'
            '    if mode == "template":\n'
            '        return _make_template_prompt_tokens(tokenizer=tokenizer, query=query, system=system)\n'
            '    if mode == "auto":\n'
            '        try:\n'
            '            return _make_template_prompt_tokens(tokenizer=tokenizer, query=query, system=system)\n'
            '        except Exception:\n'
            '            return _make_chatml_prompt_tokens(tokenizer=tokenizer, query=query, system=system)\n'
            '    raise ValueError(f"unsupported structured chat_format={chat_format}")\n\n\n',
        )
        qwen = qwen.replace(
            '        effective_chat_format = chat_format\n'
            '        if chat_format == "chatml":\n'
            '            try:\n'
            '                for s in test_samples:\n'
            '                    s.prompt_token_ids = _make_chatml_prompt_tokens(\n'
            '                        tokenizer=tokenizer, query=str(s.prompt), system=system_prompt\n'
            '                    )\n'
            '            except Exception:\n'
            '                effective_chat_format = "raw"\n'
            '                for s in test_samples:\n'
            '                    q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n'
            '                    raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n'
            '                    s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n'
            '        else:\n'
            '            for s in test_samples:\n'
            '                q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n'
            '                raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n'
            '                s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n',
            '        effective_chat_format = chat_format\n'
            '        structured_chat = str(chat_format).lower().strip() in {"chatml", "template", "auto"}\n'
            '        if structured_chat:\n'
            '            try:\n'
            '                for s in test_samples:\n'
            '                    s.prompt_token_ids = _make_prompt_token_ids(\n'
            '                        tokenizer=tokenizer, query=str(s.prompt), system=system_prompt, chat_format=chat_format\n'
            '                    )\n'
            '                if str(chat_format).lower().strip() == "auto":\n'
            '                    effective_chat_format = "template"\n'
            '            except Exception as e:\n'
            '                if str(chat_format).lower().strip() != "auto":\n'
            '                    raise\n'
            '                logger.info(f"chat_format_auto_fallback=raw err={type(e).__name__}:{e}")\n'
            '                effective_chat_format = "raw"\n'
            '                for s in test_samples:\n'
            '                    q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n'
            '                    raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n'
            '                    s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n'
            '        else:\n'
            '            for s in test_samples:\n'
            '                q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n'
            '                raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n'
            '                s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n',
        )
        qwen = qwen.replace(
            '    if chat_format == "chatml":\n'
            '        system_prompt = str(args.system_prompt)\n'
            '        for s in samples:\n'
            '            s.prompt_token_ids = _make_chatml_prompt_tokens(tokenizer=tokenizer, query=str(s.prompt), system=system_prompt)\n'
            '        for s in test_samples:\n'
            '            s.prompt_token_ids = _make_chatml_prompt_tokens(tokenizer=tokenizer, query=str(s.prompt), system=system_prompt)\n',
            '    structured_chat = str(chat_format).lower().strip() in {"chatml", "template", "auto"}\n'
            '    if structured_chat:\n'
            '        system_prompt = str(args.system_prompt)\n'
            '        try:\n'
            '            for s in samples:\n'
            '                s.prompt_token_ids = _make_prompt_token_ids(tokenizer=tokenizer, query=str(s.prompt), system=system_prompt, chat_format=chat_format)\n'
            '            for s in test_samples:\n'
            '                s.prompt_token_ids = _make_prompt_token_ids(tokenizer=tokenizer, query=str(s.prompt), system=system_prompt, chat_format=chat_format)\n'
            '            if str(chat_format).lower().strip() == "auto":\n'
            '                chat_format = "template"\n'
            '        except Exception as e:\n'
            '            if str(chat_format).lower().strip() != "auto":\n'
            '                raise\n'
            '            logger.info(f"chat_format_auto_fallback=raw err={type(e).__name__}:{e}")\n'
            '            chat_format = "raw"\n',
        )
        qwen = qwen.replace(
            '        if prompt_token_ids is not None:\n'
            '            prompt_ids_full = list(prompt_token_ids)\n'
            '        else:\n'
            '            prompt_ids_full = tokenizer(s.prompt, add_special_tokens=True, truncation=False)["input_ids"]\n',
            '        if prompt_token_ids is not None:\n'
            '            try:\n'
            '                prompt_ids_full = _coerce_token_id_list(prompt_token_ids, tokenizer=tokenizer)\n'
            '            except Exception:\n'
            '                prompt_ids_full = tokenizer(s.prompt, add_special_tokens=True, truncation=False)["input_ids"]\n'
            '        else:\n'
            '            prompt_ids_full = tokenizer(s.prompt, add_special_tokens=True, truncation=False)["input_ids"]\n',
        )
        qwen = qwen.replace(
            'method in {"fedra", "ffa-lora", "ffalora", "ffa_lora"}',
            'method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}',
        )
        qwen = qwen.replace(
            '            if new_B_state is not None and method not in {"fedra", "ffa-lora", "ffalora", "ffa_lora"}:\n'
            '                client_B_states[cid_int] = new_B_state\n'
            '            server.update_client_sensitivities(cid_int, b_sense)\n',
            '            if new_B_state is not None:\n'
            '                client_B_states[cid_int] = new_B_state\n'
            '            server.update_client_sensitivities(cid_int, b_sense)\n',
        )
        qwen = qwen.replace(
            '                max_steps=None,\n'
            '                device=device,\n',
            '                max_steps=(int(args.max_client_steps) if getattr(args, "max_client_steps", None) is not None else None),\n'
            '                device=device,\n',
        )
        qwen = replace_once_checked(
            qwen,
            '        eval_rng = np.random.RandomState(int(args.seed) + 1)\n'
            '        eval_samples = _subsample_samples(test_samples, float(getattr(args, "eval_ratio", 1.0)), eval_rng)\n',
            '        matched_round = getattr(args, "zero_shot_eval_round", None)\n'
            '        if matched_round is None:\n'
            '            eval_rng = np.random.RandomState(int(args.seed) + 1)\n'
            '            eval_samples = _subsample_samples(\n'
            '                test_samples, float(getattr(args, "eval_ratio", 1.0)), eval_rng\n'
            '            )\n'
            '            logger.info("zero_shot_eval_scope=random_global_subset")\n'
            '        else:\n'
            '            num_clients = int(args.num_clients)\n'
            '            alpha = getattr(args, "dirichlet_alpha", None)\n'
            '            if alpha is not None and float(alpha) > 0.0:\n'
            '                eval_client_splits = _split_clients_dirichlet(\n'
            '                    samples=test_samples, num_clients=num_clients,\n'
            '                    alpha=float(alpha), seed=int(args.seed) + 29,\n'
            '                )\n'
            '            else:\n'
            '                eval_client_splits = _split_clients_iid(test_samples, num_clients)\n'
            '            eval_samples = []\n'
            '            eval_ratio = float(getattr(args, "eval_ratio", 1.0))\n'
            '            for cid_int, split in enumerate(eval_client_splits):\n'
            '                if len(split) == 0:\n'
            '                    continue\n'
            '                k = int(max(1, int(len(split) * eval_ratio)))\n'
            '                k = min(k, len(split))\n'
            '                rng_eval = np.random.RandomState(\n'
            '                    int(args.seed) + 1000 * int(matched_round) + int(cid_int)\n'
            '                )\n'
            '                idx = rng_eval.choice(len(split), size=k, replace=False)\n'
            '                eval_samples.extend(split[int(i)] for i in idx)\n'
            '            logger.info(\n'
            '                f"zero_shot_eval_scope=matched_federated_round round={int(matched_round)} "\n'
            '                f"clients={num_clients} alpha={alpha} samples={len(eval_samples)}"\n'
            '            )\n',
            "gsm8k-zero-shot-matched-federated-subset",
        )
        qwen = replace_once_checked(
            qwen,
            '        else:\n'
            '            for s in test_samples:\n'
            '                q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n'
            '                raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n'
            '                s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n',
            '        else:\n'
            '            if getattr(args, "zero_shot_eval_round", None) is None:\n'
            '                for s in test_samples:\n'
            '                    q = _strip_raw_prompt_to_question(getattr(s, "prompt", ""))\n'
            '                    raw_prompt = _build_zero_shot_raw_prompt(question=q, lang=str(getattr(s, "lang", "en")))\n'
            '                    s.prompt = _join_fewshot_prompt(fewshot_prompt=fewshot_prompt, prompt=raw_prompt)\n'
            '            else:\n'
            '                logger.info("zero_shot_prompt_scope=matched_training_eval_raw_prompt")\n',
            "gsm8k-zero-shot-matched-raw-prompt",
        )
        qwen = replace_once_checked(
            qwen,
            '        eval_zh = [s for s in eval_samples if str(getattr(s, "lang", "zh")).lower() == "zh"]\n'
            '        eval_en = [s for s in eval_samples if str(getattr(s, "lang", "zh")).lower() != "zh"]\n'
            '        if len(eval_zh) > 0 and len(eval_zh) != len(eval_samples):\n'
            '            a, c, t = _evaluate_gsm8k_accuracy(\n'
            '                model=model,\n'
            '                tokenizer=tokenizer,\n'
            '                samples=eval_zh,\n'
            '                chat_format=effective_chat_format,\n'
            '                max_length=int(args.max_length),\n'
            '                max_new_tokens=int(getattr(args, "eval_max_new_tokens", 128)),\n'
            '                device=device,\n'
            '            )\n'
            '            logger.info(f"zero_shot zh acc={a:.4f} correct={c} total={t} samples={len(eval_zh)}")\n'
            '        if len(eval_en) > 0 and len(eval_en) != len(eval_samples):\n'
            '            a, c, t = _evaluate_gsm8k_accuracy(\n'
            '                model=model,\n'
            '                tokenizer=tokenizer,\n'
            '                samples=eval_en,\n'
            '                chat_format=effective_chat_format,\n'
            '                max_length=int(args.max_length),\n'
            '                max_new_tokens=int(getattr(args, "eval_max_new_tokens", 128)),\n'
            '                device=device,\n'
            '            )\n'
            '            logger.info(f"zero_shot en acc={a:.4f} correct={c} total={t} samples={len(eval_en)}")\n',
            '        if bool(getattr(args, "zero_shot_report_subgroups", True)):\n'
            '            eval_zh = [s for s in eval_samples if str(getattr(s, "lang", "zh")).lower() == "zh"]\n'
            '            eval_en = [s for s in eval_samples if str(getattr(s, "lang", "zh")).lower() != "zh"]\n'
            '            if len(eval_zh) > 0 and len(eval_zh) != len(eval_samples):\n'
            '                a, c, t = _evaluate_gsm8k_accuracy(\n'
            '                    model=model, tokenizer=tokenizer, samples=eval_zh,\n'
            '                    chat_format=effective_chat_format, max_length=int(args.max_length),\n'
            '                    max_new_tokens=int(getattr(args, "eval_max_new_tokens", 128)), device=device,\n'
            '                )\n'
            '                logger.info(f"zero_shot zh acc={a:.4f} correct={c} total={t} samples={len(eval_zh)}")\n'
            '            if len(eval_en) > 0 and len(eval_en) != len(eval_samples):\n'
            '                a, c, t = _evaluate_gsm8k_accuracy(\n'
            '                    model=model, tokenizer=tokenizer, samples=eval_en,\n'
            '                    chat_format=effective_chat_format, max_length=int(args.max_length),\n'
            '                    max_new_tokens=int(getattr(args, "eval_max_new_tokens", 128)), device=device,\n'
            '                )\n'
            '                logger.info(f"zero_shot en acc={a:.4f} correct={c} total={t} samples={len(eval_en)}")\n',
            "gsm8k-zero-shot-optional-subgroups",
        )
        qwen = qwen.replace(
            '        logger.info(f"zero_shot overall acc={eval_acc:.4f} correct={eval_correct} total={eval_total} samples={len(eval_samples)}")\n\n'
            '        if bool(getattr(args, "zero_shot_report_subgroups", True)):\n',
            '        logger.info(f"zero_shot overall acc={eval_acc:.4f} correct={eval_correct} total={eval_total} samples={len(eval_samples)}")\n'
            '        if bool(getattr(args, "mec", False)):\n'
            '            mec_cfg = FedHelloMecConfig(\n'
            '                num_clients=int(getattr(args, "num_clients", 1)), num_rounds=1, seed=int(args.seed),\n'
            '                heterogeneity_type="uniform", allocation_ratio=1.0,\n'
            '                channel_model=str(args.mec_channel), channel_correlation=float(args.mec_channel_correlation),\n'
            '                channel_gain_mean=float(args.mec_channel_gain_mean), dropout_model=str(args.mec_dropout_model),\n'
            '                dropout_rate=float(args.mec_dropout_rate), ge_good_to_bad=float(args.mec_ge_good_to_bad),\n'
            '                duration_hazard_rate=float(args.mec_duration_hazard_rate),\n'
            '                total_bandwidth_mhz=float(args.mec_total_bw_mhz), power_max_dbm=float(args.mec_power_max_dbm),\n'
            '                deadline_s=float(args.mec_deadline_s),\n'
            '            )\n'
            '            mec_controller = FedHelloMecController(mec_cfg)\n'
            '            mec_round = mec_controller.run_round(\n'
            '                round_num=1, selected_client_ids=list(range(int(args.num_clients))),\n'
            '                payload_bytes={cid: 0 for cid in range(int(args.num_clients))},\n'
            '                active_blocks={cid: 0 for cid in range(int(args.num_clients))},\n'
            '                train_samples={cid: 0 for cid in range(int(args.num_clients))},\n'
            '                wall_clock_s=0.0, download_bytes=0.0,\n'
            '            )\n'
            '            _append_metrics_jsonl(os.path.join(str(args.log_dir), "mec_metrics.jsonl"), {\n'
            '                "round": 1, "row_type": "mec_round", "run_mode": "zero_shot",\n'
            '                **mec_round.diagnostics,\n'
            '            })\n'
            '            logger.info(f"mec_zero_shot policy=equal_fixed arrived={mec_round.diagnostics[\'arrived_clients\']} "\n'
            '                        f"compute_energy_j={mec_round.diagnostics[\'compute_energy_round_j\']:.6f} "\n'
            '                        f"tx_energy_j={mec_round.diagnostics[\'tx_energy_round_j\']:.6f} "\n'
            '                        f"energy_model=model-based simulated")\n'
            '\n        if bool(getattr(args, "zero_shot_report_subgroups", True)):\n',
            1,
        )
        qwen = replace_once_checked(
            qwen,
            '    client_sample_counts: Dict[int, int] = {cid: len(client_splits[cid]) for cid in range(num_clients)}\n'
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n',
            '    client_capacities = [\n'
            '        float(server.client_layer_counts.get(cid, num_total_layers)) / float(max(1, num_total_layers))\n'
            '        for cid in range(num_clients)\n'
            '    ]\n'
            '    flora_client_ranks = _build_flora_client_ranks(\n'
            '        client_capacities=client_capacities,\n'
            '        max_rank=int(args.lora_rank),\n'
            '        policy=str(getattr(args, "flora_rank_policy", "capacity")),\n'
            '        min_rank=int(getattr(args, "flora_min_rank", 1)),\n'
            '        explicit_ranks=getattr(args, "flora_local_ranks", None),\n'
            '    )\n'
            '    if method == "flora":\n'
            '        logger.info(\n'
            '            f"flora_rank_policy={args.flora_rank_policy} ranks={flora_client_ranks} "\n'
            '            f"min_rank={min(flora_client_ranks)} max_rank={max(flora_client_ranks)}"\n'
            '        )\n'
            '    client_sample_counts: Dict[int, int] = {cid: len(client_splits[cid]) for cid in range(num_clients)}\n'
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n',
            "gsm8k-flora-rank-assignment",
        )
        qwen = replace_once_checked(
            qwen,
            '        global_A = server.get_global_A_matrices()\n'
            '        global_B = server.get_global_B_matrices() if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None\n',
            '        global_A = server.get_global_A_matrices()\n'
            '        global_B = server.get_global_B_matrices() if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None\n'
            '        if method == "flora":\n'
            '            for layer in lora_layers.values():\n'
            '                layer.merge_lora_weights()\n'
            '            logger.info(f"round={round_num} flora_global_adapter_merged_once rank={next(iter(global_A.values())).shape[0]}")\n',
            "gsm8k-flora-round-merge",
        )
        qwen = replace_once_checked(
            qwen,
            '            _set_all_A(lora_layers, global_A)\n'
            '            if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}:\n'
            '                if global_B is None:\n'
            '                    raise ValueError("fedra 需要 global_B")\n'
            '                _set_all_B(lora_layers, global_B)\n',
            '            if method == "flora":\n'
            '                for layer in lora_layers.values():\n'
            '                    layer.reset_lora(rank=flora_client_ranks[cid_int])\n'
            '                logger.info(f"round={round_num} client={cid_int} flora_local_rank={flora_client_ranks[cid_int]}")\n'
            '            else:\n'
            '                _set_all_A(lora_layers, global_A)\n'
            '                if method in {"fedra", "fedhello", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}:\n'
            '                    if global_B is None:\n'
            '                        raise ValueError(f"{method} needs global_B")\n'
            '                    _set_all_B(lora_layers, global_B)\n',
            "gsm8k-flora-client-reset",
        )
        qwen = replace_once_checked(
            qwen,
            '        eval_ratio = float(getattr(args, "eval_ratio", 0.1))\n'
            '        round_eval_correct = 0\n'
            '        round_eval_total = 0\n'
            '        for cid_int in range(num_clients):\n',
            '        eval_ratio = float(getattr(args, "eval_ratio", 0.1))\n'
            '        round_eval_correct = 0\n'
            '        round_eval_total = 0\n'
            '        personalized_b_methods = {\n'
            '            "fedharp", "fedharp_r", "fedharp_g", "fedharp_l", "fedharp_a",\n'
            '            "fedsalora", "fedsa-lora", "fedsa_lora",\n'
            '        }\n'
            '        if method in personalized_b_methods:\n'
            '            eval_adapter_policy = "global_A+local_B"\n'
            '        else:\n'
            '            eval_adapter_policy = "global_aggregated_adapter"\n'
            '        logger.info(\n'
            '            f"round={round_num} eval_policy adapter={eval_adapter_policy} head=causal_lm"\n'
            '        )\n'
            '        for cid_int in range(num_clients):\n',
            "gsm8k-eval-policy-log",
        )
        qwen = replace_once_checked(
            qwen,
            '            _set_all_A(lora_layers, global_A_after)\n'
            '            if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}:\n'
            '                if global_B_after is None:\n'
            '                    raise ValueError("fedra 需要 global_B")\n'
            '                _set_all_B(lora_layers, global_B_after)\n'
            '            else:\n'
            '                b_state = client_B_states.get(cid_int)\n'
            '                if b_state is not None:\n'
            '                    _set_all_B(lora_layers, b_state)\n',
            '            _set_all_A(lora_layers, global_A_after)\n'
            '            if method in personalized_b_methods:\n'
            '                b_state = client_B_states.get(cid_int)\n'
            '                if b_state is None:\n'
            '                    raise ValueError(f"missing personalized B state for client {cid_int}")\n'
            '                _set_all_B(lora_layers, b_state)\n'
            '            else:\n'
            '                if global_B_after is None:\n'
            '                    raise ValueError(f"{method} evaluation requires aggregated global B")\n'
            '                _set_all_B(lora_layers, global_B_after)\n',
            "gsm8k-method-consistent-eval",
        )
        qwen = replace_once_checked(
            qwen,
            '    p.add_argument("--zero_shot_report_subgroups", action=argparse.BooleanOptionalAction, default=True)\n',
            '    p.add_argument("--zero_shot_report_subgroups", action=argparse.BooleanOptionalAction, default=True)\n'
            '    p.add_argument("--enable_lpc", action=argparse.BooleanOptionalAction, default=False)\n'
            '    p.add_argument("--lpc_interval", type=int, default=4)\n'
            '    p.add_argument("--lpc_num_epochs", type=int, default=1)\n'
            '    p.add_argument("--save_final_state", action=argparse.BooleanOptionalAction, default=True)\n',
            "gsm8k-lpc-and-checkpoint-args",
        )
        qwen = replace_once_checked(
            qwen,
            '    cut_tokens = ["<|endoftext|>", "\\n\\n\\n", "\\n\\n", "Question:"]\n',
            '    cut_tokens = ["<|endoftext|>", "Question:"]\n',
            "gsm8k-preserve-multistep-completion",
        )
        qwen = replace_once_checked(
            qwen,
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n',
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n'
            '    client_layer_staleness: Dict[int, Dict[str, int]] = {\n'
            '        cid: {name: 0 for name in server.layer_names} for cid in range(num_clients)\n'
            '    }\n'
            '    lpc_event_count = 0\n'
            '    mec_controller = None\n'
            '    mec_metrics_path = None\n'
            '    mec_run_log_path = None\n'
            '    if bool(getattr(args, "mec", False)):\n'
            '        mec_output_dir = str(getattr(args, "mec_output_dir", None) or args.log_dir)\n'
            '        os.makedirs(mec_output_dir, exist_ok=True)\n'
            '        mec_metrics_path = os.path.join(mec_output_dir, "mec_metrics.jsonl")\n'
            '        mec_run_log_path = os.path.join(mec_output_dir, "mec_run.log")\n'
            '        mec_cfg = FedHelloMecConfig(\n'
            '            num_clients=int(num_clients), num_rounds=int(args.num_rounds), seed=int(args.seed),\n'
            '            heterogeneity_type="uniform", allocation_ratio=1.0,\n'
            '            channel_model=str(args.mec_channel), channel_correlation=float(args.mec_channel_correlation),\n'
            '            channel_gain_mean=float(args.mec_channel_gain_mean), dropout_model=str(args.mec_dropout_model),\n'
            '            dropout_rate=float(args.mec_dropout_rate), ge_good_to_bad=float(args.mec_ge_good_to_bad),\n'
            '            duration_hazard_rate=float(args.mec_duration_hazard_rate),\n'
            '            total_bandwidth_mhz=float(args.mec_total_bw_mhz), power_max_dbm=float(args.mec_power_max_dbm),\n'
            '            deadline_s=float(args.mec_deadline_s),\n'
            '        )\n'
            '        mec_controller = FedHelloMecController(mec_cfg)\n'
            '        with open(mec_run_log_path, "w", encoding="utf-8") as f:\n'
            '            f.write(f"START task=gsm8k method={method} policy=equal_fixed "\n'
            '                    f"trace_hash={mec_controller.trace_hash} energy_model=model-based simulated\\n")\n'
            '        logger.info(f"mec_enabled=True policy=equal_fixed trace_hash={mec_controller.trace_hash} "\n'
            '                    f"energy_model=model-based simulated")\n'
            '    logger.info(\n'
            '        f"lpc_config enabled={bool(args.enable_lpc and method == \'fedharp\')} "\n'
            '        f"interval={int(args.lpc_interval)} epochs={int(args.lpc_num_epochs)}"\n'
            '    )\n',
            "gsm8k-lpc-state",
        )
        qwen = replace_once_checked(
            qwen,
            '            t_client_train = time.time()\n'
            '            delta_A, delta_B, b_sense, new_B_state, train_avg_loss = _train_one_client_round(\n',
            '            lpc_layers: Set[str] = set()\n'
            '            if bool(args.enable_lpc) and method == "fedharp":\n'
            '                interval = max(1, int(args.lpc_interval))\n'
            '                lpc_layers = {\n'
            '                    name for name in allocated_layers\n'
            '                    if int(client_layer_staleness[cid_int].get(name, 0)) >= interval\n'
            '                }\n'
            '                if lpc_layers:\n'
            '                    _set_all_A(lora_layers, global_A)\n'
            '                    _set_all_B(lora_layers, client_B_states[cid_int])\n'
            '                    logger.info(\n'
            '                        f"round={round_num} client={cid_int} lpc_start "\n'
            '                        f"layers={len(lpc_layers)} interval={interval}"\n'
            '                    )\n'
            '                    _, _, _, _, lpc_loss = _train_one_client_round(\n'
            '                        model=model, lora_layers=lora_layers, dataloader=loader,\n'
            '                        allocated_layers=set(lpc_layers), method="ffa-lora",\n'
            '                        lr=float(args.client_lr), num_epochs=int(args.lpc_num_epochs),\n'
            '                        grad_accum_steps=int(args.grad_accum_steps),\n'
            '                        max_steps=(int(args.max_client_steps) if getattr(args, "max_client_steps", None) is not None else None),\n'
            '                        device=device, client_B_state=None, client_id=cid_int,\n'
            '                        round_num=round_num, logger=logger, log_interval=int(args.log_interval),\n'
            '                    )\n'
            '                    client_B_states[cid_int] = _get_all_B(lora_layers)\n'
            '                    lpc_event_count += 1\n'
            '                    logger.info(\n'
            '                        f"round={round_num} client={cid_int} lpc_done "\n'
            '                        f"layers={len(lpc_layers)} loss={float(lpc_loss):.6f} events={lpc_event_count}"\n'
            '                    )\n'
            '            t_client_train = time.time()\n'
            '            delta_A, delta_B, b_sense, new_B_state, train_avg_loss = _train_one_client_round(\n',
            "gsm8k-lpc-before-ordinary-training",
        )
        qwen = replace_once_checked(
            qwen,
            '    p.add_argument("--save_final_state", action=argparse.BooleanOptionalAction, default=True)\n',
            '    p.add_argument("--save_final_state", action=argparse.BooleanOptionalAction, default=True)\n'
            '    p.add_argument("--lpc_mode", type=str, default="train", choices=["train", "projection"])\n'
            '    p.add_argument("--projection_ridge", type=float, default=1e-6)\n',
            "gsm8k-lpc-projection-args",
        )
        qwen = replace_once_checked(
            qwen,
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n',
            '    client_B_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_B(lora_layers) for cid in range(num_clients)}\n'
            '    client_A_states: Dict[int, Dict[str, torch.Tensor]] = {cid: _get_all_A(lora_layers) for cid in range(num_clients)}\n',
            "gsm8k-lpc-projection-aux",
        )
        qwen = replace_once_checked(
            qwen,
            '                    _, _, _, _, lpc_loss = _train_one_client_round(\n'
            '                        model=model, lora_layers=lora_layers, dataloader=loader,\n'
            '                        allocated_layers=set(lpc_layers), method="ffa-lora",\n'
            '                        lr=float(args.client_lr), num_epochs=int(args.lpc_num_epochs),\n'
            '                        grad_accum_steps=int(args.grad_accum_steps),\n'
            '                        max_steps=(int(args.max_client_steps) if getattr(args, "max_client_steps", None) is not None else None),\n'
            '                        device=device, client_B_state=None, client_id=cid_int,\n'
            '                        round_num=round_num, logger=logger, log_interval=int(args.log_interval),\n'
            '                    )\n'
            '                    client_B_states[cid_int] = _get_all_B(lora_layers)\n'
            '                    lpc_event_count += 1\n'
            '                    logger.info(\n'
            '                        f"round={round_num} client={cid_int} lpc_done "\n'
            '                        f"layers={len(lpc_layers)} loss={float(lpc_loss):.6f} events={lpc_event_count}"\n'
            '                    )\n',
            '                    if str(getattr(args, "lpc_mode", "train")).lower() == "projection":\n'
            '                        from factor_projection import project_B_to_new_A as _lpc_project_B\n'
            '                        _lpc_old_A = client_A_states.get(cid_int)\n'
            '                        if _lpc_old_A is None:\n'
            '                            _lpc_old_A = {n: t.detach().clone() for n, t in _get_all_A(lora_layers).items()}\n'
            '                        _lpc_old_B = client_B_states[cid_int]\n'
            '                        _lpc_new_B = _get_all_B(lora_layers)\n'
            '                        for _ln in sorted(set(lpc_layers)):\n'
            '                            if _ln not in _lpc_old_A or _ln not in _lpc_old_B or _ln not in global_A:\n'
            '                                continue\n'
            '                            _Bstar, _diag = _lpc_project_B(\n'
            '                                _lpc_old_A[_ln], _lpc_old_B[_ln], global_A[_ln],\n'
            '                                ridge=float(getattr(args, "projection_ridge", 1e-6)),\n'
            '                            )\n'
            '                            _lpc_new_B[_ln] = _Bstar\n'
            '                            logger.info(\n'
            '                                f"round={round_num} client={cid_int} type=lpc_projection_event "\n'
            '                                f"layer={_ln} residual_before={float(_diag[\'residual_before\']):.8g} "\n'
            '                                f"residual_after={float(_diag[\'residual_after\']):.8g} "\n'
            '                                f"relative_reduction={float(_diag[\'relative_reduction\']):.8g}"\n'
            '                            )\n'
            '                        _set_all_B(lora_layers, _lpc_new_B)\n'
            '                        lpc_loss = 0.0\n'
            '                    else:\n'
            '                        _, _, _, _, lpc_loss = _train_one_client_round(\n'
            '                            model=model, lora_layers=lora_layers, dataloader=loader,\n'
            '                            allocated_layers=set(lpc_layers), method="ffa-lora",\n'
            '                            lr=float(args.client_lr), num_epochs=int(args.lpc_num_epochs),\n'
            '                            grad_accum_steps=int(args.grad_accum_steps),\n'
            '                            max_steps=(int(args.max_client_steps) if getattr(args, "max_client_steps", None) is not None else None),\n'
            '                            device=device, client_B_state=None, client_id=cid_int,\n'
            '                            round_num=round_num, logger=logger, log_interval=int(args.log_interval),\n'
            '                        )\n'
            '                    client_B_states[cid_int] = _get_all_B(lora_layers)\n'
            '                    client_A_states[cid_int] = _get_all_A(lora_layers)\n'
            '                    lpc_event_count += 1\n'
            '                    logger.info(\n'
            '                        f"round={round_num} client={cid_int} lpc_done "\n'
            '                        f"layers={len(lpc_layers)} loss={float(lpc_loss):.6f} events={lpc_event_count}"\n'
            '                    )\n',
            "gsm8k-lpc-projection-branch",
        )
        qwen = replace_once_checked(
            qwen,
            '            if new_B_state is not None:\n'
            '                client_B_states[cid_int] = new_B_state\n',
            '            if new_B_state is not None:\n'
            '                client_B_states[cid_int] = new_B_state\n'
            '                client_A_states[cid_int] = _get_all_A(lora_layers)\n',
            "gsm8k-lpc-projection-asave",
        )
        qwen = replace_once_checked(
            qwen,
            '            server.update_client_sensitivities(cid_int, b_sense)\n'
            '\n'
            '        server.aggregate_matrices(\n',
            '            server.update_client_sensitivities(cid_int, b_sense)\n'
            '\n'
            '        selected_ids = {int(cid) for cid in selected}\n'
            '        for stale_cid in range(num_clients):\n'
            '            active = set(allocation_map.get(stale_cid, set())) if stale_cid in selected_ids else set()\n'
            '            for stale_name in server.layer_names:\n'
            '                if stale_name in active:\n'
            '                    client_layer_staleness[stale_cid][stale_name] = 0\n'
            '                else:\n'
            '                    client_layer_staleness[stale_cid][stale_name] += 1\n'
            '\n'
            '        server.aggregate_matrices(\n',
            "gsm8k-lpc-staleness-update",
        )
        qwen = replace_once_checked(
            qwen,
            '        server.aggregate_matrices(\n'
            '            client_updates_A=client_updates_A,\n'
            '            client_updates_B=(client_updates_B if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None),\n'
            '            client_sample_counts=client_sample_counts,\n'
            '            allocation_map=allocation_map,\n'
            '        )\n',
            '        if mec_controller is not None:\n'
            '            payload_by_client = {\n'
            '                int(cid): int(_dict_tensor_bytes(client_updates_A.get(int(cid), {})))\n'
            '                + int(_dict_tensor_bytes(client_updates_B.get(int(cid), {})))\n'
            '                for cid in selected\n'
            '            }\n'
            '            mec_round = mec_controller.run_round(\n'
            '                round_num=round_num, selected_client_ids=[int(cid) for cid in selected],\n'
            '                payload_bytes=payload_by_client,\n'
            '                active_blocks={int(cid): len(allocation_map.get(int(cid), set())) for cid in selected},\n'
            '                train_samples={int(cid): int(client_sample_counts[int(cid)]) for cid in selected},\n'
            '                wall_clock_s=float(time.time() - t_round),\n'
            '                download_bytes=float(sum(client_train_metrics[int(cid)].get("download_bytes", 0) for cid in selected)),\n'
            '            )\n'
            '            arrived_ids = set(int(cid) for cid in mec_round.arrived_client_ids)\n'
            '            client_updates_A = {cid: value for cid, value in client_updates_A.items() if int(cid) in arrived_ids}\n'
            '            client_updates_B = {cid: value for cid, value in client_updates_B.items() if int(cid) in arrived_ids}\n'
            '            if len(client_updates_A) == 0:\n'
            '                raise RuntimeError(f"round {round_num}: no GSM8K update arrived; refusing empty aggregation")\n'
            '            _append_metrics_jsonl(mec_metrics_path, {\n'
            '                "round": int(round_num), "row_type": "mec_round",\n'
            '                **mec_round.diagnostics,\n'
            '            })\n'
            '            with open(mec_run_log_path, "a", encoding="utf-8") as f:\n'
            '                f.write(f"round={round_num} arrived={mec_round.diagnostics[\'arrived_clients\']} "\n'
            '                        f"deadline_miss={mec_round.diagnostics[\'deadline_miss_clients\']} "\n'
            '                        f"compute_energy_j={mec_round.diagnostics[\'compute_energy_round_j\']:.6f} "\n'
            '                        f"tx_energy_j={mec_round.diagnostics[\'tx_energy_round_j\']:.6f}\\n")\n'
            '\n'
            '        server.aggregate_matrices(\n'
            '            client_updates_A=client_updates_A,\n'
            '            client_updates_B=(client_updates_B if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None),\n'
            '            client_sample_counts=client_sample_counts,\n'
            '            allocation_map=allocation_map,\n'
            '        )\n',
            "gsm8k-mec-arrival-gate",
        )
        qwen = replace_once_checked(
            qwen,
            '            train_time_sec = float(time.time() - t_client_train)\n'
            '            upload_bytes = _dict_tensor_bytes(delta_A) + (_dict_tensor_bytes(delta_B) if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else 0)\n'
            '            peak_alloc = int(torch.cuda.max_memory_allocated(device=device)) if device.type == "cuda" else 0\n',
            '            train_time_sec = float(time.time() - t_client_train)\n'
            '            upload_bytes = _dict_tensor_bytes(delta_A) + (_dict_tensor_bytes(delta_B) if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else 0)\n'
            '            trainable_numel = sum(\n'
            '                int(lora_layers[name].lora_A.numel()) + int(lora_layers[name].lora_B.numel())\n'
            '                for name in allocated_layers if name in lora_layers\n'
            '            )\n'
            '            compute_flops = 3.0 * float(trainable_numel) * float(len(client_splits[cid_int])) * float(args.num_epochs)\n'
            '            compute_tflops = compute_flops / 1e12\n'
            '            flops_per_sec = compute_flops / max(1e-9, train_time_sec)\n'
            '            peak_alloc = int(torch.cuda.max_memory_allocated(device=device)) if device.type == "cuda" else 0\n',
            "gsm8k-local-flops",
        )
        qwen = replace_once_checked(
            qwen,
            '            client_train_metrics[cid_int] = {\n'
            '                "train_time_sec": train_time_sec,\n'
            '                "train_avg_loss": float(train_avg_loss),\n'
            '                "upload_bytes": int(upload_bytes),\n',
            '            client_train_metrics[cid_int] = {\n'
            '                "train_time_sec": train_time_sec,\n'
            '                "train_avg_loss": float(train_avg_loss),\n'
            '                "trainable_numel": int(trainable_numel),\n'
            '                "compute_flops": float(compute_flops),\n'
            '                "compute_tflops": float(compute_tflops),\n'
            '                "flops_per_sec": float(flops_per_sec),\n'
            '                "upload_bytes": int(upload_bytes),\n',
            "gsm8k-local-flops-metrics",
        )
        qwen = replace_once_checked(
            qwen,
            '            logger.info(\n'
            '                f"round={round_num} client={cid_int} train_done time_sec={train_time_sec:.2f} "\n'
            '                f"upload_bytes={upload_bytes} download_bytes={down_bytes} peak_alloc={peak_alloc} peak_reserved={peak_reserved}"\n'
            '            )\n',
            '            logger.info(\n'
            '                f"round={round_num} client={cid_int} train_done time_sec={train_time_sec:.2f} "\n'
            '                f"compute_tflops={compute_tflops:.6f} flops_per_sec={flops_per_sec:.3f} "\n'
            '                f"upload_bytes={upload_bytes} download_bytes={down_bytes} peak_alloc={peak_alloc} peak_reserved={peak_reserved}"\n'
            '            )\n',
            "gsm8k-local-flops-log",
        )
        qwen = qwen.replace(
            '                "train_time_sec": float(train_m.get("train_time_sec", 0.0)),\n'
            '                "train_avg_loss": float(train_m.get("train_avg_loss", 0.0)),\n',
            '                "train_time_sec": float(train_m.get("train_time_sec", 0.0)),\n'
            '                "train_avg_loss": float(train_m.get("train_avg_loss", 0.0)),\n'
            '                "trainable_numel": int(train_m.get("trainable_numel", 0)),\n'
            '                "compute_flops": float(train_m.get("compute_flops", 0.0)),\n'
            '                "compute_tflops": float(train_m.get("compute_tflops", 0.0)),\n'
            '                "flops_per_sec": float(train_m.get("flops_per_sec", 0.0)),\n',
            1,
        )
        qwen = replace_once_checked(
            qwen,
            '        global_A_after = server.get_global_A_matrices()\n'
            '        global_B_after = server.get_global_B_matrices() if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None\n'
            '\n'
            '        eval_every_rounds = int(getattr(args, "eval_every_rounds", 5) or 5)\n',
            '        global_A_after = server.get_global_A_matrices()\n'
            '        global_B_after = server.get_global_B_matrices() if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None\n'
            '\n'
            '        if bool(args.save_final_state) and round_num == int(args.num_rounds):\n'
            '            state_path = os.path.join(args.log_dir, f"state_round_{round_num}.pt")\n'
            '            cpu_global_A = {name: value.detach().cpu() for name, value in global_A_after.items()}\n'
            '            cpu_global_B = None if global_B_after is None else {\n'
            '                name: value.detach().cpu() for name, value in global_B_after.items()\n'
            '            }\n'
            '            cpu_client_B = {\n'
            '                cid: {name: value.detach().cpu() for name, value in state.items()}\n'
            '                for cid, state in client_B_states.items()\n'
            '            }\n'
            '            torch.save({\n'
            '                "round": int(round_num), "method": str(method), "args": dict(vars(args)),\n'
            '                "global_A": cpu_global_A, "global_B": cpu_global_B,\n'
            '                "client_B_states": cpu_client_B,\n'
            '                "client_layer_staleness": client_layer_staleness,\n'
            '                "lpc_event_count": int(lpc_event_count),\n'
            '            }, state_path)\n'
            '            logger.info(f"saved_final_state={state_path} lpc_events={lpc_event_count}")\n'
            '\n'
            '        eval_every_rounds = int(getattr(args, "eval_every_rounds", 5) or 5)\n',
            "gsm8k-save-final-state",
        )
        _EMBEDDED_TASK_SOURCES["qwen"] = qwen
        _EMBEDDED_TASK_SOURCES["gsm8k"] = qwen


_patch_embedded_task_sources_for_personalized_eval()


def _patch_embedded_task_sources_for_fedsonata():
    """Port FedSONATA-CORA (RIFT-LEF + CORA predictor/corrector) into the
    embedded causal-LM harness (qwen/gsm8k), mirroring the text entry D052.

    Opt-in via ``--factor_policy {equal_fixed,fedsonata_cora,cora}``; the
    default keeps the existing FedHelloMecController path untouched.  Fail-safe:
    if ANY anchor is missing the source is left unchanged.
    """
    src = _EMBEDDED_TASK_SOURCES.get("qwen")
    if not src:
        return
    try:
        a = "from mec.fedhello_mec import FedHelloMecConfig, FedHelloMecController\n"
        assert a in src
        src = src.replace(a, a + "from mec.fedsonata_text_mec import FedSonataTextMecConfig, FedSonataTextMecController\n", 1)

        b = '    p.add_argument("--mec_deadline_s", type=float, default=10.0)\n'
        assert b in src
        src = src.replace(b, b +
            '    p.add_argument("--factor_policy", type=str, default="equal_fixed",\n'
            '                   choices=["equal_fixed", "fedsonata_cora", "cora"])\n'
            '    p.add_argument("--rift_budget_ratio", type=float, default=0.35)\n'
            '    p.add_argument("--rift_ef_decay", type=float, default=0.5)\n'
            '    p.add_argument("--mec_d_min", type=float, default=0.5)\n'
            '    p.add_argument("--mec_d_max", type=float, default=10.0)\n'
            '    p.add_argument("--mec_lambda_e", type=float, default=0.1)\n'
            '    p.add_argument("--mec_lambda_u", type=float, default=1.0)\n'
            '    p.add_argument("--mec_omega_net", type=float, default=1.0)\n'
            '    p.add_argument("--mec_reliability_beta", type=float, default=0.8)\n'
            '    p.add_argument("--mec_deadline_grid_size", type=int, default=24)\n', 1)

        c = ("        mec_controller = FedHelloMecController(mec_cfg)\n"
             "        with open(mec_run_log_path, \"w\", encoding=\"utf-8\") as f:\n")
        assert c in src
        src = src.replace(c,
            '        globals()["_fs_enabled"] = str(getattr(args, "factor_policy", "equal_fixed")) in {"fedsonata_cora", "cora"}\n'
            '        if _fs_enabled:\n'
            '            _fs_cfg = FedSonataTextMecConfig(\n'
            '                num_clients=int(num_clients), num_rounds=int(args.num_rounds), seed=int(args.seed),\n'
            '                factor_policy=str(args.factor_policy), heterogeneity_type="uniform",\n'
            '                allocation_ratio=1.0, channel_model=str(args.mec_channel),\n'
            '                channel_correlation=float(args.mec_channel_correlation),\n'
            '                channel_gain_mean=float(args.mec_channel_gain_mean),\n'
            '                dropout_model=str(args.mec_dropout_model), dropout_rate=float(args.mec_dropout_rate),\n'
            '                ge_good_to_bad=float(args.mec_ge_good_to_bad),\n'
            '                duration_hazard_rate=float(args.mec_duration_hazard_rate),\n'
            '                total_bandwidth_mhz=float(args.mec_total_bw_mhz), power_max_dbm=float(args.mec_power_max_dbm),\n'
            '                d_min_s=float(getattr(args, "mec_d_min", 0.5)), d_max_s=float(getattr(args, "mec_d_max", 10.0)),\n'
            '                deadline_grid_size=int(getattr(args, "mec_deadline_grid_size", 24)),\n'
            '                lambda_e=float(getattr(args, "mec_lambda_e", 0.1)), lambda_u=float(getattr(args, "mec_lambda_u", 1.0)),\n'
            '                omega_net=float(getattr(args, "mec_omega_net", 1.0)), reliability_beta=float(getattr(args, "mec_reliability_beta", 0.8)),\n'
            '                rift_budget_ratio=float(getattr(args, "rift_budget_ratio", 0.35)), rift_ef_decay=float(getattr(args, "rift_ef_decay", 0.5)),\n'
            '            )\n'
            '            mec_controller = FedSonataTextMecController(_fs_cfg)\n'
            '        else:\n'
            '            globals()["_fs_enabled"] = False\n'
            '            mec_controller = FedHelloMecController(mec_cfg)\n'
            '        _fs_rift_records: Dict[int, object] = {}\n'
            '        with open(mec_run_log_path, \"w\", encoding=\"utf-8\") as f:\n', 1)

        d0 = "    mec_controller = None\n"
        assert d0 in src
        src = src.replace(d0, "    mec_controller = None\n    globals().setdefault(\"_fs_enabled\", False)\n    _fs_rift_records: Dict[int, object] = {}\n", 1)

        d = "        for cid in selected:\n            cid_int = int(cid)\n"
        assert d in src
        src = src.replace(d,
            '        _fs_rift_records = {}\n'
            '        if _fs_enabled:\n'
            '            _fs_full_bytes = {\n'
            '                int(c): int(sum(int(lora_layers[n].lora_A.numel()) + int(lora_layers[n].lora_B.numel())\n'
            '                                for n in allocation_map.get(int(c), set()) if n in lora_layers) * 4)\n'
            '                for c in selected\n'
            '            }\n'
            '            mec_controller.pre_round(\n'
            '                round_num=round_num, selected_client_ids=[int(c) for c in selected],\n'
            '                full_ab_bytes=_fs_full_bytes,\n'
            '                allocated_layers={int(c): set(allocation_map.get(int(c), set())) for c in selected},\n'
            '                train_samples={int(c): int(client_sample_counts[int(c)]) for c in selected},\n'
            '                layer_names=list(server.layer_names),\n'
            '            )\n'
            + d, 1)

        e = ("                client_B_state=client_B_states.get(cid_int),\n"
             "                client_id=cid_int,\n"
             "                round_num=round_num,\n"
             "                logger=logger,\n"
             "                log_interval=int(args.log_interval),\n"
             "            )\n")
        assert e in src
        src = src.replace(e, e +
            '            if _fs_enabled:\n'
            '                if not delta_B:\n'
            '                    raise ValueError(\n'
            '                        "factor_policy=fedsonata_cora needs per-client LoRA B updates "\n'
            '                        "(causal-LM matrix mode ab); the selected --method produced none. "\n'
            '                        "Running anyway silently zeroes client_B_states (sever-side B overwrite) "\n'
            '                        "and yields meaningless accuracy."\n'
            '                    )\n'
            '                delta_A, delta_B, _fs_sel = mec_controller.select_factors(\n'
            '                    round_num=round_num, client_id=cid_int,\n'
            '                    a_after=_get_all_A(lora_layers), b_after=_get_all_B(lora_layers),\n'
            '                    delta_a=delta_A, delta_b=(delta_B if delta_B is not None else {}),\n'
            '                    allocated_layers=set(allocated_layers),\n'
            '                )\n'
            '                if _fs_sel is not None:\n'
            '                    _fs_rift_records[cid_int] = _fs_sel\n', 1)

        f = "            payload_by_client = {\n"
        assert f in src
        src = src.replace(f,
            '            _fs_run_extra = ({"all_layer_names": list(server.layer_names),\n'
            '                             "rift_records": dict(_fs_rift_records)} if _fs_enabled else {})\n'
            + f, 1)
        g = ('                download_bytes=float(sum(client_train_metrics[int(cid)].get("download_bytes", 0) for cid in selected)),\n'
             '            )\n')
        assert g in src
        src = src.replace(g,
            '                download_bytes=float(sum(client_train_metrics[int(cid)].get("download_bytes", 0) for cid in selected)),\n'
            '                **_fs_run_extra,\n'
            '            )\n', 1)

        # --- Fix A: FedSONATA (CORA) needs A+B upload (vision matrix_mode=ab) ---
        _fsb = ' or bool(globals().get("_fs_enabled", False))'
        h1 = '    upload_B = method_l in {"fedra", "fedhello", "fedit", "fed-it", "fed_it"} or is_ffa_lora or is_flexlora or is_flora\n'
        assert h1 in src
        src = src.replace(h1, h1 + '    upload_B = bool(upload_B) or bool(globals().get("_fs_enabled", False))\n', 1)
        h2 = '(_dict_tensor_bytes(delta_B) if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else 0)'
        assert h2 in src
        src = src.replace(h2, '(_dict_tensor_bytes(delta_B) if (method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}' + _fsb + ') else 0)', 1)
        h3 = '            if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}:\n'
        assert h3 in src
        src = src.replace(h3, '            if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}' + _fsb + ':\n', 1)
        # --- Fix B: FedSONATA needs B aggregated server-side AND kept client-side ---
        h4 = ('    new_client_B_state = client_B_state\n'
              '    if not upload_B:\n'
              '        new_client_B_state = _get_all_B(lora_layers)\n')
        assert h4 in src
        src = src.replace(h4, '    new_client_B_state = client_B_state\n'
            '    if (not upload_B) or bool(globals().get("_fs_enabled", False)):\n'
            '        new_client_B_state = _get_all_B(lora_layers)\n', 1)
        h5 = '            client_updates_B=(client_updates_B if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None),\n'
        assert h5 in src
        src = src.replace(h5, '            client_updates_B=(client_updates_B if (method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} or bool(globals().get("_fs_enabled", False))) else None),\n', 1)
        h6 = '        global_B_after = server.get_global_B_matrices() if method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} else None\n'
        assert h6 in src
        src = src.replace(h6, '        global_B_after = server.get_global_B_matrices() if (method in {"fedra", "fedhello", "flora", "ffa-lora", "ffalora", "ffa_lora", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"} or bool(globals().get("_fs_enabled", False))) else None\n', 1)
        # --- task 2: log client_updates_B provenance/bytes at aggregation ---
        h7 = '        server.aggregate_matrices(\n'
        assert h7 in src
        src = src.replace(h7, '        if globals().get("_fs_enabled", False):\n'
            '            _fs_b_bytes = sum(int(_t.numel()) * 4 for _u in client_updates_B.values() for _t in _u.values()) if client_updates_B else 0\n'
            '            logger.info(f"round={round_num} fedsonata client_updates_B bytes={_fs_b_bytes} clients={sorted(client_updates_B.keys())} source=rift_selected_delta_B")\n'
            + h7, 1)
        compile(src, "<embedded-qwen-fedsonata>", "exec")
    except Exception as exc:
        print(f"[warn] FedSONATA patch skipped: {exc}")
        return
    _EMBEDDED_TASK_SOURCES["qwen"] = src
    _EMBEDDED_TASK_SOURCES["gsm8k"] = src
    _EMBEDDED_TASK_SOURCES["pubmedqa"] = src


_patch_embedded_task_sources_for_fedsonata()

# PubMedQA task (Step 0): wrapper over the GSM8K causal-LM harness, loaded
# from scripts/embedded_pubmedqa_task.py to keep main.py free of a second
# giant embedded source string.
try:
    _PQA_TASK_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scripts", "embedded_pubmedqa_task.py"
    )
    with open(_PQA_TASK_PATH, encoding="utf-8") as _pqa_f:
        _EMBEDDED_TASK_SOURCES["pubmedqa"] = _pqa_f.read()
except OSError:
    pass


def main(argv=None):
    raw_argv = sys.argv[1:] if argv is None else list(argv)

    def _has_flag(args: List[str], flag: str) -> bool:
        prefix = flag + "="
        return any(a == flag or a.startswith(prefix) for a in args)

    def _get_flag_value(args: List[str], flag: str) -> str | None:
        prefix = flag + "="
        for i, a in enumerate(args):
            if a == flag:
                if i + 1 < len(args):
                    return args[i + 1]
                return None
            if a.startswith(prefix):
                return a[len(prefix) :]
        return None

    def _remove_flag_with_value(args: List[str], flag: str) -> List[str]:
        out: List[str] = []
        prefix = flag + "="
        i = 0
        while i < len(args):
            a = args[i]
            if a == flag:
                i += 2
                continue
            if a.startswith(prefix):
                i += 1
                continue
            out.append(a)
            i += 1
        return out

    def _set_flag_value(args: List[str], flag: str, value: str) -> List[str]:
        return [flag, str(value)] + _remove_flag_with_value(list(args), flag)

    def _normalize_legacy_lora_targets(value: str) -> str:
        s = str(value).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = []
        for p in s.replace(";", ",").split(","):
            p = p.strip().strip("'").strip('"')
            if p:
                parts.append(p)
        return ",".join(parts) if parts else str(value)

    def _translate_legacy_gsm8k_args(args: List[str]) -> List[str]:
        flag_map = {
            "--global_model": "--model_path",
            "--dev_data_path": "--test_data_path",
            "--num_communication_rounds": "--num_rounds",
            "--local_num_epochs": "--num_epochs",
            "--local_learning_rate": "--client_lr",
            "--cutoff_len": "--max_length",
            "--lora_r": "--lora_rank",
            "--output_dir": "--log_dir",
        }
        unsupported_with_values = {
            "--client_selection_strategy",
            "--client_selection_frac",
            "--local_val_set_size",
            "--local_save_steps",
            "--resume_from_checkpoint",
            "--prompt_template_name",
            "--local_ranks",
        }
        unsupported_booleans = {
            "--group_by_length",
            "--stacking",
            "--heter",
            "--zero_padding",
            "--Adalora",
            "--adalora",
            "--full",
        }

        out: List[str] = []
        local_batch_size = None
        local_micro_batch_size = None
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--":
                i += 1
                continue

            if a == "--dataset":
                if i + 1 < len(args):
                    i += 2
                else:
                    i += 1
                continue
            if a.startswith("--dataset="):
                i += 1
                continue

            split_value = None
            flag = a
            if a.startswith("--") and "=" in a:
                flag, split_value = a.split("=", 1)

            def read_value() -> str | None:
                nonlocal i
                if split_value is not None:
                    return split_value
                if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                    i += 1
                    return args[i]
                return None

            if flag == "--local_batch_size":
                local_batch_size = read_value()
                i += 1
                continue
            if flag == "--local_micro_batch_size":
                local_micro_batch_size = read_value()
                i += 1
                continue
            if flag == "--lora_target_modules":
                v = read_value()
                if v is not None:
                    out.extend(["--lora_targets", _normalize_legacy_lora_targets(v)])
                i += 1
                continue
            if flag == "--train_on_inputs":
                v = read_value()
                if v is None or str(v).lower() in {"1", "true", "yes", "y"}:
                    out.append("--train_on_inputs")
                else:
                    out.append("--no-train_on_inputs")
                i += 1
                continue
            if flag in flag_map:
                v = read_value()
                if v is not None:
                    out.extend([flag_map[flag], v])
                i += 1
                continue
            if flag in unsupported_with_values:
                _ = read_value()
                i += 1
                continue
            if flag in unsupported_booleans:
                _ = read_value()
                i += 1
                continue

            out.append(a)
            i += 1

        if local_micro_batch_size is not None:
            out = _set_flag_value(out, "--batch_size", local_micro_batch_size)
            if local_batch_size is not None:
                try:
                    grad_accum = max(1, int(int(local_batch_size) / max(1, int(local_micro_batch_size))))
                    out = _set_flag_value(out, "--grad_accum_steps", str(grad_accum))
                except Exception:
                    pass
        elif local_batch_size is not None:
            out = _set_flag_value(out, "--batch_size", local_batch_size)

        return out

    entry = argparse.ArgumentParser(description="统一 Fed-HARP 实验入口", add_help=True)
    entry.add_argument(
        "--task",
        type=str,
        default="vision",
        choices=["vision", "roberta", "flora", "fedra", "qwen", "gsm8k", "pubmedqa"],
    )
    known, rest = entry.parse_known_args(args=raw_argv)
    rest = _strip_double_dash(rest)

    task_explicit = _has_flag(raw_argv, "--task")
    method_raw = _get_flag_value(raw_argv, "--method")
    dataset_raw = _get_flag_value(raw_argv, "--dataset")
    model_path_raw = _get_flag_value(raw_argv, "--model_path")
    global_model_raw = _get_flag_value(raw_argv, "--global_model")
    data_path_raw = _get_flag_value(raw_argv, "--data_path")
    test_data_path_raw = _get_flag_value(raw_argv, "--test_data_path")
    dev_data_path_raw = _get_flag_value(raw_argv, "--dev_data_path")
    task_raw = _get_flag_value(raw_argv, "--task")

    method = (method_raw or "").strip().lower()
    dataset = (dataset_raw or "").strip().lower()
    task = (task_raw or str(known.task)).strip().lower()

    is_nlp = bool(model_path_raw) or dataset in {"sst2", "qnli"} or _has_flag(raw_argv, "--lora_targets")
    is_vision = dataset in {"cifar10", "cifar100", "chestxray8"} or _has_flag(raw_argv, "--model_name")
    qwen_hint = " ".join(
        str(x or "") for x in [model_path_raw, global_model_raw, data_path_raw, test_data_path_raw, dev_data_path_raw]
    ).lower()
    is_gsm8k = (
        dataset == "gsm8k"
        or "gsm8k" in qwen_hint
        or "qwen" in qwen_hint
        or _has_flag(raw_argv, "--run_mode")
        or _has_flag(raw_argv, "--test_data_path")
        or _has_flag(raw_argv, "--dev_data_path")
        or _has_flag(raw_argv, "--global_model")
        or _has_flag(raw_argv, "--num_communication_rounds")
        or _has_flag(raw_argv, "--local_micro_batch_size")
    )

    if (not task_explicit) and is_gsm8k and not is_vision:
        task = "gsm8k"

    if method in {"flora", "fedra", "flexlora", "flex-lora", "flex_lora", "fedit", "fed-it", "fed_it"}:
        if (not task_explicit) or task in {"vision", "flora", "fedra"}:
            if is_nlp and not is_vision:
                task = "roberta"
            else:
                task = "vision"

    if task == "flora":
        task = "vision"
        if method != "flora":
            rest = ["--method", "flora"] + _remove_flag_with_value(list(rest), "--method")

    if task == "fedra":
        if is_nlp and not is_vision:
            task = "roberta"
        else:
            task = "vision"
        if method != "fedra":
            rest = ["--method", "fedra"] + _remove_flag_with_value(list(rest), "--method")

    if task == "vision":
        return vision_main(rest)
    if task == "roberta":
        return _run_embedded_task("roberta", rest)
    if task == "qwen":
        return _run_embedded_task("qwen", _translate_legacy_gsm8k_args(rest))
    if task == "gsm8k":
        return _run_embedded_task("gsm8k", _translate_legacy_gsm8k_args(rest))
    if task == "pubmedqa":
        return _run_embedded_task("pubmedqa", _translate_legacy_gsm8k_args(rest))
    raise SystemExit(f"未知 task: {task}")


if __name__ == "__main__":
    main()
