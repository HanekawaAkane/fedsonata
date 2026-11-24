"""
Fed-HARP 客户端实现。
处理本地训练、B 矩阵的持久化保存以及 B 矩阵敏感度计算逻辑。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Set, Optional, Tuple
import copy
from models import LoRALinear, get_lora_layer_names


def effective_weight_gradient_proxy(
    current_A: torch.Tensor,
    current_B: torch.Tensor,
    grad_A: Optional[torch.Tensor],
    grad_B: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Map factor gradients to a first-order effective-weight proxy."""
    if grad_A is None and grad_B is None:
        return None
    proxy = torch.zeros(
        (int(current_B.shape[0]), int(current_A.shape[1])),
        device=current_A.device,
        dtype=torch.float32,
    )
    if grad_B is not None:
        proxy = proxy + grad_B.detach().float().to(proxy.device) @ current_A.detach().float().to(proxy.device)
    if grad_A is not None:
        proxy = proxy + current_B.detach().float().to(proxy.device) @ grad_A.detach().float().to(proxy.device)
    return proxy


class FedHarpClient:
    """
    Fed-HARP 联邦学习客户端。
    
    核心功能:
    - 本地持久化 B 矩阵 (永不上传，实现 Native-privacy)
    - 联合训练分配的层 (Heterogeneous Allocation)
    - 计算 B 矩阵敏感度 (L2 Norm) 用于指导服务器分配策略
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        lora_layers: Dict[str, LoRALinear],
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),   
        lr: float = 0.001,
        lora_lr: Optional[float] = None,
        classifier_lr: Optional[float] = None,
        warmup_steps: int = 10,
        warmup_lr: float = 0.0001,
        method: str = "fedharp"
    ):
        """
        初始化客户端。
        
        参数:
            client_id: 客户端唯一 ID
            model: 本地模型实例
            lora_layers: LoRA 层字典
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 运行设备
            lr: 正常训练的学习率
            warmup_steps: B 矩阵对齐的预热步数 (预留参数)
            warmup_lr: 预热阶段的学习率 (预留参数)
        """
        self.client_id = client_id
        self.model = model
        self.lora_layers = lora_layers
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        # 可选的本地 test loader；若提供，则 evaluate() 无参时默认使用本地 test
        self.test_loader = test_loader
        self.device = device
        self.lr = float(lr)
        # D032 separates adapter and classifier optimization.  ``lr`` remains
        # the compatibility fallback for legacy callers.
        self.lora_lr = float(self.lr if lora_lr is None else lora_lr)
        self.classifier_lr = float(
            self.lora_lr if classifier_lr is None else classifier_lr
        )
        if self.lora_lr <= 0.0 or self.classifier_lr <= 0.0:
            raise ValueError("lora_lr and classifier_lr must be positive")
        self.method = "fedharp" if str(method).lower() == "fedanon" else method
        
        # 兼容保留参数
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        
        # 记录上一轮中被冻结的层 (用于潜在的陈旧度控制逻辑)
        self.previously_frozen_layers: Set[str] = set()
        
        # 将模型移动到设备
        self.model.to(device)
        
        # 初始化优化器 (将在每次训练时根据可训练参数重建)
        self.optimizer = None

        # Local training gradient proxy consumed only by the matched
        # loss-aware wire selector after training.
        self.last_effective_gradients: Dict[str, torch.Tensor] = {}

    def _classifier_module(self) -> Optional[nn.Module]:
        """Return the model's task head without assuming a backend layout."""
        module = None
        getter = getattr(self.model, "get_classifier", None)
        if callable(getter):
            try:
                module = getter()
            except Exception:
                module = None
        if module is None:
            for attr in ("head", "classifier", "fc"):
                candidate = getattr(self.model, attr, None)
                if isinstance(candidate, nn.Module):
                    module = candidate
                    break
        if module is None and hasattr(self.model, "backbone"):
            backbone = getattr(self.model, "backbone")
            getter = getattr(backbone, "get_classifier", None)
            if callable(getter):
                try:
                    module = getter()
                except Exception:
                    module = None
            if module is None:
                for attr in ("head", "classifier", "fc"):
                    candidate = getattr(backbone, attr, None)
                    if isinstance(candidate, nn.Module):
                        module = candidate
                        break
        return module if isinstance(module, nn.Module) else None

    def get_classifier_state(self) -> Dict[str, torch.Tensor]:
        module = self._classifier_module()
        if module is None:
            return {}
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    def set_classifier_state(self, state: Dict[str, torch.Tensor]) -> None:
        if not state:
            return
        module = self._classifier_module()
        if module is None:
            raise RuntimeError("model has no classifier/head module")
        current = module.state_dict()
        compatible = {
            k: v.to(device= current[k].device, dtype=current[k].dtype)
            for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)
        }
        module.load_state_dict(compatible, strict=False)
    
    def get_all_A_matrices(self) -> Dict[str, torch.Tensor]:
        """获取所有 A 矩阵 (用于上传给服务器聚合)"""
        return {name: layer.get_A() for name, layer in self.lora_layers.items()}
    
    def get_all_B_matrices(self) -> Dict[str, torch.Tensor]:
        """获取所有 B 矩阵"""
        return {name: layer.get_B() for name, layer in self.lora_layers.items()}
    
    def set_A_matrices(self, A_matrices: Dict[str, torch.Tensor]):
        """设置来自服务器的 A 矩阵"""
        for name, A in A_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_A(A)

    def set_B_matrices(self, B_matrices: Dict[str, torch.Tensor]):
        """设置来自服务器的 B 矩阵"""
        for name, B in B_matrices.items():
            if name in self.lora_layers:
                self.lora_layers[name].set_B(B)

    @staticmethod
    def _slice_or_zero_pad_rank(
        matrix: torch.Tensor,
        target_rank: int,
        rank_dim: int,
    ) -> torch.Tensor:
        """Keep the leading rank directions while preserving a local rank shape."""
        target_rank = max(1, int(target_rank))
        if matrix.ndim != 2:
            raise ValueError(f"LoRA matrix must be two-dimensional, got shape={tuple(matrix.shape)}")

        shape = list(matrix.shape)
        shape[rank_dim] = target_rank
        result = matrix.new_zeros(shape)
        copied_rank = min(int(matrix.shape[rank_dim]), target_rank)
        slices = [slice(None), slice(None)]
        slices[rank_dim] = slice(0, copied_rank)
        result[tuple(slices)] = matrix[tuple(slices)]
        return result

    def prepare_global_matrices_for_rank(
        self,
        A_matrices: Dict[str, torch.Tensor],
        B_matrices: Optional[Dict[str, torch.Tensor]],
        target_rank: Optional[int],
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Prepare the rank-limited A/B payload received by a heterogeneous client."""
        if target_rank is None:
            return A_matrices, B_matrices

        local_A = {
            name: self._slice_or_zero_pad_rank(A, target_rank, rank_dim=0)
            for name, A in A_matrices.items()
        }
        local_B = None
        if B_matrices is not None:
            local_B = {
                name: self._slice_or_zero_pad_rank(B, target_rank, rank_dim=1)
                for name, B in B_matrices.items()
            }
        return local_A, local_B

    def receive_global_matrices(
        self,
        A_matrices: Dict[str, torch.Tensor],
        B_matrices: Optional[Dict[str, torch.Tensor]] = None,
        method: Optional[str] = None
    ):
        method = method or self.method
        method_l = str(method).lower()

        if method_l in {
            "flora",
            "flexlora",
            "flex-lora",
            "flex_lora",
            "ffa-lora",
            "ffalora",
            "ffa_lora",
            "fedit",
            "fed-it",
            "fed_it",
        }:
            if B_matrices is None:
                raise ValueError(f"{method_l} 需要 B_matrices")
            self.set_A_matrices(A_matrices)
            self.set_B_matrices(B_matrices)
            return

        self.set_A_matrices(A_matrices)
        if method_l in {"fedhello", "fedra"} or B_matrices is not None:
            if B_matrices is None:
                raise ValueError(f"{method_l} 需要 B_matrices")
            self.set_B_matrices(B_matrices)

    def flora_merge_and_reset(self, target_rank: int):
        target_rank = max(1, int(target_rank))
        for layer in self.lora_layers.values():
            layer.merge_lora_weights()
            layer.reset_lora(rank=target_rank)
        self.flora_rank = target_rank
    
    def calculate_B_sensitivity(self) -> Dict[str, float]:
        sensitivities = {}
        for name in self.lora_layers.keys():
            sensitivities[name] = 0.0
        return sensitivities

    def _setup_layer_trainability(self, allocated_layers: Set[str]):
        """
        根据分配结果设置层的可训练性。
        当前策略：
        - 被分配的层: A 与 B 都参与训练
        - 未分配的层: A 与 B 都冻结
        """
        for name, layer in self.lora_layers.items():
            if name in allocated_layers:
                # 被分配的层: 同时训练 A 和 B
                layer.unfreeze_all()
            else:
                # 未分配的层: A 和 B 都冻结
                layer.freeze_all()

    def _setup_layer_trainability_B_only(
        self,
        train_B_layers: Set[str],
        train_classifier: bool = False,
    ):
        for param in self.model.parameters():
            param.requires_grad = False

        # FFA-LoRA's normal local update is B-only for the adapter, but its
        # task classifier must still learn the new dataset labels.  FedHARP's
        # temporary LPC correction leaves this disabled.
        if train_classifier:
            classifier = self._classifier_module()
            if classifier is not None:
                for param in classifier.parameters():
                    param.requires_grad = True

        for name, layer in self.lora_layers.items():
            if name in train_B_layers:
                layer.freeze_A()
                layer.unfreeze_B()
            else:
                layer.freeze_all()

    def _setup_layer_trainability_A_only(self, train_A_layers: Set[str]):
        for param in self.model.parameters():
            param.requires_grad = False

        for name, layer in self.lora_layers.items():
            if name in train_A_layers:
                layer.unfreeze_A()
                layer.freeze_B()
            else:
                layer.freeze_all()
    
    def _create_optimizer(self, warmup_mode: bool = False):
        """创建分组优化器。LoRA 因子与任务分类器使用独立学习率。"""
        lora_params = []
        classifier_ids = set()
        classifier_module = self._classifier_module()
        if classifier_module is not None:
            classifier_ids = {id(p) for p in classifier_module.parameters()}
        seen = set()
        for layer in self.lora_layers.values():
            for p in layer.get_trainable_params():
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                lora_params.append(p)

        classifier_params = []
        other_params = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if pid in classifier_ids:
                classifier_params.append(p)
            else:
                other_params.append(p)
        
        if len(lora_params) == 0 and len(classifier_params) == 0 and len(other_params) == 0:
            return None

        lora_lr = float(self.warmup_lr if warmup_mode else self.lora_lr)
        classifier_lr = float(self.warmup_lr if warmup_mode else self.classifier_lr)
        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": lora_lr, "group": "lora"})
        if classifier_params:
            groups.append({"params": classifier_params, "lr": classifier_lr, "group": "classifier"})
        # A future trainable task-specific parameter is treated like an adapter
        # parameter, keeping the protocol deterministic rather than silently
        # falling back to a backend default learning rate.
        if other_params:
            groups.append({"params": other_params, "lr": lora_lr, "group": "other"})
        return optim.Adam(groups)

    def _train_epoch_collect_B_grad_norms(
        self, warmup_mode: bool = False, include_A: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
        grad_a_sum: Dict[str, Optional[torch.Tensor]] = {
            name: None for name in self.lora_layers.keys()
        }
        grad_b_sum: Dict[str, Optional[torch.Tensor]] = {
            name: None for name in self.lora_layers.keys()
        }

        criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(warmup_mode=warmup_mode)

        if self.optimizer is None:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        for _, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            for name, layer in self.lora_layers.items():
                grad_sq = 0.0
                grad_b = layer.lora_B.grad
                grad_a = layer.lora_A.grad
                if grad_b is not None:
                    n_b = torch.norm(grad_b.detach(), p=2).item()
                    grad_sq += float(n_b) * float(n_b)
                if include_A:
                    if grad_a is not None:
                        n_a = torch.norm(grad_a.detach(), p=2).item()
                        grad_sq += float(n_a) * float(n_a)
                if grad_a is not None:
                    g = grad_a.detach()
                    grad_a_sum[name] = g.clone() if grad_a_sum[name] is None else grad_a_sum[name] + g
                if grad_b is not None:
                    g = grad_b.detach()
                    grad_b_sum[name] = g.clone() if grad_b_sum[name] is None else grad_b_sum[name] + g
                if grad_sq <= 0.0:
                    continue
                n = float(grad_sq) ** 0.5
                grad_norm_sq_sum[name] += n * n

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return 0.0, {name: 0.0 for name in self.lora_layers.keys()}

        effective_gradients: Dict[str, torch.Tensor] = {}
        for name, layer in self.lora_layers.items():
            proxy = effective_weight_gradient_proxy(
                layer.get_A().detach(), layer.get_B().detach(),
                grad_a_sum[name], grad_b_sum[name],
            )
            if proxy is not None:
                effective_gradients[name] = proxy / float(max(1, num_batches))
        self._last_effective_gradients = effective_gradients
        grad_sensitivity = {name: (grad_norm_sq_sum[name] / num_batches) ** 0.5 for name in grad_norm_sq_sum.keys()}
        return total_loss / num_batches, grad_sensitivity

    def _train_epoch(self, warmup_mode: bool = False) -> float:
        """
        训练一个 Epoch。
        
        返回:
            平均 Loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer(warmup_mode=warmup_mode)
        
        if self.optimizer is None:
            return 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def local_train(
        self,
        allocated_layers: Set[str],
        num_epochs: int = 10,
        method: Optional[str] = None,
        matrix_update_mode: str = "a",
        adaptive_matrix_threshold: float = 1.25,
    ):
        """
        执行本地训练流程。
        
        参数:
            allocated_layers: 本轮分配给该客户端的层
            num_epochs: 训练轮数
        
        返回:
            delta_A: A 矩阵的更新量字典
            b_sensitivity: B 矩阵的敏感度字典 (用于下一轮分配)
        """
        method = method or self.method
        method_l = str(method).lower()
        matrix_mode = str(matrix_update_mode or "a").lower().replace("-", "_")
        if matrix_mode not in {"a", "b", "ab", "adaptive"}:
            matrix_mode = "a"
        is_fedharp_family = method_l in {"fedharp", "fedharp_r", "fedharp_g", "fedharp_l", "fedharp_a", "fedanon"}
        is_ffa_lora = method_l in {"ffa-lora", "ffalora", "ffa_lora"}
        is_full_matrix_upload = method_l in {"flora", "flexlora", "flex-lora", "flex_lora"}
        fedharp_upload_B = is_fedharp_family and matrix_mode in {"b", "ab", "adaptive"}
        upload_B = fedharp_upload_B or method_l in {
            "fedhello",
            "flora",
            "flexlora",
            "flex-lora",
            "flex_lora",
            "fedra",
            "ffa-lora",
            "ffalora",
            "ffa_lora",
            "fedit",
            "fed-it",
            "fed_it",
        }

        # 记录训练前的 A/B 矩阵，用于计算 delta
        A_before = self.get_all_A_matrices()
        B_before = self.get_all_B_matrices() if upload_B else None
        
        # 设置所有分配层的可训练性
        if is_ffa_lora:
            self._setup_layer_trainability_B_only(
                allocated_layers, train_classifier=True
            )
        else:
            self._setup_layer_trainability(allocated_layers)

        grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
        effective_grad_sum: Dict[str, torch.Tensor] = {}
        effective_grad_steps = 0
        grad_norm_steps = 0
        
        # 正常训练阶段 (联合更新 A 和 B)
        for epoch in range(num_epochs):
            train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(
                warmup_mode=False, include_A=str(method).lower() == "fedhello"
            )
            # 打印每个 Epoch 信息
            print(f"  客户端 {self.client_id}: 训练 Epoch {epoch+1}/{num_epochs}, Loss = {train_loss:.4f}")

            for name, v in grad_sensitivity_epoch.items():
                grad_norm_sq_sum[name] += v * v
            grad_norm_steps += 1
            for name, value in getattr(self, "_last_effective_gradients", {}).items():
                effective_grad_sum[name] = (
                    value.detach().clone()
                    if name not in effective_grad_sum
                    else effective_grad_sum[name] + value.detach()
                )
            effective_grad_steps += 1
        
        A_after = self.get_all_A_matrices()
        B_after = self.get_all_B_matrices() if upload_B else None

        updates_A: Dict[str, torch.Tensor] = {}
        updates_B: Optional[Dict[str, torch.Tensor]] = None

        if method_l == "flora":
            updates_B = {}
            for name in allocated_layers:
                if name in A_after and B_after is not None and name in B_after:
                    updates_A[name] = A_after[name]
                    updates_B[name] = B_after[name]
        elif is_full_matrix_upload:
            updates_B = {}
            for name in allocated_layers:
                if name in A_after and B_after is not None and name in B_after:
                    updates_A[name] = A_after[name]
                    updates_B[name] = B_after[name]
        elif is_ffa_lora:
            updates_B = {}
            if B_before is not None and B_after is not None:
                for name in allocated_layers:
                    if name in B_before and name in B_after:
                        updates_B[name] = B_after[name] - B_before[name]
        elif is_fedharp_family and matrix_mode in {"b", "ab", "adaptive"}:
            updates_B = {}
            for name in allocated_layers:
                delta_a = A_after[name] - A_before[name] if name in A_before and name in A_after else None
                delta_b = (
                    B_after[name] - B_before[name]
                    if B_before is not None and B_after is not None and name in B_before and name in B_after
                    else None
                )
                if matrix_mode == "b":
                    if delta_b is not None:
                        updates_B[name] = delta_b
                    continue
                if matrix_mode == "ab":
                    if delta_a is not None:
                        updates_A[name] = delta_a
                    if delta_b is not None:
                        updates_B[name] = delta_b
                    continue
                if delta_a is None and delta_b is None:
                    continue
                norm_a = float(torch.norm(delta_a).item()) if delta_a is not None else 0.0
                norm_b = float(torch.norm(delta_b).item()) if delta_b is not None else 0.0
                threshold = max(1.0, float(adaptive_matrix_threshold))
                if norm_b > threshold * max(norm_a, 1e-12):
                    if delta_b is not None:
                        updates_B[name] = delta_b
                elif norm_a > threshold * max(norm_b, 1e-12):
                    if delta_a is not None:
                        updates_A[name] = delta_a
                else:
                    if delta_a is not None:
                        updates_A[name] = delta_a
                    if delta_b is not None:
                        updates_B[name] = delta_b
        else:
            for name in allocated_layers:
                if name in A_before and name in A_after:
                    updates_A[name] = A_after[name] - A_before[name]

            if upload_B and B_before is not None and B_after is not None:
                updates_B = {}
                for name in allocated_layers:
                    if name in B_before and name in B_after:
                        updates_B[name] = B_after[name] - B_before[name]
        
        # 更新 previously_frozen_layers (为了支持潜在的未来扩展)
        all_layer_names = set(self.lora_layers.keys())
        self.previously_frozen_layers = all_layer_names - allocated_layers

        if grad_norm_steps > 0:
            b_grad_sensitivity = {name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5 for name in grad_norm_sq_sum.keys()}
        else:
            b_grad_sensitivity = {name: 0.0 for name in self.lora_layers.keys()}

        self.last_effective_gradients = {
            name: value / float(max(1, effective_grad_steps))
            for name, value in effective_grad_sum.items()
        }

        return updates_A, updates_B, b_grad_sensitivity

    def train_B_only(self, train_B_layers: Set[str], num_epochs: int = 1) -> Dict[str, float]:
        requires_grad_state = [(param, bool(param.requires_grad)) for param in self.model.parameters()]
        result = {name: 0.0 for name in self.lora_layers.keys()}
        try:
            self._setup_layer_trainability_B_only(train_B_layers)
            grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
            grad_norm_steps = 0
            for epoch in range(num_epochs):
                train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(
                    warmup_mode=False
                )
                print(
                    f"  Client {self.client_id}: B-only correction Epoch {epoch+1}/{num_epochs}, "
                    f"Loss = {train_loss:.4f}"
                )
                for name, value in grad_sensitivity_epoch.items():
                    grad_norm_sq_sum[name] += value * value
                grad_norm_steps += 1
            if grad_norm_steps > 0:
                result = {
                    name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5
                    for name in grad_norm_sq_sum.keys()
                }
        finally:
            for param, requires_grad in requires_grad_state:
                param.requires_grad = requires_grad
            self.optimizer = None
        return result

    def train_A_only(self, train_A_layers: Set[str], num_epochs: int = 1) -> Dict[str, float]:
        requires_grad_state = [(param, bool(param.requires_grad)) for param in self.model.parameters()]
        result = {name: 0.0 for name in self.lora_layers.keys()}
        try:
            self._setup_layer_trainability_A_only(train_A_layers)
            grad_norm_sq_sum = {name: 0.0 for name in self.lora_layers.keys()}
            grad_norm_steps = 0
            for epoch in range(num_epochs):
                train_loss, grad_sensitivity_epoch = self._train_epoch_collect_B_grad_norms(
                    warmup_mode=False,
                    include_A=True,
                )
                print(
                    f"  Client {self.client_id}: A-only correction Epoch {epoch+1}/{num_epochs}, "
                    f"Loss = {train_loss:.4f}"
                )
                for name, value in grad_sensitivity_epoch.items():
                    grad_norm_sq_sum[name] += value * value
                grad_norm_steps += 1
            if grad_norm_steps > 0:
                result = {
                    name: (grad_norm_sq_sum[name] / grad_norm_steps) ** 0.5
                    for name in grad_norm_sq_sum.keys()
                }
        finally:
            for param, requires_grad in requires_grad_state:
                param.requires_grad = requires_grad
            self.optimizer = None
        return result

    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        在测试集上评估模型性能。
        如果未显式传入 test_loader，则优先使用客户端保存的本地 'test' 数据集。
        """
        if test_loader is None:
            if self.test_loader is None:
                raise ValueError("未提供测试集 DataLoader，且客户端未配置本地 test_loader")
            test_loader = self.test_loader
            
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
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


FedAnonClient = FedHarpClient
