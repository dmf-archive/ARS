from dataclasses import dataclass, field
from importlib import import_module
from enum import Enum, auto, Flag
from typing import Any, Callable, Optional, Set
import torch
from utils.nn import disable_running_stats, enable_running_stats

class Capability(Flag):
    NONE = 0
    REQUIRES_CLOSURE = auto()      # 需要 step(closure)
    REQUIRES_LOSS = auto()         # step 接收 loss 张量
    REQUIRES_MODEL = auto()        # 初始化需要 model 实例
    BN_PROTECTION = auto()         # 闭包期间需要保护 BN
    SECOND_ORDER = auto()          # 需要 create_graph=True
    PI_AWARE = auto()              # 支持 PI 信号注入

class GroupingStrategy(Enum):
    NONE = auto()
    MUON = auto()    # 2D+ 权重使用 'use_muon' 标志位
    RMSUON = auto()  # 2D+ 权重使用 'is_rmsuon_group' 标志位

@dataclass
class OptimizerMetadata:
    cls_name: str
    module_name: str
    capabilities: Capability = Capability.NONE
    grouping: GroupingStrategy = GroupingStrategy.NONE
    expects_param_groups: bool = False
    extra_config_keys: list[str] = field(default_factory=list)

OPTIMIZER_REGISTRY: dict[str, OptimizerMetadata] = {
    "AdamW": OptimizerMetadata(
        cls_name="AdamW", module_name="torch.optim", 
        expects_param_groups=True
    ),
    "AdaHessian": OptimizerMetadata(
        cls_name="Adahessian", module_name="ada_hessian", 
        capabilities=Capability.SECOND_ORDER
    ),
    "Muon": OptimizerMetadata(
        cls_name="SingleDeviceMuonWithAuxAdam", module_name="muon",
        grouping=GroupingStrategy.MUON, expects_param_groups=True,
        extra_config_keys=["momentum", "betas", "eps", "ns_steps"]
    ),
    "RMSuon": OptimizerMetadata(
        cls_name="RMSuon", module_name="rmsuon", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        extra_config_keys=["betas", "eps"]
    ),
    "AdaRMSuon": OptimizerMetadata(
        cls_name="AdaRMSuon", module_name="ada_rmsuon", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        extra_config_keys=["betas", "eps"]
    ),
    "ARS": OptimizerMetadata(
        cls_name="ARSOptimizer", module_name="ars", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        capabilities=Capability.REQUIRES_CLOSURE | Capability.BN_PROTECTION,
        extra_config_keys=["betas", "eps", "rho", "k", "alpha"]
    ),
    "ARG": OptimizerMetadata(
        cls_name="ARGOptimizer", module_name="arg", 
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        capabilities=Capability.REQUIRES_CLOSURE | Capability.SECOND_ORDER | Capability.BN_PROTECTION,
        extra_config_keys=["betas", "eps", "rho"]
    ),
    "KFAC": OptimizerMetadata(
        cls_name="KFACOptimizer", module_name="kfac", 
        capabilities=Capability.REQUIRES_MODEL
    ),
    "DiagHadron": OptimizerMetadata(
        cls_name="DiagHadron", module_name="diag_hadron", 
        capabilities=Capability.REQUIRES_MODEL, expects_param_groups=True
    ),
    "LARS": OptimizerMetadata(
        cls_name="LARSOptimizer", module_name="lars",
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        capabilities=Capability.REQUIRES_CLOSURE | Capability.BN_PROTECTION,
        extra_config_keys=["betas", "eps", "rho", "k", "alpha", "adaptive_alpha"]
    ),
    "ARS2-Neo": OptimizerMetadata(
        cls_name="SingleDeviceARS2Neo", module_name="ars2_neo",
        grouping=GroupingStrategy.RMSUON, expects_param_groups=True,
        capabilities=Capability.REQUIRES_CLOSURE | Capability.BN_PROTECTION,
        extra_config_keys=[
            "betas", "eps", "rho", "k", "alpha", "ns_steps",
            "adaptive_sync", "asi_enabled", "adaptive_beta", "adaptive_lambda", "adaptive_gamma"
        ]
    ),
    "AdaMuon": OptimizerMetadata(
        cls_name="AdaMuon", module_name="adamuon",
        grouping=GroupingStrategy.MUON, expects_param_groups=True,
        extra_config_keys=["betas", "eps", "ns_steps"]
    ),
}

class SmartOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, metadata: OptimizerMetadata, model: torch.nn.Module, criterion: Any, device: torch.device):
        self.optimizer = optimizer
        self.metadata = metadata
        self.model = model
        self.criterion = criterion
        self.device = device
        self.name = metadata.cls_name
        
        # 暴露给外部的标签
        self.tags = {
            "accepts_pi_signal": Capability.PI_AWARE in metadata.capabilities,
            "requires_second_order": Capability.SECOND_ORDER in metadata.capabilities
        }
        
        self._step_logits = None
        self._step_loss = None

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def diagnostics(self) -> dict[str, Any]:
        return getattr(self.optimizer, 'diagnostics', {})

    def _base_closure(self, train_fn: Callable, batch: Any) -> torch.Tensor:
        lgt, ls = train_fn(
            model=self.model, batch=batch, criterion=self.criterion,
            device=self.device,
            needs_second_order=Capability.SECOND_ORDER in self.metadata.capabilities
        )
        # 缓存第一次调用的结果用于指标统计
        if self._step_logits is None: self._step_logits = lgt
        if self._step_loss is None: self._step_loss = ls
        return ls

    def step(self, batch: Any, train_fn: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        self._step_logits, self._step_loss = None, None
        self.optimizer.zero_grad()

        if Capability.REQUIRES_CLOSURE in self.metadata.capabilities:
            if Capability.BN_PROTECTION in self.metadata.capabilities:
                call_count = 0
                def protected_closure():
                    nonlocal call_count
                    if call_count == 0: enable_running_stats(self.model)
                    else: disable_running_stats(self.model)
                    res = self._base_closure(train_fn, batch)
                    call_count += 1
                    return res
                step_output = self.optimizer.step(protected_closure)
                enable_running_stats(self.model) # 确保最后恢复
            else:
                step_output = self.optimizer.step(lambda: self._base_closure(train_fn, batch))
            
            logits, loss = self._step_logits, self._step_loss
            if loss is None and isinstance(step_output, torch.Tensor):
                loss = step_output
        else:
            logits, loss = train_fn(
                model=self.model, batch=batch, criterion=self.criterion,
                device=self.device,
                needs_second_order=Capability.SECOND_ORDER in self.metadata.capabilities
            )
            if Capability.REQUIRES_LOSS in self.metadata.capabilities:
                self.optimizer.step(loss)
            else:
                loss.backward(create_graph=Capability.SECOND_ORDER in self.metadata.capabilities)
                self.optimizer.step()
        
        return logits, loss

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

def _import_optimizer(module_name: str, class_name: str) -> type[torch.optim.Optimizer]:
    if module_name == "torch.optim":
        return getattr(torch.optim, class_name)
    module = import_module(f".{module_name}", package="optimizer")
    return getattr(module, class_name)

def _create_specialized_param_groups(params: list[torch.nn.Parameter], meta: OptimizerMetadata, config: dict) -> list[dict]:
    is_special = lambda p: p.ndim >= 2 and max(p.shape) < 10000
    flag_name = "use_muon" if meta.grouping == GroupingStrategy.MUON else "is_rmsuon_group"
    
    special_params = [p for p in params if p.requires_grad and is_special(p)]
    adam_params = [p for p in params if p.requires_grad and not is_special(p)]
    
    groups = []
    if special_params:
        grp = {
            'params': special_params,
            flag_name: True,
            'lr': config.get("lr", 1e-3),
            'weight_decay': config.get("weight_decay", 0.1),
        }
        for k in meta.extra_config_keys:
            if k in config: grp[k] = config[k]
        groups.append(grp)
        
    if adam_params:
        groups.append({
            'params': adam_params,
            flag_name: False,
            'lr': config.get("adam_lr", config.get("lr", 1e-3)),
            'betas': config.get("adam_betas", (0.9, 0.999)),
            'eps': config.get("adam_eps", 1e-8),
            'weight_decay': config.get("adam_weight_decay", 0.01),
        })
    return groups

def get_optimizer(name: str, params: list[dict], model: torch.nn.Module, criterion: Any, device: torch.device, **config) -> SmartOptimizer:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")

    meta = OPTIMIZER_REGISTRY[name]
    opt_cls = _import_optimizer(meta.module_name, meta.cls_name)
    
    opt_config = config.copy()
    if meta.grouping != GroupingStrategy.NONE:
        flag_name = "use_muon" if meta.grouping == GroupingStrategy.MUON else "is_rmsuon_group"
        # 优先尊重手动分组
        if any(flag_name in g for g in params):
            init_params = params
        else:
            all_params = [p for g in params for p in g['params']]
            init_params = _create_specialized_param_groups(all_params, meta, opt_config)
        
        # 清理冗余配置
        for key in ["adam_lr", "adam_betas", "adam_eps", "adam_weight_decay"]:
            opt_config.pop(key, None)
    elif meta.expects_param_groups:
        init_params = params
    else:
        init_params = next(iter(params))['params']
    
    if Capability.REQUIRES_MODEL in meta.capabilities:
        optimizer = opt_cls(model, **opt_config)
    else:
        optimizer = opt_cls(init_params, **opt_config)

    return SmartOptimizer(optimizer, meta, model, criterion, device)
