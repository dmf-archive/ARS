def get_optimizer(name: str, params, **config):
    """按需延迟加载优化器，避免不必要的依赖"""
    if name == "AdamW":
        import torch
        return torch.optim.AdamW(params, **config)
    elif name == "AdaHessian":
        from .ada_hessian import Adahessian
        return Adahessian(params, **config)
    elif name == "AdaFisher":
        from .ada_fisher import AdaFisher
        # AdaFisher需要模型对象作为第一个参数，而不是params
        # 这里需要特殊处理，因为AdaFisher的构造函数签名不同
        if "model" in config:
            model = config.pop("model")
            return AdaFisher(model, **config)
        else:
            raise ValueError("AdaFisher optimizer requires 'model' parameter in config")
    elif name == "F3EO":
        from .F3EO import F3EO
        return F3EO(params, **config)
    elif name == "F3EL":
        from .F3EL import F3EL
        return F3EL(params, **config)
    elif name == "F3EW":
        from .F3EW import F3EW
        return F3EW(params, **config)
    elif name == "F3EPI":
        from .F3EPI import F3EPI
        return F3EPI(params, **config)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
