# model_registry.py
import importlib
import inspect
from typing import Callable, Dict, Optional, Any

_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_builder(name: str):
    """Decorator to register a model builder by name."""
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn
    return _wrap


def get_builder(name: str) -> Callable[..., Any]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model builder: {name}")
    return _REGISTRY[name]


def _import_from_string(spec: str) -> Any:
    """
    Import an object from a string path.
    Supports 'pkg.mod:obj' or 'pkg.mod.obj'.
    """
    if ":" in spec:
        mod, obj = spec.split(":", 1)
    else:
        parts = spec.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(f"Invalid import spec: {spec}")
        mod, obj = parts
    module = importlib.import_module(mod)
    return getattr(module, obj)


def _call_builder(builder: Callable[..., Any], model_cfg: dict, data_cfg: dict, train_cfg: dict):
    """
    Try common call signatures in order:
      (model_cfg, data_cfg, train_cfg) -> model
      (model_cfg, data_cfg) -> model
      (model_cfg) -> model
      () -> model
    """
    for args in [(model_cfg, data_cfg, train_cfg), (model_cfg, data_cfg), (model_cfg,), tuple()]:
        try:
            sig = inspect.signature(builder)
            # quick arity guard to reduce noisy TypeErrors
            if len(args) < sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
                               for p in sig.parameters.values()):
                continue
            return builder(*args)
        except TypeError:
            continue
    # last attempt, call directly and let errors bubble
    return builder(model_cfg, data_cfg, train_cfg)


def build_model_from_config(model_cfg: dict, data_cfg: dict, train_cfg: dict):
    """
    Build a model using either:
      - model_cfg['name'] resolved via registry, or
      - model_cfg['builder'] as a Python import path to a function or class.
    If the imported object is a class, instantiate it with (model_cfg) or ().
    """
    if not isinstance(model_cfg, dict):
        raise ValueError("model_cfg must be a dict")

    # 1) Dynamic builder path has priority if provided
    builder_obj: Optional[Any] = None
    if "builder" in model_cfg and model_cfg["builder"]:
        builder_obj = _import_from_string(str(model_cfg["builder"]))
        if inspect.isclass(builder_obj):
            # Prefer (model_cfg) if accepted, else default ctor
            try:
                return builder_obj(model_cfg)
            except Exception:
                return builder_obj()
        if callable(builder_obj):
            return _call_builder(builder_obj, model_cfg, data_cfg, train_cfg)
        raise TypeError(f"Imported builder is not callable or class: {model_cfg['builder']}")

    # 2) Registry by name
    name = str(model_cfg.get("name") or "").strip()
    if not name:
        raise ValueError("model_cfg requires 'name' or 'builder'")
    builder_fn = get_builder(name)
    return _call_builder(builder_fn, model_cfg, data_cfg, train_cfg)
