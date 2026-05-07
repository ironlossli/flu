from pathlib import Path
from typing import Union, Any, Dict

import modify_yaml


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 YAML 配置为字典。不存在则抛 FileNotFoundError。
    返回 {} 表示空文件或仅注释。
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = modify_yaml.safe_load(f)
    return data or {}


def write_yaml(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    将字典写入 YAML（便于保存最终运行配置等）。
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        modify_yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
