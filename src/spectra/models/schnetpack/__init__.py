import sys as _sys
_sys.modules.setdefault("schnetpack", _sys.modules[__name__])  # 兜底：本地包注册为顶层名

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorboard")

# from . import transform    # 避免 matscipy
from . import properties
from . import data
# from . import datasets     # 避免 md22 的 NameError
from . import atomistic
from . import representation
from . import interfaces
from . import nn
# from . import train        # 关键：不要在 init 时加载（会需要 torch_ema）
from . import model
from .units import *
from .task import *
from . import md

__version__ = "2.1.1"
