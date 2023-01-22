#

# a model

__all__ = [
    "ZModelConf0", "ZModel0",
]

from collections import Counter
from .core import ZModelConf, ZModel, node_reg, ZMediator

class ZModelConf0(ZModelConf):
    def __init__(self):
        super().__init__()

@node_reg(ZModelConf0)
class ZModel0(ZModel):
    def __init__(self, conf: ZModelConf0):
        super().__init__(conf)

    @property
    def mydec(self):  # shortcut for decoder
        if hasattr(self, 'Milm'):
            return self.Milm
        elif hasattr(self, 'Mslm'):
            return self.Mslm
        else:
            return None
