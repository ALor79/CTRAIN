from diffai.crelu import crelu_boxy as _crelu_boxy
from diffai.crelu import crelu_switch as _crelu_switch
from diffai.crelu import crelu_smooth as _crelu_smooth
from CTRAIN.bound.zonotope import HybridZonotope


def _wrap(fn):
    """Wrap a tensor-level crelu function to accept/return HybridZonotope."""
    def wrapped(dom):
        new_head, new_beta, new_errors = fn(dom.head, dom.beta, dom.errors)
        return HybridZonotope(new_head, new_beta, new_errors)
    wrapped.__name__ = fn.__name__
    return wrapped


crelu_boxy   = _wrap(_crelu_boxy)
crelu_switch = _wrap(_crelu_switch)
crelu_smooth = _wrap(_crelu_smooth)

RELU_TRANSFORMERS = {
    'boxy':   crelu_boxy,
    'switch': crelu_switch,
    'smooth': crelu_smooth,
}
