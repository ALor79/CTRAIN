import torch
import torch.nn.functional as F
from CTRAIN.bound.zonotope import HybridZonotope


def _bounds(dom):
    """Compute element-wise lower and upper bounds from the zonotope."""
    spread = torch.zeros_like(dom.head)
    if dom.errors is not None:
        spread = spread + dom.errors.abs().sum(0)
    if dom.beta is not None:
        spread = spread + dom.beta
    return dom.head - spread, dom.head + spread


# ------------------------------------------------------------------
# crelu_boxy
# ------------------------------------------------------------------

def crelu_boxy(dom):
    """
    Boxy ReLU transformer.
    Ported one-to-one from: diffai/ai.py::creluBoxy (lines 42-68)

    For each neuron:
      - fully negative (ub <= 0):  zero out head, beta, errors
      - fully positive (lb >= 0):  keep head, beta, errors unchanged
      - crossing (lb < 0 < ub):   collapse to box [0, ub], discard error terms
                                   new center = ub/2, new beta = ub/2
    """
    if dom.beta is None and dom.errors is None:
        return HybridZonotope(F.relu(dom.head), None, None)

    mn, mx = _bounds(dom)

    # diffai/ai.py:61
    should_box = mn.lt(0) & mx.gt(0)   # crossing neurons
    # diffai/ai.py:62
    gtz = dom.head.gt(0)               # center is positive

    # diffai/ai.py:63
    mx_half = mx / 2

    # diffai/ai.py:64 — newhead (rewritten with torch.where instead of ifThenElse)
    newhead = torch.where(should_box, mx_half,
              torch.where(gtz, dom.head, torch.zeros_like(dom.head)))

    # diffai/ai.py:65 — newbeta
    zbet = dom.beta if dom.beta is not None else torch.zeros_like(dom.head)
    newbeta = torch.where(should_box, mx_half,
              torch.where(gtz, zbet, torch.zeros_like(zbet)))

    # diffai/ai.py:66 — newerr
    if dom.errors is not None:
        keep = (~should_box & gtz).float().unsqueeze(0)
        newerr = dom.errors * keep
    else:
        newerr = None

    return HybridZonotope(newhead, newbeta, newerr)


# ------------------------------------------------------------------
# crelu_switch
# ------------------------------------------------------------------

def crelu_switch(dom):
    """
    Switch ReLU transformer.
    Ported one-to-one from: diffai/ai.py::creluSwitch (lines 100-132)

    Improves on crelu_boxy for crossing neurons by choosing between two
    strategies based on which is tighter:

      - lean-negative (|lb| > ub):  collapse to box [0, ub]  (same as boxy)
      - lean-positive (|lb| <= ub): shift head up by |lb|/2, keep error terms
                                    (tighter — preserves correlations)
    """
    if dom.beta is None and dom.errors is None:
        return HybridZonotope(F.relu(dom.head), None, None)

    mn, mx = _bounds(dom)

    # diffai/ai.py:120
    should_box = mn.lt(0) & mx.gt(0)
    # diffai/ai.py:121
    gtz = dom.head.gt(0)

    # diffai/ai.py:123-124 — mn is negated, then should_boxer computed
    abs_lb = (-mn).clamp(min=0)
    should_boxer = should_box & abs_lb.gt(mx)

    # diffai/ai.py:126 — mn /= 2 (here as half_abs_lb)
    half_abs_lb = abs_lb / 2
    mx_half = mx / 2
    zbet = dom.beta if dom.beta is not None else torch.zeros_like(dom.head)

    # diffai/ai.py:127 — newhead
    newhead = torch.where(
        should_box,
        torch.where(should_boxer, mx_half, dom.head + half_abs_lb),
        torch.where(gtz, dom.head, torch.zeros_like(dom.head))
    )

    # diffai/ai.py:129 — newbeta
    newbeta = torch.where(
        should_box,
        torch.where(should_boxer, mx_half, half_abs_lb + zbet),
        torch.where(gtz, zbet, torch.zeros_like(zbet))
    )

    # diffai/ai.py:130 — newerr
    if dom.errors is not None:
        keep = ((~should_box & gtz) | (should_box & ~should_boxer)).float().unsqueeze(0)
        newerr = dom.errors * keep
    else:
        newerr = None

    return HybridZonotope(newhead, newbeta, newerr)


# ------------------------------------------------------------------
# crelu_smooth
# ------------------------------------------------------------------

def crelu_smooth(dom):
    """
    Smooth ReLU transformer.
    Ported one-to-one from: diffai/ai.py::creluSmooth (lines 134-177)

    Instead of a hard switch between strategies, interpolates continuously
    between the boxy strategy (B) and the lean-positive strategy (S):

        t = |lb| / (ub + |lb| + eps)   in [0, 1]

        new = (1 - t) * S  +  t * B

    t close to 0: mostly lean-positive (keep errors, shift head)
    t close to 1: mostly boxy (collapse to interval)

    Benefit: no discontinuity at the strategy boundary -> better gradient flow.
    """
    if dom.beta is None and dom.errors is None:
        return HybridZonotope(F.relu(dom.head), None, None)

    mn, mx = _bounds(dom)

    # diffai/ai.py:155-156
    nmn = F.relu(-mn)
    mmx = F.relu(mx)

    # diffai/ai.py:168-169
    eps = 1e-4
    t = nmn / (mmx + nmn + eps)

    # diffai/ai.py:171
    shouldnt_zero = mx.gt(0).float()

    zbet = dom.beta if dom.beta is not None else torch.zeros_like(dom.head)

    # diffai/ai.py:157-160 — strategy S (lean-positive)
    headS = dom.head + nmn / 2
    betaS = zbet + nmn / 2

    # diffai/ai.py:162-166 — strategy B (boxy)
    headB = mmx / 2
    betaB = mmx / 2

    # diffai/ai.py:173-175
    newhead = shouldnt_zero * ((1 - t) * headS + t * headB)
    newbeta = shouldnt_zero * ((1 - t) * betaS + t * betaB)

    if dom.errors is not None:
        # diffai/ai.py:175 — newerr (errors scaled by S weight, zeroed out by B)
        newerr = shouldnt_zero.unsqueeze(0) * (1 - t).unsqueeze(0) * dom.errors
    else:
        newerr = None

    return HybridZonotope(newhead, newbeta, newerr)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

RELU_TRANSFORMERS = {
    'boxy':   crelu_boxy,
    'switch': crelu_switch,
    'smooth': crelu_smooth,
}
