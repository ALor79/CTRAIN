import torch
import torch.nn.functional as F
from CTRAIN.bound import bound_ibp, bound_zonotope


def get_diffai_loss(
    hardened_model,
    ptb,
    data,
    target,
    n_classes,
    criterion=None,
    return_bounds=False,
    return_stats=True,
    bound_method='ibp',
    relu_transformer='boxy',
    use_errors=False,
):
    """
    Compute the DiffAI certified loss.

    Args:
        hardened_model:   auto_LiRPA BoundedModule (used for IBP) or CTRAINWrapper
                          whose original_model is walked directly (used for zonotope).
        ptb:              PerturbationLpNorm. For zonotope mode, must have x_L/x_U set.
        data:             Input tensor [batch, *spatial].
        target:           Class labels [batch].
        n_classes:        Number of output classes.
        criterion:        Unused — kept for API compatibility.
        return_bounds:    If True, include (lb, ub) in the return tuple.
        return_stats:     If True, include robust_err in the return tuple.
        bound_method:     'ibp' (default) or 'zonotope'.
        relu_transformer: For zonotope mode — 'boxy', 'switch', or 'smooth'.
        use_errors:       For zonotope mode — whether to use explicit error terms.

    Returns:
        Tuple starting with certified_loss, optionally followed by
        (lb, ub) if return_bounds=True, and robust_err if return_stats=True.
    """
    if bound_method == 'ibp':
        lb, ub = bound_ibp(
            model=hardened_model,
            ptb=ptb,
            data=data,
            target=target,
            n_classes=n_classes,
        )
    elif bound_method == 'zonotope':
        lb, ub = bound_zonotope(
            model=hardened_model,
            ptb=ptb,
            data=data,
            target=target,
            n_classes=n_classes,
            relu_transformer=relu_transformer,
            use_errors=use_errors,
        )
    else:
        raise ValueError(f"Unknown bound_method '{bound_method}'. Choose 'ibp' or 'zonotope'.")

    # lb shape: [batch, n_classes-1] — margin of correct class over each other class
    worst_margin = (-lb).max(dim=1).values        # [batch] — most violated margin
    certified_loss = F.softplus(worst_margin).mean()

    return_tuple = (certified_loss,)
    if return_bounds:
        return_tuple = return_tuple + (lb, ub)
    if return_stats:
        robust_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)
        return_tuple = return_tuple + (robust_err,)
    return return_tuple
