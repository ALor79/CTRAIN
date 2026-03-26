import torch.nn.functional as F
from CTRAIN.bound import bound_ibp

def get_diffai_loss(hardened_model, ptb, data, target, n_classes,
                    return_bounds=False, return_stats=True):

        ilb, iub = bound_ibp(
        model=hardened_model,
        ptb=ptb,
        data=data,
        target=target,
        n_classes=n_classes,
        # bound_upper=True if loss_fusion else False,
        # loss_fusion=loss_fusion,
    )
    # lb shape: (batch, n_classes-1) — margins relative to correct class
    worst_margin = (-lb).max(dim=1).values   # (batch,)
    certified_loss = F.softplus(worst_margin).mean()
    return_tuple = (certified_loss,)
    if return_bounds:
        return_tuple = return_tuple + (robust_err,)
    if return_stats:
        robust_err = torch.sum((lb < 0).any(dim=1)).item() / data.size(0)
        return_tuple = return_tuple + (robust_err,)
    return return_tuple