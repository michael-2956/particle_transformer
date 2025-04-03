import torch
import torch.nn.functional as F

FOCAL_LOSS_ALPHA = 1 - 0.92579  # for mirror tagging on iteration 7 splitter
FOCAL_LOSS_GAMMA = 2.0

def focal_loss(inputs, targets, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, reduction='mean'):
    targets = targets.view(-1, 1).long()

    log_probs = F.log_softmax(inputs, dim=1)
    log_p_t = log_probs.gather(1, targets).view(-1)

    probs = log_probs.exp()
    # gathers probabilities of true class according to target
    p_t = probs.gather(1, targets).view(-1)

    # Build alpha_t so that alpha applies to label=1, (1-alpha) applies to label=0
    alpha_t = alpha * (targets.squeeze() == 1).float() + (1 - alpha) * (targets.squeeze() == 0).float()

    focal_factor = (1 - p_t) ** gamma
    loss = -alpha_t * focal_factor * log_p_t

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    
    return loss


def get_loss_focal(data_config, **kwargs):
    print(f"Focal loss Alpha: {FOCAL_LOSS_ALPHA}")
    print(f"Focal loss Gamma: {FOCAL_LOSS_GAMMA}")
    return focal_loss


def inverse_focal_loss(inputs, targets, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, reduction='mean'):
    targets = targets.view(-1, 1).long()

    log_probs = F.log_softmax(inputs, dim=1)
    log_p_t = log_probs.gather(1, targets).view(-1)

    probs = log_probs.exp()
    # gathers probabilities of true class according to target
    p_t = probs.gather(1, targets).view(-1)

    # Build alpha_t so that alpha applies to label=1, (1-alpha) applies to label=0
    alpha_t = alpha * (targets.squeeze() == 1).float() + (1 - alpha) * (targets.squeeze() == 0).float()

    inverse_focal_factor = p_t ** gamma
    loss = -alpha_t * inverse_focal_factor * log_p_t

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    
    return loss


def get_loss_inverse_focal(data_config, **kwargs):
    print(f"Inverse focal loss Alpha: {FOCAL_LOSS_ALPHA}")
    print(f"Inverse focal loss Gamma: {FOCAL_LOSS_GAMMA}")
    return inverse_focal_loss


NO_OUTLIER_CE_THR = 0.5

def nooutlier_cross_entropy_loss(inputs, targets, threshold=NO_OUTLIER_CE_THR, reduction='mean'):
    targets = targets.view(-1, 1)

    log_probs = F.log_softmax(inputs, dim=1)
    probs = torch.exp(log_probs)
    p_t = probs.gather(1, targets).view(-1)

    loss = -log_probs.gather(1, targets).view(-1)

    valid_mask = p_t >= threshold
    # ensure at least 90% of the data is above threshold
    if valid_mask.float().mean().cpu().item() < 0.9:
        valid_mask = p_t >= 0  # if not, act as cross entropy

    filtered_loss = loss[valid_mask]
    
    # If no samples meet the threshold, return loss for threshold=0
    if filtered_loss.numel() == 0:
        return nooutlier_cross_entropy_loss(
            inputs, targets, threshold=0, reduction=reduction
        )
    
    if reduction == 'mean':
        return filtered_loss.mean()
    elif reduction == 'sum':
        return filtered_loss.sum()
    elif reduction == 'none':
        return filtered_loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

def get_loss_nooutlier_cross_entropy(data_config, **kwargs):
    print(f"No outlier cross entropy threshold: {NO_OUTLIER_CE_THR}")
    return nooutlier_cross_entropy_loss
