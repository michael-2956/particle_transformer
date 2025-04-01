import torch
import torch.nn.functional as F

FOCAL_LOSS_GAMMA = 2

def focal_loss(inputs, targets, alpha=1.0, gamma=FOCAL_LOSS_GAMMA, reduction='mean'):
    targets = targets.view(-1, 1)

    log_probs = F.log_softmax(inputs, dim=1)
    log_p_t = log_probs.gather(1, targets).view(-1)
    
    probs = torch.exp(log_probs)
    # gathers probabilities of true class according to target
    p_t = probs.gather(1, targets).view(-1)

    loss = -alpha * (1 - p_t) ** gamma * log_p_t

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss


def get_loss_focal(data_config, **kwargs):
    print(f"Focal loss Gamma: {FOCAL_LOSS_GAMMA}")
    return focal_loss


def inverse_focal_loss(inputs, targets, alpha=1.0, gamma=FOCAL_LOSS_GAMMA, reduction='mean'):
    targets = targets.view(-1, 1)

    log_probs = F.log_softmax(inputs, dim=1)
    log_p_t = log_probs.gather(1, targets).view(-1)
    
    probs = torch.exp(log_probs)
    # gathers probabilities of true class according to target
    p_t = probs.gather(1, targets).view(-1)

    # change here
    loss = -alpha * p_t ** gamma * log_p_t

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss


def get_loss_inverse_focal(data_config, **kwargs):
    print(f"Inverse focal loss Gamma: {FOCAL_LOSS_GAMMA}")
    return inverse_focal_loss


NO_OUTLIER_CE_THR = 0.96

def nooutlier_cross_entropy_loss(inputs, targets, threshold=NO_OUTLIER_CE_THR, reduction='mean'):
    targets = targets.view(-1, 1)

    log_probs = F.log_softmax(inputs, dim=1)
    probs = torch.exp(log_probs)
    p_t = probs.gather(1, targets).view(-1)

    loss = -log_probs.gather(1, targets).view(-1)

    valid_mask = p_t >= threshold
    filtered_loss = loss[valid_mask]
    
    # If no samples meet the threshold, return 0
    if filtered_loss.numel() == 0:
        return nooutlier_cross_entropy_loss(
            inputs, targets, threshold=threshold/2, reduction=reduction
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
