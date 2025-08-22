import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_loss(
    preds,
    targets,
    mask=None,
    radius_loss_fn=None,
    direction_loss_fn=None,
    class_loss_fn=None,
    target_radius_log=True,
    vector_class=None,
):
    predicted_radius = preds["radius"]  #
    predicted_direction = preds["direction"]
    predicted_class = preds["class_l"]
    target_class = targets[:, [-1]].long()
    target_direction = targets[:, 1:-1]
    target_radius = targets[:, [0]]

    if mask is not None:
        predicted_radius = predicted_radius[mask]
        predicted_direction = predicted_direction[mask]
        predicted_class = predicted_class[mask]
        target_radius = target_radius[mask]
        target_direction = target_direction[mask]
        target_class = target_class[mask]

    # Only compute vector loss on branch points...
    if vector_class is not None:
        vector_mask = target_class == vector_class
        vector_mask = vector_mask.view(-1)
        predicted_radius = predicted_radius[vector_mask]
        predicted_direction = predicted_direction[vector_mask]
        target_radius = target_radius[vector_mask]
        target_direction = target_direction[vector_mask]

    if target_radius_log:
        target_radius = torch.log(target_radius)

    losses = {}

    losses["radius"] = radius_loss_fn(predicted_radius.view(-1), target_radius.view(-1))
    losses["direction"] = direction_loss_fn(predicted_direction, target_direction)
    losses["class_l"] = class_loss_fn(predicted_class, target_class)

    return losses


def L1Loss(outputs, targets):
    loss = nn.L1Loss()
    return loss(outputs, targets)


def cosine_similarity_loss(outputs, targets):
    loss = nn.CosineSimilarity()
    return torch.mean(1 - loss(outputs, targets))


def dice_loss(outputs, targets):
    # https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
    smooth = 1
    outputs = F.softmax(outputs, dim=1)
    targets = F.one_hot(targets)

    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()

    return 1 - (
        (2.0 * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)
    )


def focal_loss(outputs, targets):
    # https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
    gamma = 2
    input = outputs

    if input.dim() > 2:
        input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
    targets = targets.view(-1, 1)

    logpt = F.log_softmax(input, dim=1)
    logpt = logpt.gather(1, targets)
    logpt = logpt.view(-1)
    pt = logpt.exp()
    loss = -1 * (1 - pt) ** gamma * logpt
    # return loss.sum()
    return loss.mean()


def nll_loss(outputs, targets):
    return torch.tensor([0]).cuda()
    weights = targets.shape[0] / (torch.bincount(targets))  # Balance class weights
    return F.nll_loss(F.log_softmax(outputs, dim=1), targets, weight=weights)
