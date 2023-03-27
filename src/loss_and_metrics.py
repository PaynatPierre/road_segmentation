import torch

class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, y_true, y_pred):
        # Flatten predictions and ground truths
        y_pred = torch.round(y_pred)

        y_true_flat = y_true.view(-1, 2)
        y_pred_flat = y_pred.view(-1, 2)

        # Compute intersection and union for each class
        intersection = torch.sum(y_true_flat * y_pred_flat, dim=0)
        union = torch.sum(y_true_flat + y_pred_flat, dim=0) - intersection

        # Compute IoU for each class and weight it by class proportion
        class_weights = torch.tensor([0.5, 0.5]).cuda()
        IoU = intersection / (union + torch.finfo(torch.float32).eps)
        weighted_IoU = class_weights * IoU

        # Compute final loss as 1 - average weighted IoU
        loss = 1.0 - torch.mean(weighted_IoU)

        return loss
    
def IoU_metric(y_true, y_pred):
    # Flatten predictions and ground truths
    # y_true_flat = y_true.view(-1, 2)
    # y_pred_flat = y_pred.view(-1, 2)

    y_true_flat = torch.flatten(torch.swapaxes(y_true,0,1), 1).cuda()
    y_pred_flat = torch.flatten(torch.swapaxes(y_pred,0,1), 1).cuda()

    # Compute intersection and union for each class
    intersection = torch.sum(y_true_flat * y_pred_flat, dim=1)
    union = torch.sum(y_true_flat + y_pred_flat, dim=1) - intersection

    # Compute IoU for each class and weight it by class proportion
    class_weights = torch.tensor([0, 1]).cuda()
    IoU = intersection / (union + torch.finfo(torch.float32).eps)
    weighted_IoU = class_weights * IoU

    # Compute final loss as 1 - average weighted IoU
    loss = 1.0 - torch.mean(weighted_IoU)

    return 1 - loss