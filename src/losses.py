import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(input: torch.Tensor, target: torch.Tensor, gamma: float =2.0, reduction: str='mean', eps:float=1e-6):
    log_sigmoids = F.logsigmoid(input)
    prob = torch.exp(log_sigmoids)
    
    pos_weight = torch.pow(1. - prob, gamma)
    neg_weight = torch.pow(prob, gamma)
    
    focal = -(target*torch.log(prob+eps)*pos_weight+(1-target)*torch.log(1-prob+eps)*neg_weight)
    
    if reduction == 'None':
        return focal
    elif reduction == 'mean':
        return torch.mean(focal)
    elif reduction == 'sum':
        return torch.sum(focal)
    else:
         raise NotImplementedError("Invalid reduction mode: {}".format(reduction))
            
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-10
        self.__name__ = 'BinaryFocalLoss'

    def forward(self, input, target):
        return focal_loss(input, target, self.gamma, self.reduction, self.eps)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #get probabilities
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice



class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #get probabilities
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #get probabilities
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #get probabilities
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss