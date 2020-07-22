import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LabelSmoothing(nn.Module):
    def __init__(self,smoothing=0.2):
        super(LabelSmoothing,self).__init__()
        self.criterion = nn.KLDivLoss()
        self.smoothing = smoothing

    def forward(self,y:torch.tensor,target:torch.tensor):
        true_y = y.clone()
        true_y.fill_(self.smoothing/4)
        true_y.scatter_(-1,target.unsqueeze(-1),1-self.smoothing)
        y = F.log_softmax(y,-1)
        return self.criterion(y,true_y)


