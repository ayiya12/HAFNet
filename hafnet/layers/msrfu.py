import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class SKConv(nn.Module):  
    """Multi-Scale Receptive Field Selection Unit (MSRFU) from HAFNet"""  
    def __init__(self, channels, num_branches=2, reduction=16):  
        super().__init__()  
        self.channels = channels  
        self.num_branches = num_branches 
        self.branches = nn.ModuleList([  
            nn.Sequential(  
                nn.Conv2d(channels, channels, kernel_size=3+2*i,   
                         padding=1+i, groups=32),  
                nn.ReLU(inplace=True)  
            )  
            for i in range(num_branches)  
        ])  
  
        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(  
            nn.Conv2d(channels, channels // reduction, 1),  
            nn.ReLU(inplace=True)  
        )  
        self.fcs = nn.ModuleList([  
            nn.Conv2d(channels // reduction, channels, 1)  
            for _ in range(num_branches)  
        ])  
          
    def forward(self, x):    
        branch_outs = [branch(x) for branch in self.branches]    
        fused = sum(branch_outs)   
        attn = self.gap(fused)  
        attn = self.fc(attn)  
          
        attn_vectors = [fc(attn) for fc in self.fcs]  
        attn_vectors = torch.stack(attn_vectors, dim=1)  # [B, num_branches, C, 1, 1]  
        attn_vectors = F.softmax(attn_vectors, dim=1)  
  
        out = sum([attn_vectors[:, i] * branch_outs[i]   
                   for i in range(self.num_branches)])  
        return out