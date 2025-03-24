
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter

from basicsr.models.archs.UHDM_arch import UHDM
from basicsr.models.archs.DWTFFC_arch import DWT_FFC

class DWTFFC_UHDM(nn.Module):
    def __init__(self, dim=48):
        super(DWTFFC_UHDM, self).__init__()
        self.lap_branch = DWT_FFC()
        self.image_branch = UHDM()

    def forward(self, x):

        ## padding the image
        _, _, H, W = x.shape
        rate = 2 ** 5
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        res3, res2, res1 = self.lap_branch(x)
        
        out_1, out_2, out_3 = self.image_branch(x, res3, res2, res1)

        out_1 = out_1[:,:,:H,:W]
        out_2 = out_2[:,:,:H//2,:W//2]
        out_3 = out_3[:,:,:H//4,:W//4]
        #print(out_1.shape)
        return out_1, out_2, out_3
