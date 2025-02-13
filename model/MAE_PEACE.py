import torch
import torch.nn as nn
import torchvision.models as models
import math 
from marlin_pytorch import Marlin
from .ASTModel import ASTModel


class MAE_AST(nn.Module):

    def __init__(self):
        super().__init__()

        self.mae = Marlin.from_online("marlin_vit_base_ytf")
        self.mae.eval()
        self.ast = ASTModel(label_dim=2, input_tdim=600, imagenet_pretrain=True, audioset_pretrain=True)
        self.ast.eval()
        # self.norm = nn.BatchNorm1d(768*2)
        self.decoder = nn.Sequential(
                nn.Linear(768*6, 512),
                # nn.BatchNorm1d(512),
                nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(512, 2)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, fea, x_a, phrase='train'):
        """
        Input x is shape (B, L, d_input)
        """
        fea.zero_()
        
        for i in range(4):
            fea[:,:,i] = self.mae.extract_features(x[:,:,16*i:16*(i+1),:,:], keep_seq=False)
        
        fea = fea.flatten(1)  # (32, 768*4)
        fea = torch.cat([fea,x_a],axis=-1) # (32, 768*6)
        # fea = self.norm(fea)
        pred_logit = self.decoder(fea)  # (B, d_model) -> (B, d_output)
        pred_score = self.softmax(pred_logit)
        if phrase=='eval':
            return pred_score
        return pred_score, pred_logit
    
    
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    x1 = torch.rand((16, 40, 3, 224, 224)).cuda()
    model = ViS4mer(n_layers=3, d_model=3072,d_input=1024,d_output=2,dropout=0.2,l_max=40).cuda()
    model.train()
    o1 = model(x1)
    