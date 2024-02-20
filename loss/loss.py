import torch
import torch.nn as nn
import os,sys


class Loss(nn.Module):
    def __init__(self,radius=1.0):
        super(Loss,self).__init__()
        self.radius=radius

    def get_discriminator_loss(self,pred_fake,pred_real):
        real_loss=torch.mean((pred_real-1)**2)
        fake_loss=torch.mean(pred_fake**2)
        loss=real_loss+fake_loss
        return loss
    def get_generator_loss(self,pred_fake):
        fake_loss=torch.mean((pred_fake-1)**2)
        return fake_loss
    def get_discriminator_loss_single(self,pred,label=True):
        if label==True:
            loss=torch.mean((pred-1)**2)
            return loss
        else:
            loss=torch.mean((pred)**2)
            return loss
if __name__=="__main__":
    loss=Loss().cuda()
    point_cloud=torch.rand(4,4096,3).cuda()
    uniform_loss=loss.get_uniform_loss(point_cloud)
    repulsion_loss=loss.get_repulsion_loss(point_cloud)

