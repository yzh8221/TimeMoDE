import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math

class D_constraint1(torch.nn.Module):
    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        I = torch.eye(d.shape[1]).to(d.device)  # .cuda()
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return 	1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):
    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim, n_clusters):
        S = torch.ones(d.shape[1], d.shape[1]).to(d.device)
        zero = torch.zeros(dim, dim)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return 1e-3 * loss_d2_constraint
    

class EDESC(nn.Module):

    def __init__(self,
                 d_model,
                 n_clusters,
                 eta):
        super(EDESC, self).__init__()
        self.n_clusters = n_clusters
        self.eta = eta

        # Subspace bases proxy
        # self.D = Parameter(torch.Tensor(n_z, n_clusters))
        n_z = d_model
        self.d = 8
        self.D = Parameter(torch.Tensor(n_z, n_clusters * self.d))
        # init.xavier_uniform_(self.D)
        init.kaiming_normal_(self.D, mode='fan_in', nonlinearity='relu')

    def forward(self, z):  # z: [bs * patch_num_out x nvars * d_model]
        # x_bar, z = self.ae(x)  # x_bar: [bs * patch_num x nvars * patch_len]
        # x_bar = torch.reshape(x_bar, (self.bs, -1, self.c_out, self.patch_len))
        # x_bar = self.reverse_unfold(x_bar, length, self.stride)    # x_bar: [bs, c_out, context_window]
        s = None
        
        # Calculate subspace affinity
        for i in range(self.n_clusters):

            # si = torch.mm(z, self.D[:, i * self.d:(i + 1) * self.d])
            # si = torch.sum(torch.pow(si, 2), 1, keepdim=True)
            
            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * self.d:(i + 1) * self.d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + self.eta * self.d) / ((self.eta + 1) * self.d)
        s = (s.t() / torch.sum(s, 1)).t()  # Eq 13  [b_s, num_expert]
        return s

    def total_loss(self, n_clusters): # x, x_bar,
        # Reconstruction loss   Eq 9
        # reconstr_loss = F.mse_loss(x_bar, x)

        # Constraints   Eq 12
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, self.d, n_clusters)

        # Total_loss    Eq 16
        total_loss = loss_d1 + loss_d2  # reconstr_loss +

        return total_loss