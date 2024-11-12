# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import ot
import numpy as np

from scipy.optimize import linear_sum_assignment
#import cupy as cp
#function for assignment problem
def assignment_problem(noise, image, labels=None, reg=0.1):
    N, C, W, H = image.shape
    
    # Compute the distance matrix
    dist_matrix = torch.cdist(image.reshape(N, C * W * H), noise.reshape(N, C * W * H))
    dist_matrix_np = dist_matrix.detach().cpu().numpy()
    del dist_matrix  # Free memory
    
    # Uniform distributions for Sinkhorn algorithm
    a = np.ones(N) / N  # source distribution (image side)
    b = np.ones(N) / N  # target distribution (noise side)

    # Compute the transport plan using Sinkhorn
    transport_plan = ot.sinkhorn(a, b, dist_matrix_np, reg)

    # Select noise based on transport plan (using probabilities for soft assignment)
    noise_indices = np.argmax(transport_plan, axis=1)
    selected_noise = noise[torch.tensor(noise_indices, device=noise.device)]
    
    return selected_noise


def assignment_problem_label(noise, image, labels, reg=0.1):
    N, C, W, H = image.shape
    _, L = labels.shape
    labels_idx = [[] for _ in range(L)]
    a_label = torch.argmax(labels, 1)  # Convert one-hot labels to indices

    # Collect indices by label
    for i in range(N):
        labels_idx[a_label[i]].append(i)

    b_lst = []
    for i in range(L):
        if len(labels_idx[i]) == 0:
            continue  # Skip if no samples for this label

        # Select images and noise for the current label
        image_subset = image.reshape(N, C * W * H)[labels_idx[i]]
        noise_subset = noise.reshape(N, C * W * H)[labels_idx[i]]
        
        # Calculate the distance matrix for this label subset
        dist_matrix = torch.cdist(image_subset, noise_subset)
        dist_matrix_np = dist_matrix.detach().cpu().numpy()
        del dist_matrix  # Free memory

        # Uniform distributions for Sinkhorn algorithm within each label subset
        a = np.ones(len(labels_idx[i])) / len(labels_idx[i])
        b = np.ones(len(labels_idx[i])) / len(labels_idx[i])

        # Compute the transport plan using Sinkhorn
        transport_plan = ot.sinkhorn(a, b, dist_matrix_np, reg)

        # Get noise indices based on transport plan
        noise_indices = np.argmax(transport_plan, axis=1)
        b_lst.append([labels_idx[i][idx] for idx in noise_indices])

    # Flatten the list to match original noise order
    new_lst = [0 for _ in range(N)]
    for i in range(N):
        label_index = a_label[i]
        new_lst[i] = b_lst[label_index].pop(0)  # Assign based on label-specific matching
    
    return noise[torch.tensor(new_lst, device=noise.device)]


#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss_tp:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,batch = 32):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.batch = batch

    def __call__(self, net, images, labels=None, augment_pipe=None, noise = None):
        N = self.batch
        rnd_normal = torch.randn([N, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        with torch.no_grad():
            n_ = torch.randn_like(y)
            #n_ = assignment_problem(n_,y)
            n_ = assignment_problem_label(n_,y,labels)
        n = n_[0:N,:,:,:] * sigma
        y_= y[0:N,:,:,:]
        del y, n_
        D_yn = net(y_ + n, sigma, labels[0:N,:], augment_labels=augment_labels[0:N,:])
        loss = weight * ((D_yn - y_) ** 2)
        return loss


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,batch = 32):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.batch = batch

    def __call__(self, net, images, labels=None, augment_labels=None, noise = None):
        N = self.batch
        rnd_normal = torch.randn([N, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        #y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        y = images
        n =  noise * sigma
        
        
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
