from logging import log
import torch
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits, normalize

def contrastive_loss(scr_gts, u_labels, projections, max_nsample, temperature):
  # ready for sampling 'pixel-wise features' cuz it's too much
  scr0_bs, scr0_ys, scr0_xs = torch.nonzero(torch.logical_or(scr_gts==0, u_labels==0), as_tuple=True)
  scr1_bs, scr1_ys, scr1_xs = torch.nonzero(torch.logical_or(scr_gts==1, u_labels==1), as_tuple=True)
  
  scr_sampler0 = np.arange(len(scr0_bs))
  scr_sampler1 = np.arange(len(scr1_bs))
  np.random.shuffle(scr_sampler0)
  np.random.shuffle(scr_sampler1)
  # class balancing & feature number control
  idx = min(len(scr_sampler0), len(scr_sampler1), max_nsample)
  scr0_bs, scr0_ys, scr0_xs = scr0_bs[scr_sampler0[:idx]], scr0_ys[scr_sampler0[:idx]], scr0_xs[scr_sampler0[:idx]]
  scr1_bs, scr1_ys, scr1_xs = scr1_bs[scr_sampler1[:idx]], scr1_ys[scr_sampler1[:idx]], scr1_xs[scr_sampler1[:idx]]
  scr0_mat = normalize(projections[scr0_bs, :, scr0_ys, scr0_xs], p=2, dim=1) # [idx, proj_ch]
  scr1_mat = normalize(projections[scr1_bs, :, scr1_ys, scr1_xs], p=2, dim=1) # [idx, proj_ch]

  # L2 normalization on feature vectors (pixel)
  cls0_vecs, cls1_vecs = list(), list()
  cls0_vecs.append(scr0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  cls1_vecs.append(scr1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]

  cls0_mat = torch.cat(cls0_vecs, dim=0) if len(cls0_vecs) > 1 else cls0_vecs[0] # [idx*proj_ch, proj_ch]
  cls1_mat = torch.cat(cls1_vecs, dim=0) if len(cls1_vecs) > 1 else cls1_vecs[0] # [idx*proj_ch, proj_ch]

  # p-p, n-n, p-n similarity
  mask = torch.eye(idx, dtype=torch.bool)
  logits_00 = torch.matmul(cls0_mat, cls0_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_11 = torch.matmul(cls1_mat, cls1_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_01 = torch.matmul(cls0_mat, cls1_mat.T) / temperature # [idx, idx]

  # exclude diagonal parts
  if idx != 0:
    logits_00 = logits_00[~mask].view(logits_00.shape[0], -1)
    logits_11 = logits_11[~mask].view(logits_11.shape[0], -1)

  loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
  loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
  loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))

  loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01

  return loss_contrast

def center_cont_loss(scr_gts, u_labels, projections, max_nsample, temperature):
  # ready for sampling 'pixel-wise features' cuz it's too much
  scr0_bs, scr0_ys, scr0_xs = torch.nonzero(torch.logical_or(scr_gts==0, u_labels==0), as_tuple=True)
  scr1_bs, scr1_ys, scr1_xs = torch.nonzero(torch.logical_or(scr_gts==1, u_labels==1), as_tuple=True)
  
  scr_sampler0 = np.arange(len(scr0_bs))
  scr_sampler1 = np.arange(len(scr1_bs))
  np.random.shuffle(scr_sampler0)
  np.random.shuffle(scr_sampler1)
  # class balancing & feature number control
  idx = min(len(scr_sampler0), len(scr_sampler1), max_nsample)
  scr0_bs, scr0_ys, scr0_xs = scr0_bs[scr_sampler0[:idx]], scr0_ys[scr_sampler0[:idx]], scr0_xs[scr_sampler0[:idx]]
  scr1_bs, scr1_ys, scr1_xs = scr1_bs[scr_sampler1[:idx]], scr1_ys[scr_sampler1[:idx]], scr1_xs[scr_sampler1[:idx]]
  scr0_mat = normalize(projections[scr0_bs, :, scr0_ys, scr0_xs], p=2, dim=1) # [idx, proj_ch]
  scr1_mat = normalize(projections[scr1_bs, :, scr1_ys, scr1_xs], p=2, dim=1) # [idx, proj_ch]
  
  cls0_cen = torch.mean(scr0_mat, dim=0, keepdim=True)
  cls1_cen = torch.mean(scr1_mat, dim=0, keepdim=True)

  cls0_cen = torch.mean(projections[scr0_bs, :, scr0_ys, scr0_xs], 0, keepdim=True)
  cls1_cen = torch.mean(projections[scr1_bs, :, scr1_ys, scr1_xs], 0, keepdim=True)
  cls0_norm = torch.sum(normalize(cls0_cen - projections[scr0_bs, :, scr0_ys, scr0_xs], p=2, dim=1))
  cls1_norm = torch.sum(normalize(cls1_cen - projections[scr1_bs, :, scr1_ys, scr1_xs], p=2, dim=1))

  loss_ctc = cls0_norm/(cls1_norm+1) + cls1_norm/(cls0_norm+1)

  # L2 normalization on feature vectors (pixel)
  # cls0_vecs, cls1_vecs = list(), list()
  # cls0_vecs.append(scr0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  # cls1_vecs.append(scr1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]

  # cls0_mat = torch.cat(cls0_vecs, dim=0) if len(cls0_vecs) > 1 else cls0_vecs[0] # [idx*proj_ch, proj_ch]
  # cls1_mat = torch.cat(cls1_vecs, dim=0) if len(cls1_vecs) > 1 else cls1_vecs[0] # [idx*proj_ch, proj_ch]

  # p-p, n-n, p-n similarity
  mask = torch.eye(idx, dtype=torch.bool)
  logits_00 = torch.matmul(scr0_mat, scr0_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_11 = torch.matmul(scr1_mat, scr1_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_01 = torch.matmul(scr0_mat, scr1_mat.T) / temperature # [idx, idx]

  # exclude diagonal parts
  if idx != 0:
    logits_00 = logits_00[~mask].view(logits_00.shape[0], -1)
    logits_11 = logits_11[~mask].view(logits_11.shape[0], -1)

  loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
  loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
  loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))

  loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01

  return loss_contrast, loss_ctc

def center_cont_loss2(scr_gts, u_labels, projections, max_nsample, temperature):
  # ready for sampling 'pixel-wise features' cuz it's too much
  scr0_bs, scr0_ys, scr0_xs = torch.nonzero(torch.logical_or(scr_gts==0, u_labels==0), as_tuple=True)
  scr1_bs, scr1_ys, scr1_xs = torch.nonzero(torch.logical_or(scr_gts==1, u_labels==1), as_tuple=True)
  
  scr_sampler0 = np.arange(len(scr0_bs))
  scr_sampler1 = np.arange(len(scr1_bs))
  np.random.shuffle(scr_sampler0)
  np.random.shuffle(scr_sampler1)

  # class balancing & feature number control
  idx = min(len(scr_sampler0), len(scr_sampler1), max_nsample)
  scr0_bs, scr0_ys, scr0_xs = scr0_bs[scr_sampler0[:idx]], scr0_ys[scr_sampler0[:idx]], scr0_xs[scr_sampler0[:idx]]
  scr1_bs, scr1_ys, scr1_xs = scr1_bs[scr_sampler1[:idx]], scr1_ys[scr_sampler1[:idx]], scr1_xs[scr_sampler1[:idx]]
  scr0_mat = normalize(projections[scr0_bs, :, scr0_ys, scr0_xs], p=2, dim=1) # [idx, proj_ch]
  scr1_mat = normalize(projections[scr1_bs, :, scr1_ys, scr1_xs], p=2, dim=1) # [idx, proj_ch]
  
  cls0_cen = torch.mean(scr0_mat, dim=0, keepdim=True)
  cls1_cen = torch.mean(scr1_mat, dim=0, keepdim=True)  

  # L2 normalization on feature vectors (pixel)
  # cls0_vecs, cls1_vecs = list(), list()
  # cls0_vecs.append(scr0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  # cls1_vecs.append(scr1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]

  # cls0_mat = torch.cat(cls0_vecs, dim=0) if len(cls0_vecs) > 1 else cls0_vecs[0] # [idx*proj_ch, proj_ch]
  # cls1_mat = torch.cat(cls1_vecs, dim=0) if len(cls1_vecs) > 1 else cls1_vecs[0] # [idx*proj_ch, proj_ch]

  # p-p, n-n, p-n similarity
  mask = torch.eye(idx, dtype=torch.bool)
  logits_0c = torch.matmul(scr0_mat, cls0_cen.T) / temperature
  logits_1c = torch.matmul(scr1_mat, cls1_cen.T) / temperature
  logits_cc = torch.matmul(cls0_cen, cls1_cen.T) / temperature
  logits_00 = torch.matmul(scr0_mat, scr0_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_11 = torch.matmul(scr1_mat, scr1_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_01 = torch.matmul(scr0_mat, scr1_mat.T) / temperature # [idx, idx]

  # exclude diagonal parts
  if idx != 0:
    logits_00 = logits_00[~mask].view(logits_00.shape[0], -1)
    logits_11 = logits_11[~mask].view(logits_11.shape[0], -1)

  loss_contrast_0c = binary_cross_entropy_with_logits(logits_0c, torch.ones_like(logits_0c))
  loss_contrast_1c = binary_cross_entropy_with_logits(logits_1c, torch.ones_like(logits_1c))
  loss_contrast_cc = binary_cross_entropy_with_logits(logits_cc, torch.zeros_like(logits_cc))

  loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
  loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
  loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))

  loss_center = loss_contrast_0c + loss_contrast_1c + loss_contrast_cc
  loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01

  return loss_contrast, loss_center

def selective_center_cont_loss(scr_gts, u_labels, projections, max_nsample, temperature):
  # ready for sampling 'pixel-wise features' cuz it's too much
  scr0_bs, scr0_ys, scr0_xs = torch.nonzero(torch.logical_or(scr_gts==0, u_labels==0), as_tuple=True)
  scr1_bs, scr1_ys, scr1_xs = torch.nonzero(torch.logical_or(scr_gts==1, u_labels==1), as_tuple=True)
  
  scr_sampler0 = np.arange(len(scr0_bs))
  scr_sampler1 = np.arange(len(scr1_bs))
  np.random.shuffle(scr_sampler0)
  np.random.shuffle(scr_sampler1)
  # class balancing & feature number control
  idx = min(len(scr_sampler0), len(scr_sampler1), max_nsample)
  scr0_bs, scr0_ys, scr0_xs = scr0_bs[scr_sampler0[:idx]], scr0_ys[scr_sampler0[:idx]], scr0_xs[scr_sampler0[:idx]]
  scr1_bs, scr1_ys, scr1_xs = scr1_bs[scr_sampler1[:idx]], scr1_ys[scr_sampler1[:idx]], scr1_xs[scr_sampler1[:idx]]
  scr0_mat = normalize(projections[scr0_bs, :, scr0_ys, scr0_xs], p=2, dim=1) # [idx, proj_ch]
  scr1_mat = normalize(projections[scr1_bs, :, scr1_ys, scr1_xs], p=2, dim=1) # [idx, proj_ch]

  # L2 normalization on feature vectors (pixel)
  cls0_vecs, cls1_vecs = list(), list()
  cls0_vecs.append(scr0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  cls1_vecs.append(scr1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]

  cls0_mat = torch.cat(cls0_vecs, dim=0) if len(cls0_vecs) > 1 else cls0_vecs[0] # [idx*proj_ch, proj_ch]
  cls1_mat = torch.cat(cls1_vecs, dim=0) if len(cls1_vecs) > 1 else cls1_vecs[0] # [idx*proj_ch, proj_ch]

  # p-p, n-n, p-n similarity
  mask = torch.eye(idx, dtype=torch.bool)
  logits_00 = torch.matmul(cls0_mat, cls0_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_11 = torch.matmul(cls1_mat, cls1_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_01 = torch.matmul(cls0_mat, cls1_mat.T) / temperature # [idx, idx]

  # exclude diagonal parts
  if idx != 0:
    logits_00 = logits_00[~mask].view(logits_00.shape[0], -1)
    logits_11 = logits_11[~mask].view(logits_11.shape[0], -1)

  loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
  loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
  loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))

  loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01

  return loss_contrast

def selective_cont_loss(scr_gts, u_labels, projections, max_nsample, temperature):
  # ready for sampling 'pixel-wise features' cuz it's too much
  scr0_bs, scr0_ys, scr0_xs = torch.nonzero(torch.logical_or(scr_gts==0, u_labels==0), as_tuple=True)
  scr1_bs, scr1_ys, scr1_xs = torch.nonzero(torch.logical_or(scr_gts==1, u_labels==1), as_tuple=True)
  
  scr_sampler0 = np.arange(len(scr0_bs))
  scr_sampler1 = np.arange(len(scr1_bs))
  idx = min(len(scr_sampler0), len(scr_sampler1), max_nsample)

  _, order0 = torch.sort(u_labels[scr0_bs, scr0_ys, scr0_xs], dim=0, descending=False)
  _, order1 = torch.sort(u_labels[scr1_bs, scr1_ys, scr1_xs], dim=0, descending=True)
  idx0 = order0[:idx]
  idx1 = order1[:idx]
  print(idx0, idx1)
  print(len(idx0), len(idx1))

  # _, order0 = torch.topk(u_labels[scr0_bs, scr0_ys, scr0_xs], k=idx, )
  # _, order1 = torch.topk(u_labels[scr1_bs, scr1_ys, scr1_xs], k=idx, )

  # np.random.shuffle(scr_sampler0)
  # np.random.shuffle(scr_sampler1) 
  # class balancing & feature number control
  # scr0_bs, scr0_ys, scr0_xs = scr0_bs[scr_sampler0[:idx0]], scr0_ys[scr_sampler0[:idx0]], scr0_xs[scr_sampler0[:idx0]]
  # scr1_bs, scr1_ys, scr1_xs = scr1_bs[scr_sampler1[:idx1]], scr1_ys[scr_sampler1[:idx1]], scr1_xs[scr_sampler1[:idx1]]
  scr0_mat = normalize(projections[scr0_bs[idx0], :, scr0_ys[idx0], scr0_xs[idx0]], p=2, dim=1) # [idx, proj_ch]
  scr1_mat = normalize(projections[scr1_bs[idx1], :, scr1_ys[idx1], scr1_xs[idx1]], p=2, dim=1) # [idx, proj_ch]
  exit()
  # L2 normalization on feature vectors (pixel)
  cls0_vecs, cls1_vecs = list(), list()
  cls0_vecs.append(scr0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  cls1_vecs.append(scr1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]

  cls0_mat = torch.cat(cls0_vecs, dim=0) if len(cls0_vecs) > 1 else cls0_vecs[0] # [idx*proj_ch, proj_ch]
  cls1_mat = torch.cat(cls1_vecs, dim=0) if len(cls1_vecs) > 1 else cls1_vecs[0] # [idx*proj_ch, proj_ch]

  # p-p, n-n, p-n similarity
  mask = torch.eye(idx, dtype=torch.bool)
  logits_00 = torch.matmul(cls0_mat, cls0_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_11 = torch.matmul(cls1_mat, cls1_mat.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_01 = torch.matmul(cls0_mat, cls1_mat.T) / temperature # [idx, idx]

  # exclude diagonal parts
  if idx != 0:
    logits_00 = logits_00[~mask].view(logits_00.shape[0], -1)
    logits_11 = logits_11[~mask].view(logits_11.shape[0], -1)

  loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
  loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
  loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))

  loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01

  return loss_contrast

def intraclass_contrastive_loss(scr_gts, u_labels, projections, max_nsample, temperature):
  # ready for sampling 'pixel-wise features' cuz it's too much
  scr0_bs, scr0_ys, scr0_xs = torch.nonzero(torch.logical_and(scr_gts==0, scr_gts==0), as_tuple=True)
  scr1_bs, scr1_ys, scr1_xs = torch.nonzero(torch.logical_and(scr_gts==1, scr_gts==1), as_tuple=True)
  u0_bs, u0_ys, u0_xs = torch.nonzero(torch.logical_and(scr_gts==0, scr_gts==0), as_tuple=True)
  u1_bs, u1_ys, u1_xs = torch.nonzero(torch.logical_and(scr_gts==1, scr_gts==1), as_tuple=True)
  
  scr_sampler0 = np.arange(len(scr0_bs))
  scr_sampler1 = np.arange(len(scr1_bs))
  u_sampler0 = np.arange(len(u0_bs))
  u_sampler1 = np.arange(len(u1_bs))
  np.random.shuffle(scr_sampler0)
  np.random.shuffle(scr_sampler1)
  np.random.shuffle(u_sampler0)
  np.random.shuffle(u_sampler1)
  # class balancing & feature number control
  idx = min(len(scr_sampler0), len(scr_sampler1), max_nsample)
  idx = min(len(u_sampler0), len(u_sampler1), max_nsample)

  scr0_bs, scr0_ys, scr0_xs = scr0_bs[scr_sampler0[:idx]], scr0_ys[scr_sampler0[:idx]], scr0_xs[scr_sampler0[:idx]]
  scr1_bs, scr1_ys, scr1_xs = scr1_bs[scr_sampler1[:idx]], scr1_ys[scr_sampler1[:idx]], scr1_xs[scr_sampler1[:idx]]
  u0_bs, u0_ys, u0_xs = u0_bs[u_sampler0[:idx]], u0_ys[u_sampler0[:idx]], u0_xs[u_sampler0[:idx]]
  u1_bs, u1_ys, u1_xs = u1_bs[u_sampler1[:idx]], u1_ys[u_sampler1[:idx]], u1_xs[u_sampler1[:idx]]
  scr0_mat = normalize(projections[scr0_bs, :, scr0_ys, scr0_xs], p=2, dim=1) # [idx, proj_ch]
  scr1_mat = normalize(projections[scr1_bs, :, scr1_ys, scr1_xs], p=2, dim=1) # [idx, proj_ch]
  u0_mat = normalize(projections[u0_bs, :, u0_ys, u0_xs], p=2, dim=1) # [idx, proj_ch]
  u1_mat = normalize(projections[u1_bs, :, u1_ys, u1_xs], p=2, dim=1) # [idx, proj_ch]

  # L2 normalization on feature vectors (pixel)
  cls0_vecs, cls1_vecs = list(), list()
  cls0_vecs.append(scr0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  cls1_vecs.append(scr1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  cls0_vecs.append(u0_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]
  cls1_vecs.append(u1_mat) # [[idx, proj_ch], [idx, proj_ch], ... , [idx, proj_ch]]

  cls0_mat = torch.cat(cls0_vecs, dim=0) if len(cls0_vecs) > 1 else cls0_vecs[0] # [idx*proj_ch, proj_ch]
  cls1_mat = torch.cat(cls1_vecs, dim=0) if len(cls1_vecs) > 1 else cls1_vecs[0] # [idx*proj_ch, proj_ch]

  # For center loss
  cls0_cen = torch.mean(cls0_mat, 0, keepdim=True)
  cls1_cen = torch.mean(cls1_mat, 0, keepdim=True)

  cls0_mat_shuffled = cls0_mat[torch.randperm(len(cls0_mat)),:] # shuffle
  cls1_mat_shuffled = cls1_mat[torch.randperm(len(cls1_mat)),:] # shuffle

  # p-p, n-n, p-n similarity
  logits_00 = torch.matmul(cls0_mat, cls0_mat_shuffled.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_11 = torch.matmul(cls1_mat, cls1_mat_shuffled.T) / temperature # [idx,proj_ch] * [proj_ch,idx] = [idx, idx]
  logits_01 = torch.matmul(cls0_mat, cls1_mat_shuffled.T) / temperature # [idx, idx]

  loss_contrast_00 = binary_cross_entropy_with_logits(logits_00, torch.ones_like(logits_00))
  loss_contrast_11 = binary_cross_entropy_with_logits(logits_11, torch.ones_like(logits_11))
  loss_contrast_01 = binary_cross_entropy_with_logits(logits_01, torch.zeros_like(logits_01))

  # similarity to feature center
  logits_cen0 = torch.matmul(cls0_mat, cls0_cen.T) / temperature
  logits_cen1 = torch.matmul(cls1_mat, cls1_cen.T) / temperature
  loss_cen0 = binary_cross_entropy_with_logits(logits_cen0, torch.ones_like(logits_cen0))
  loss_cen1 = binary_cross_entropy_with_logits(logits_cen1, torch.ones_like(logits_cen1))

  # unsimilarity between centers
  logits_cens = torch.matmul(cls0_cen, cls1_cen.T) / temperature
  loss_cens = binary_cross_entropy_with_logits(logits_cens, torch.zeros_like(logits_cens))

  # loss normal
  # loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01 
  
  # loss center
  loss_contrast = loss_contrast_00 + loss_contrast_11 + loss_contrast_01 + loss_cen0 + loss_cen1 + loss_cens 

  return loss_contrast