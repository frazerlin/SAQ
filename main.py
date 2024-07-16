

import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# Crop the image and split gts into crop ones with relax
def crop_with_relax(img,gt_split,offset_ratio=0.05,min_offset=10):
    gt_merge=np.uint8(sum(gt_split)!=0)
    x,y,w,h = cv2.boundingRect(gt_merge)
    x_offset,y_offset=max(int(w*offset_ratio),min_offset),max(int(h*offset_ratio),min_offset)
    x_s,y_s=max(x-x_offset,0),max(y-y_offset,0)
    x_e,y_e=min(x_s+w+2*x_offset-1,gt_merge.shape[1]-1),min(y_s+h+2*y_offset-1,gt_merge.shape[0]-1)
    img_crop = img[y_s:y_e+1,x_s:x_e+1,...] if img is not None else None
    gt_split_crop= [gt[y_s:y_e+1,x_s:x_e+1,...] for gt in gt_split] if gt_split is not None else None
    return img_crop,gt_split_crop

# Resize the too large image and split gts with short side equal to ref_size
def ref_resize(img,gt_split,ref_size=512):
    src_size=gt_split[0].shape[:2][::-1]
    if min(src_size)>ref_size:
        dst_size=(int(ref_size*src_size[0]/min(src_size)), int(ref_size*src_size[1]/min(src_size)))
        img=cv2.resize(img, dst_size, interpolation=cv2.INTER_NEAREST) if img is not None else None
        gt_split= [cv2.resize(gt, dst_size, interpolation=cv2.INTER_NEAREST) for gt in gt_split] if gt_split is not None else None
    return img,gt_split

# Get 360 degree convolution kernels for normal lines
def get_deg_kernels(normal_size=2):
    kernel_size=normal_size*2+1
    degs=list(range(-180,180,10))
    kernels=np.zeros([len(degs),kernel_size,kernel_size],dtype=np.uint8)
    spt=np.array([normal_size,normal_size])
    for idx,deg in enumerate(degs):
        pt_delta=[int(kernel_size*np.cos(deg/180.0*np.pi)+0.5),int(kernel_size*np.sin(deg/180.0*np.pi)+0.5)]
        cv2.line(kernels[idx],tuple(spt),tuple(spt+pt_delta),1,thickness=1)
    kernels=kernels*3+kernels[:,::-1,::-1]
    kernels[(kernels==0)|(kernels==4)]=2
    kernels=np.int64(kernels)-2
    return kernels

# Get 360 degree gradients with 360 degree convolution kernels
def get_img_kernel_grads(img,kernels,backend='pytorch-gpu'):
    normal_size=kernels.shape[-1]//2
    kernels=kernels/((np.sum(np.abs(kernels),axis=(1,2))/2.0)[:,None,None])

    if backend.split('-')[0]=='python':
        img=np.float32(img/255.0)
        img_kernel_grads=np.float32([np.mean(np.abs(cv2.filter2D(img,-1,kernel)),axis=2) for kernel in np.float32(kernels)])

    elif backend.split('-')[0]=='pytorch':
        img_tensor=torch.from_numpy((img/255.0).transpose((2, 0, 1)))[:,None,:,:].float()
        kernels_tensor=torch.from_numpy(kernels)[:,None,:,:].float()
        if backend.split('-')[1]=='gpu':
            img_tensor,kernels_tensor=img_tensor.cuda(),kernels_tensor.cuda()
        img_tensor_pad=F.pad(img_tensor,(normal_size,normal_size,normal_size,normal_size),mode='reflect')
        img_kernel_grads=F.conv2d(img_tensor_pad,kernels_tensor).abs().mean(0)

    elif backend.split('-')[0]=='jittor':
        img_tensor=jt.array((img/255.0).transpose((2,0,1)))[:,None,:,:]#.float()
        kernels_tensor=jt.array(kernels)[:,None,:,:].float()
        if backend.split('-')[1]=='gpu':
            pass
        img_tensor_pad=jt.nn.pad(img_tensor,(normal_size,normal_size,normal_size,normal_size),mode='reflect')
        img_kernel_grads=jt.nn.conv2d(img_tensor_pad,kernels_tensor).abs().mean(0)#.float32()

    return img_kernel_grads

# Get 360 degree rank scores
def get_grad_rank_scores(img,kernels,backend='pytorch-gpu'):
    normal_size=kernels.shape[-1]//2
    img_kernel_grads=get_img_kernel_grads(img,kernels,backend)
    # img_kernel_grads_extend=F.pad(img_kernel_grads[:,None,:,:],(normal_size,normal_size,normal_size,normal_size),mode='reflect')[:,0,:,:]
    rank_scores=[]
    for idx,kernel in enumerate(kernels):
        img_kernel_grad=img_kernel_grads[idx]
        if backend.split('-')[0]=='python':
            img_kernel_grad_extend=np.pad(img_kernel_grad,normal_size,mode='reflect')
            ks_line_elem=np.stack([img_kernel_grad_extend[krcidx[0]:krcidx[0]+img.shape[0],krcidx[1]:krcidx[1]+img.shape[1]] for krcidx in np.argwhere(kernel!=0)]+[img_kernel_grad])
            rank_sort=np.argsort(ks_line_elem,axis=0,kind='stable')
            rank_index=  np.argmax((rank_sort==(ks_line_elem.shape[0]-1)),axis=0)
            rank_score=np.float32(rank_index/(ks_line_elem.shape[0]-1))

        elif backend.split('-')[0]=='pytorch':
            # img_kernel_grad_extend=img_kernel_grads_extend[idx]
            img_kernel_grad_extend=F.pad(img_kernel_grad[None,None,:,:],(normal_size,normal_size,normal_size,normal_size),mode='reflect')[0,0,:,:]
            ks_line_elem=torch.stack([img_kernel_grad_extend[krcidx[0]:krcidx[0]+img.shape[0],krcidx[1]:krcidx[1]+img.shape[1]] for krcidx in np.argwhere(kernel!=0)]+[img_kernel_grad])
            rank_sort=torch.argsort(ks_line_elem,dim=0,stable=True)
            rank_index=  torch.argmax((rank_sort==(ks_line_elem.shape[0]-1)).float(),dim=0)
            rank_score=(rank_index/(ks_line_elem.shape[0]-1)).cpu()
        elif backend.split('-')[0]=='jittor':
            img_kernel_grad_extend=jt.nn.pad(img_kernel_grad[None,None,:,:],(normal_size,normal_size,normal_size,normal_size),mode='reflect')[0,0,:,:]
            ks_line_elem=jt.stack([img_kernel_grad_extend[krcidx[0]:krcidx[0]+img.shape[0],krcidx[1]:krcidx[1]+img.shape[1]] for krcidx in np.argwhere(kernel!=0)]+[img_kernel_grad])
            rank_sort=jt.argsort(ks_line_elem,dim=0)[0]
            rank_index=  jt.argmax((rank_sort==(ks_line_elem.shape[0]-1)),dim=0)[0]
            rank_score=(rank_index/(ks_line_elem.shape[0]-1))

        rank_scores.append(rank_score)
    rank_scores=np.stack(rank_scores)
    return rank_scores

#Get the normal degree of each pixels in one contour (-pi~pi]
def get_deg_contour_normal(contour,tangent_size=4):
    contour=contour[:,0,:]
    pt_num=len(contour)
    contour_tile=np.tile(contour,[tangent_size+1,1])
    delta_back=np.array([contour-contour_tile[tangent_size*pt_num-i:(tangent_size+1)*pt_num-i] for i in range(1,tangent_size+1)])
    delta_front=np.array([contour_tile[i:pt_num+i]-contour for i in range(1,tangent_size+1)])
    delta_line=(delta_back.mean(axis=0)+delta_front.mean(axis=0))/2
    rad_contour=np.arctan2(delta_line[:,1],delta_line[:,0])
    rad_contour_normal=(rad_contour+np.pi/2)
    deg_contour_normal=(rad_contour_normal/np.pi)*180
    deg_contour_normal[deg_contour_normal>180]=deg_contour_normal[deg_contour_normal>180]-360   
    return deg_contour_normal

#Get the map of kernel index for normal degree with contours
def get_contour_normal_degree_kernel_idx_map(gt_split,tangent_size=4,step=10):
    kernel_idx_map=np.zeros(gt_split[0].shape,dtype=np.uint8)
    deg_contour_normal_map=500*np.ones(gt_split[0].shape,dtype=np.float64)
    for gt in gt_split:
        gt_contours=(cv2.findContours(gt,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE))[-2]
        for gt_contour in gt_contours:
            deg_contour_normal=get_deg_contour_normal(gt_contour,tangent_size)
            deg_contour_normal_map[gt_contour[:,0,1],gt_contour[:,0,0]]=deg_contour_normal
            kernel_idxs=np.uint8(np.round(deg_contour_normal/step)+(180//step))
            kernel_idxs[kernel_idxs==(360//step)]=0
            kernel_idx_map[gt_contour[:,0,1],gt_contour[:,0,0]]=kernel_idxs+1
    return kernel_idx_map,deg_contour_normal_map

#Get the final score from the map of rank scores and kernel index
def get_contour_grad(rank_scores,kernel_idx_map):
    kernel_idx_map[(0,-1),:]=kernel_idx_map[:,(0,-1)]=0
    idx_coord=np.argwhere(kernel_idx_map>0)
    idx_value=kernel_idx_map[idx_coord[:,0],idx_coord[:,1]][:,np.newaxis]-1
    idx_vcd=np.concatenate([idx_value,idx_coord],axis=1)
    contour_grads=rank_scores[idx_vcd[:,0],idx_vcd[:,1],idx_vcd[:,2]]
    score= None if len(contour_grads)==0 else contour_grads.mean()
    contour_grad_map=-np.ones(kernel_idx_map.shape,dtype=np.float64)
    contour_grad_map[idx_vcd[:,1],idx_vcd[:,2]]=contour_grads
    return score,contour_grad_map


def get_saq_score(img,gt,ref_size=512,tangent_size=4,normal_size=[2,4,8],backend='pytorch-gpu'):
    # Preprocess
    gt_split=[np.uint8(gt==i) for i in set(gt.flat) if i!=0]
    img,gt_split=crop_with_relax(img,gt_split,0.05)
    img,gt_split=ref_resize(img,gt_split,ref_size)

    # Get best rank scores of all pixels for 360 degrees 
    rank_scores=np.max(np.stack([get_grad_rank_scores(img,get_deg_kernels(ns),backend=backend) for ns in normal_size],axis=0),axis=0)

    # Get the map of kernel index for normal degree with contours
    kernel_idx_map,_=get_contour_normal_degree_kernel_idx_map(gt_split,tangent_size)

    # Get the final score
    score,_=get_contour_grad(rank_scores,kernel_idx_map)
    return score


# Get the SAQ score from path, the path can be a file or a folder
def get_score_from_path(img_path,gt_path,score_kind='SAQ'):
    # Load image paths and gt paths
    if Path(img_path).is_dir() and Path(gt_path).is_dir():
        img_paths=sorted(list(Path(img_path).glob('*.*')))
        gt_paths=[list(Path(gt_path).glob('{}.png'.format(t.stem)))[0] for t in img_paths]
    else:
        img_paths=[Path(img_path)]
        gt_paths=[Path(gt_path)]

    # Calculate the mean SAQ and OCC score for all pairs
    saq_scores=[]
    for img_path,gt_path in tqdm(zip(img_paths,gt_paths),total=len(img_paths)):
        gt= np.uint8(Image.open(str(gt_path)))
        if gt.ndim!=2:gt=gt[:,:,0]
        if 'SAQ' in score_kind:
            img = np.uint8(Image.open(str(img_path)).convert('RGB'))
            saq_score=get_saq_score(img,gt,backend=args.backend)
            saq_scores.append(saq_score)

    saq_score_mean=sum(saq_scores)/len(saq_scores) if 'SAQ' in score_kind else None
    return saq_score_mean


if __name__ == "__main__":
    # Input parameters
    parser = argparse.ArgumentParser(description='Segmentation Annotation Quality Assessment')
    parser.add_argument('--img', type=str, default='img.jpg')
    parser.add_argument('--gt', type=str, default='gt.png')
    parser.add_argument('--backend', type=str, choices=['python-cpu','pytorch-cpu','pytorch-gpu','jittor-cpu','jittor-gpu'],default='python-cpu')
    parser.add_argument('--score_kind', type=str, default='SAQ')
    args = parser.parse_args()

    print(args.backend)

    if args.backend.split('-')[0]=='pytorch':
        import torch
        import torch.nn.functional as F
    elif args.backend.split('-')[0]=='jittor':
        import jittor as jt
        if  args.backend.split('-')[1]=='gpu':
            jt.flags.use_cuda = 1 if jt.has_cuda else 0
            print("Using CUDA:", jt.flags.use_cuda)

    saq_score_mean=get_score_from_path(args.img,args.gt,score_kind=args.score_kind)
    print('The SAQ Score for <{}> and <{}> is {}'.format(args.img,args.gt,saq_score_mean))
    
    




