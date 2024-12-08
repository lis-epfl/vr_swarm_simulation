import os
import gc
import cv2
import glob
import sys

import yaml
# import utils
# import datasets
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur


from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# from BaseStitcher import *


# UDIS2 imports
sys.path.append(os.path.abspath("UDIS2_main\Warp\Codes"))
import UDIS2_main.Warp.Codes.utils_udis as udis_utils
import UDIS2_main.Warp.Codes.utils_udis.torch_DLT as torch_DLT
import UDIS2_main.Warp.Codes.grid_res as grid_res
from UDIS2_main.Warp.Codes.network import Network, build_model
from UDIS2_main.Warp.Codes.loss import cal_lp_loss2
from UDIS2_main.Warp.Codes.dataset import *

# NIS imports
sys.path.append(os.path.abspath("Neural_Image_Stitching_main"))
import Neural_Image_Stitching_main.srwarp
from Neural_Image_Stitching_main.srwarp import transform
import Neural_Image_Stitching_main.utils as nis_utils
from Neural_Image_Stitching_main.models.ihn import *
from Neural_Image_Stitching_main.models import *
# from Neural_Image_Stitching_main import stitch
from Neural_Image_Stitching_main import stitch
import Neural_Image_Stitching_main.pretrained
# import Neural_Image_Stitching_main.datasets
import Neural_Image_Stitching_main.datasetsNIS


# import Neural_Image_Stitching_main.datasets.image_folder
# print("I")
# import Neural_Image_Stitching_main.datasets
# print(dir(Neural_Image_Stitching_main.datasets))
# print(Neural_Image_Stitching_main.datasets.__file__)
# import Neural_Image_Stitching_main.datasets as datasets

# from Neural_Image_Stitching_main.datasets.datasets import datasets
# print(datasets)
# import Neural_Image_Stitching_main.datasets.image_folder
# from Neural_Image_Stitching_main.datasets.datasets import datasets

# REWARP imports
# sys.path.append(os.path.abspath("Residual_Elastic_Warp_main"))
# import Residual_Elastic_Warp_main.models
# import Residual_Elastic_Warp_main.utils

from pathlib import Path

# Add the `Residual_Elastic_Warp_main` folder to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "Residual_Elastic_Warp_main"))
import Residual_Elastic_Warp_main.models as RE_models
import Residual_Elastic_Warp_main.utils as RE_utils
import Residual_Elastic_Warp_main.datasets as RE_datasets
from Residual_Elastic_Warp_main.datasets import image_folder  # Ensures classes with @register are loaded
from Residual_Elastic_Warp_main.datasets import wrappers  # Same as above


c1 = [0, 0]; c2 = [0, 0]
finder = cv2.detail.SeamFinder.createDefault(2)
# UDIS : average psnr: 25.42688753348356, average ssim: 0.83781886
# NIS (only IHN) : PSNR:26.4804, SSIM:0.8206
# NIS :
# REWRARP : PSNR:22.9463/24.4961/26.2651, Avg PSNR: 24.9925, SSIM:0.7488/0.7848/0.8207, SSIM:0.7942

#### FROM UDIS

def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_path = r"testing"
    # dataset
    # test_data = TestDataset(data_path=args.test_path)
    # test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    test_data = TestDataset(data_path=test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    model_path ="UDIS2_main\Warp"
    MODEL_DIR = os.path.join(model_path, 'model')

    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')



    print("##################start testing#######################")
    psnr_list = []
    ssim_list = []
    net.eval()

    for i, batch_value in enumerate(test_loader):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

            with torch.no_grad():
                batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor, is_training=False)

            warp_mesh_mask = batch_out['warp_mesh_mask']
            warp_mesh = batch_out['warp_mesh']


            warp_mesh_np = ((warp_mesh[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
            warp_mesh_mask_np = warp_mesh_mask[0].cpu().detach().numpy().transpose(1,2,0)
            inpu1_np = ((inpu1_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)


            # print((inpu1_np * warp_mesh_mask_np).shape)
            # print((warp_mesh_np * warp_mesh_mask_np).shape)
            # calculate psnr/ssim
            psnr = compare_psnr(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, data_range=255)
            ssim = compare_ssim(inpu1_np*warp_mesh_mask_np, warp_mesh_np*warp_mesh_mask_np, data_range=255, multichannel=True, channel_axis=-1)


            print('i = {}, psnr = {:.6f}'.format( i+1, psnr))

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            torch.cuda.empty_cache()

        # if i>10:
        #     break

    print("=================== Analysis ==================")
    print("psnr")
    psnr_list.sort(reverse = True)
    psnr_list_30 = psnr_list[0 : 331]
    psnr_list_60 = psnr_list[331: 663]
    psnr_list_100 = psnr_list[663: -1]
    print("top 30%", np.mean(psnr_list_30))
    print("top 30~60%", np.mean(psnr_list_60))
    print("top 60~100%", np.mean(psnr_list_100))
    print('average psnr:', np.mean(psnr_list))

    ssim_list.sort(reverse = True)
    ssim_list_30 = ssim_list[0 : 331]
    ssim_list_60 = ssim_list[331: 663]
    ssim_list_100 = ssim_list[663: -1]
    print("top 30%", np.mean(ssim_list_30))
    print("top 30~60%", np.mean(ssim_list_60))
    print("top 60~100%", np.mean(ssim_list_100))
    print('average ssim:', np.mean(ssim_list))
    print("##################end testing#######################")

# For IHN fine tuned in NIS

def prepare_eval(config):
    spec = config.get('eval_dataset')
    dataset = Neural_Image_Stitching_main.datasetsNIS.datasets.make(spec['dataset'])
    print("a")
    dataset = Neural_Image_Stitching_main.datasetsNIS.datasets.make(spec['wrapper'], args={'dataset': dataset})
    print("b")
    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    print("c")

    sv_file = torch.load(config['resume_align'])
    H_model = Neural_Image_Stitching_main.models.IHN().cuda()
    H_model.load_state_dict(sv_file['model']['sd'])

    return loader, H_model


def eval_IHN(loader, model):
    model.eval()

    failures = 0

    tot_rmse, tot_mace = 0, 0
    tot_psnr, tot_ssim = 0, 0

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)
    
    desc= None
    i=0
    for b_id in pbar:
        i +=1
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp_tgt = batch['inp_ref'].permute(0,3,1,2)
        inp_src = batch['inp_tgt'].permute(0,3,1,2)

        inp_tgt_ = inp_tgt * 255
        inp_src_ = inp_src * 255

        mask = torch.ones_like(inp_src).cuda()
        b, c, h, w = inp_tgt.shape
        with torch.no_grad():
            four_pred, _ = model(inp_src_, inp_tgt_, iters_lev0=6, iters_lev1=3, test_mode=True)
        shift = four_pred.reshape(b, 2, -1).permute(0, 2, 1)

        shape = (128, 128)
        H, w_max, w_min, h_max, h_min = nis_utils.get_H(shift, shape)
        H = nis_utils.compens_H(H, size=shape)

        img_h = torch.ceil(h_max - h_min).int().item()
        img_w = torch.ceil(w_max - w_min).int().item()

        h_max = h_max.item(); h_min = h_min.item()
        w_max = w_max.item(); w_min = w_min.item()

        src_w_ovl = nis_utils.STN(inp_src, torch.inverse(H))
        mask_ovl = nis_utils.STN(mask, torch.inverse(H)).round().bool()
        tgt_ovl = inp_tgt * mask_ovl

        src_samples = src_w_ovl[mask_ovl].cpu().numpy()
        tgt_samples = tgt_ovl[mask_ovl].cpu().numpy()

        if len(src_samples) == 0:
            failures += 1
            continue
        
        # if i==1:
        #     print(tgt_samples.shape, src_samples.shape)
        psnr = compare_psnr(tgt_samples, src_samples, data_range=1.)
        ssim = compare_ssim(tgt_samples * 255, src_samples * 255, data_range=255.)

        tot_psnr += psnr
        tot_ssim += ssim

        pbar.set_description_str(
        desc="PSNR:{:.4f}, SSIM:{:.4f}, Failures:{}".format(
            tot_psnr/(b_id+1), tot_ssim/(b_id+1), failures), refresh=True)
        torch.cuda.empty_cache()
        # if i>10:
        #     break
    print(desc)
    

# Blending eval NIS

def seam_finder(ref, tgt, ref_m, tgt_m):
    ref_ = ref.mean(dim=1, keepdim=True)
    ref_ -= ref_.min()
    ref_ /= ref_.max()

    tgt_ = tgt.mean(dim=1, keepdim=True)
    tgt_ -= tgt_.min()
    tgt_ /= tgt_.max()

    ref_ = (ref_.cpu().numpy() * 255).astype(np.uint8)
    tgt_ = (tgt_.cpu().numpy() * 255).astype(np.uint8)
    ref_m = (ref_m[0,0,:,:].cpu().numpy() * 255).astype(np.uint8)
    tgt_m = (tgt_m[0,0,:,:].cpu().numpy() * 255).astype(np.uint8)

    inp = np.concatenate([ref_, tgt_], axis=0).transpose(0,2,3,1)
    inp = np.repeat(inp, 3, -1)

    masks = np.stack([ref_m, tgt_m], axis=0)[..., None]
    corners = np.stack([c1, c2], axis=0).astype(np.uint8)

    ref_m, tgt_m = finder.find(inp, corners, masks)
    ref *= torch.Tensor(cv2.UMat.get(ref_m).reshape(1,1,*ref.shape[-2:])/255).cuda()
    tgt *= torch.Tensor(cv2.UMat.get(tgt_m).reshape(1,1,*tgt.shape[-2:])/255).cuda()

    stit_rep = ref + tgt

    return stit_rep


def batched_predict(model, ref, ref_grid, ref_cell, ref_mask,
                    tgt, tgt_grid, tgt_cell, tgt_mask,
                    stit_grid, sizes, bsize, seam_cut=False):

    fea_ref, fea_ref_grid, ref_coef, ref_freq = model.gen_feat(ref)
    fea_ref_w = model.NeuralWarping(
        ref, fea_ref, fea_ref_grid,
        ref_freq, ref_coef, ref_grid, ref_cell, sizes
    )

    fea_tgt, fea_tgt_grid, tgt_coef, tgt_freq = model.gen_feat(tgt)
    fea_tgt_w = model.NeuralWarping(
        tgt, fea_tgt, fea_tgt_grid,
        tgt_freq, tgt_coef, tgt_grid, tgt_cell, sizes
    )

    if seam_cut:
        fea = seam_finder(fea_ref_w, fea_tgt_w, ref_mask, tgt_mask).repeat(1,2,1,1)

    else:
        fea_ref_w *= ref_mask
        fea_tgt_w *= tgt_mask
        fea = torch.cat([fea_ref_w, fea_tgt_w], dim=1)

    stit_rep = model.gen_feat_for_blender(fea)

    ql = 0
    preds = []
    n = ref_grid.shape[1]

    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(stit_rep, stit_grid[:, ql: qr, :])
        preds.append(pred)
        ql = qr

    pred = torch.cat(preds, dim=1)

    return pred


def prepare_ingredient(model, inp_tgt, inp_ref, tgt, ref):
    b, c, h, w = tgt.shape

    four_pred, _ = model(inp_tgt, inp_ref, iters_lev0=6, iters_lev1=3, test_mode=True)
    shift = four_pred.reshape(b, 2, -1).permute(0, 2, 1)

    shape = tgt.shape[-2:]
    H_tgt2ref, w_max, w_min, h_max, h_min = nis_utils.get_H(shift * w/128, shape)

    img_h = torch.ceil(h_max - h_min).int().item()
    img_w = torch.ceil(w_max - w_min).int().item()
    sizes = (img_h, img_w)

    h_max = h_max.item(); h_min = h_min.item()
    w_max = w_max.item(); w_min = w_min.item()

    eye = torch.eye(3).double()
    T = nis_utils.get_translation(h_min, w_min)

    H_tgt2ref = H_tgt2ref[0].double().cpu()
    H_tgt2ref = T @ H_tgt2ref

    eye, _, _ = transform.compensate_matrix(ref, eye)
    eye = T @ eye

    coord = nis_utils.to_pixel_samples(None, sizes=sizes)
    cell = nis_utils.make_cell(coord, None, sizes=sizes).cuda()
    coord = coord.cuda()

    coord1 = coord.clone()
    tgt_grid, tgt_mask = nis_utils.gridy2gridx_homography(
        coord1.contiguous(), *sizes, *ref.shape[-2:], H_tgt2ref.cuda(), cpu=False
    )

    cell1 = cell.clone()
    tgt_cell = nis_utils.celly2cellx_homography(
        cell1.contiguous(), *sizes, *tgt.shape[-2:], H_tgt2ref.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    coord2 = coord.clone()
    ref_grid, ref_mask = nis_utils.gridy2gridx_homography(
        coord2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    )

    cell2 = cell.clone()
    ref_cell = nis_utils.celly2cellx_homography(
        cell2.contiguous(), *sizes, *ref.shape[-2:], eye.cuda(), cpu=False
    ).unsqueeze(0).repeat(b,1,1)

    stit_grid = nis_utils.to_pixel_samples(None, sizes).cuda()
    stit_mask = (tgt_mask + ref_mask).clamp(0,1)

    ref_grid = ref_grid.unsqueeze(0).repeat(b,1,1)
    tgt_grid = tgt_grid.unsqueeze(0).repeat(b,1,1)
    stit_grid = stit_grid.unsqueeze(0).repeat(b,1,1)

    return tgt_grid, tgt_cell, tgt_mask, ref_grid, ref_cell, ref_mask, stit_grid, stit_mask, sizes


def prepare_validation(config):
    spec = config.get('dataset')

    dataset = Neural_Image_Stitching_main.datasetsNIS.datasets.make(spec['dataset'])
    dataset = Neural_Image_Stitching_main.datasetsNIS.datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('dataset: size={}'.format(len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    # sv_file = torch.load(config['resume_stitching'])
    # model = Neural_Image_Stitching_main.models.make(sv_file['model'], load_sd=True).cuda()

    # H_model = Neural_Image_Stitching_main.models.IHN().cuda()
    # sv_file = torch.load(config['resume_align'])
    # H_model.load_state_dict(sv_file['model']['sd'])

    # n_params = nis_utils.compute_num_params(model)

    # log('model params: {}'.format(n_params))

    return loader#, model, H_model


def valid_NIS(loader, model, H_model):
    model.eval()
    H_model.eval()

    tot_psnr = 0
    tot_ssim = 0
    failures = 0

    os.makedirs("visualization/", exist_ok=True)

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)
    print("evaluating start")
    i = 0
    desc = None
    good = 0
    for b_id in pbar:
        i+=1
        batch = next(loader)

        for k, v in batch.items():
            batch[k] = v.cuda()

        ref = batch['inp_ref']
        tgt = batch['inp_tgt']

        b, c, h, w = ref.shape

        with torch.no_grad():
            if h != 128 or w != 128:
                inp_ref = F.interpolate(ref, size=(128,128), mode='bilinear') * 255
                inp_tgt = F.interpolate(tgt, size=(128,128), mode='bilinear') * 255

            else:
                inp_ref = ref * 255
                inp_tgt = tgt * 255
        with torch.no_grad():
            tgt_grid, tgt_cell, tgt_mask, \
            ref_grid, ref_cell, ref_mask, \
            stit_grid, stit_mask, sizes = prepare_ingredient(H_model, inp_tgt, inp_ref, tgt, ref)

        ref = (ref - 0.5) * 2
        tgt = (tgt - 0.5) * 2

        ref_mask = ref_mask.reshape(b,1,*sizes)
        tgt_mask = tgt_mask.reshape(b,1,*sizes)
        
        with torch.no_grad():
            pred = batched_predict(
                model, ref, ref_grid, ref_cell, ref_mask,
                tgt, tgt_grid, tgt_cell, tgt_mask,
                stit_grid, sizes, config['eval_bsize'], seam_cut=True
            )
        pred = pred.permute(0, 2, 1).reshape(b, c, *sizes)
        pred = ((pred + 1)/2).clamp(0,1) * stit_mask.reshape(b, 1, *sizes)
        

        # transforms.ToPILImage()(pred[0]).save('visualization/{0:06d}.png'.format(b_id))
        ref_samples = (ref.cpu().numpy().transpose(0, 2, 3, 1)*2+1)* 255.
        pred_samples = pred.cpu().numpy().transpose(0, 2, 3, 1)* 255.
        ref_mask = ref_mask.cpu().squeeze(0)
        # print(ref_mask.shape, pred_samples.shape)
        try:
            pred_samples = pred_samples[ref_mask.bool()].reshape(b, 512, 512, 3)
            # print(ref_samples.shape, pred_samples.shape)
            psnr = compare_psnr(ref_samples, pred_samples, data_range=255.)
            ssim = compare_ssim(ref_samples.squeeze(), pred_samples.squeeze(), data_range=255., multichannel=True, channel_axis=-1)
            tot_psnr += psnr
            tot_ssim += ssim
            good += 1
        except:
            pred_samples_n = pred_samples.squeeze(0)#.numpy()
            ref_mask = ref_mask.squeeze(0).numpy()
            # print((ref_mask>0).sum(), (tgt_mask.cpu().squeeze(0).squeeze(0).numpy()).sum(), pred_samples_n.shape, ref_mask.shape)
            print("problem with shape")
            # cv2.imwrite("pred.jpg", cv2.cvtColor(pred_samples_n.astype('uint8'), cv2.COLOR_RGB2BGR))
            # cv2.imwrite("ref_m.jpg", (ref_mask * 255).astype('uint8'))
        
        # psnr = compare_psnr(tgt_samples, src_samples, data_range=1.)
        # ssim = compare_ssim(tgt_samples * 255, src_samples * 255, data_range=255.)

        
        torch.cuda.empty_cache()

        pbar.set_description_str(
        desc="PSNR:{:.4f}, SSIM:{:.4f}, Failures:{}".format(
            tot_psnr/(good), tot_ssim/(good), failures), refresh=True)
        
        # if i>10:
        #     break

    print(f"Final PSNR: {tot_psnr / len(pbar):.4f}, SSIM: {tot_ssim / len(pbar):.4f}")
    print('Failure cases:', failures)

    # return tot_psnr / (b_id+1)



# Residula Elastic warp

def make_data_loader(spec, tag=''):
    if spec is None: return None

    dataset = RE_datasets.make(spec['dataset'])
    dataset = RE_datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    return loader

def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21,21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1-ovl_mask)*ref_m[:,0].unsqueeze(1)) * ref_m + ref_m_).clamp(0,1)
    if mask: return mask1

    mask2 = (1-mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit


def eval_REWARP(loader, IHN, model):
    model.eval()

    failures = 0
    tot_mpsnr = 0

    collection_30 = []
    collection_60 = []
    collection_100 = []

    collection_30_SSIM = []
    collection_60_SSIM = []
    collection_100_SSIM = []

    hcell_iter = 6
    tcell_iter = 3
    model.iters_lev = tcell_iter

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    print("evaluating start")
    i = 0
    for b_id in pbar:
        i +=1
        batch = next(loader)

        for k, v in batch.items():
            batch[k] = v.cuda()

        ref_ = batch['inp_ref']
        tgt_ = batch['inp_tgt']
        b, c, h, w = ref_.shape

        if h != 512 or w != 512:
            ref = F.interpolate(ref_, size=(512,512), mode='bilinear')
            tgt = F.interpolate(tgt_, size=(512,512), mode='bilinear')
        else:
            ref = ref_
            tgt = tgt_
        scale = h/ref.shape[-2]

        _, disps, hinp = IHN(ref, tgt, iters_lev0=hcell_iter)


        # Preparation of warped inputs
        H, img_h, img_w, offset = RE_utils.get_warped_coords(disps[-1], scale=(h/512, w/512), size=(h,w))
        H_, *_ = RE_utils.get_H(disps[-1].reshape(ref.shape[0],2,-1).permute(0,2,1), [*ref.shape[-2:]])
        H_ = RE_utils.compens_H(H_, [*ref.shape[-2:]])

        grid = RE_utils.make_coordinate_grid([*ref.shape[-2:]], type=H_.type())
        grid = grid.reshape(1, -1, 2).repeat(ref.shape[0], 1, 1)

        mesh_homography = RE_utils.warp_coord(grid, H_.cuda()).reshape(b,*ref.shape[-2:],-1)
        ones = torch.ones_like(ref_).cuda()
        tgt_w = F.grid_sample(tgt, mesh_homography, align_corners=True)


        # Warp Field estimated by TPS
        with torch.no_grad():
            flows = model(tgt_w, ref, iters=tcell_iter, scale=scale)
        translation = RE_utils.get_translation(*offset)
        T_ref = translation.clone()
        T_tgt = torch.inverse(H).double() @ translation.cuda()

        sizes = (img_h, img_w)
        if img_h > 5000 or img_w > 5000:
            failures += 1
            print('Fail; Evaluated Size: {}X{}'.format(img_h, img_w))
            flows, disps= None, None
            continue


        # Image Alignment
        coord1 = RE_utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_r, _ = RE_utils.gridy2gridx_homography(
            coord1.contiguous(), *sizes, *tgt_.shape[-2:], T_ref.cuda(), cpu=False
        )
        mesh_r = mesh_r.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        coord2 = RE_utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_t, _ = RE_utils.gridy2gridx_homography(
            coord2.contiguous(), *sizes, *tgt_.shape[-2:], T_tgt.cuda(), cpu=False
        )
        mesh_t = mesh_t.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        ovl = torch.round(mask_r * mask_t)
        if ovl.sum() == 0:
            failures += 1
            continue

        flow = flows[-1]/511
        if flow.shape[-2] != 512 or flow.shape[-1] != 512:
            flow = F.interpolate(flow.permute(0,3,1,2), size=(h, w), mode='bilinear').permute(0,2,3,1) * 2

        mesh_t[:, offset[0]:offset[0]+h, offset[1]:offset[1]+w, :] += flow
        ref_w = F.grid_sample(ref_, mesh_r, mode='bilinear', align_corners=True)
        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        tgt_w = F.grid_sample(tgt_, mesh_t, mode='bilinear', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        ref_w = (ref_w + 1)/2 * mask_r
        tgt_w = (tgt_w + 1)/2 * mask_t


        # Image Stitching
        stit = linear_blender(ref_w, tgt_w, mask_r, mask_t)


        # Evaluation
        ovl = (mask_r * mask_t).round().bool()
        pixels = img_h * img_w
        ovls = ovl[:, 0].sum()
        ovl_ratio = ovls / pixels

        ref_ovl = ref_w[ovl].cpu().numpy()
        tgt_ovl = tgt_w[ovl].cpu().numpy()

        psnr = compare_psnr(ref_ovl, tgt_ovl, data_range=1.)
        ssim = compare_ssim(ref_ovl, tgt_ovl, data_range=1., multichannel=True)
        stit += (1-(mask_r + mask_t).clamp(0,1))

        if ovl_ratio <= 0.3: collection_30.append(psnr)
        elif ovl_ratio > 0.3 and ovl_ratio <= 0.6 : collection_60.append(psnr)
        elif ovl_ratio > 0.6: collection_100.append(psnr)
        collections = collection_30 + collection_60 + collection_100

        if ovl_ratio <= 0.3: collection_30_SSIM.append(ssim)
        elif ovl_ratio > 0.3 and ovl_ratio <= 0.6 : collection_60_SSIM.append(ssim)
        elif ovl_ratio > 0.6: collection_100_SSIM.append(ssim)
        collections_SSIM = collection_30_SSIM + collection_60_SSIM + collection_100_SSIM

        pbar.set_description_str(
            desc="[Evaluation] PSNR:{:.4f}/{:.4f}/{:.4f}, Avg PSNR: {:.4f}, [Evaluation] SSIM:{:.4f}/{:.4f}/{:.4f}, SSIM:{:.4f}, Failures:{}".format(
                (sum(collection_30) + 1e-10)/(len(collection_30) + 1e-7),
                (sum(collection_60) + 1e-10)/(len(collection_60) + 1e-7),
                (sum(collection_100) + 1e-10)/(len(collection_100) + 1e-7),
                (sum(collections) + 1e-10)/(len(collections) + 1e-7),
                (sum(collection_30_SSIM) + 1e-10)/(len(collection_30_SSIM) + 1e-7),
                (sum(collection_60_SSIM) + 1e-10)/(len(collection_60_SSIM) + 1e-7),
                (sum(collection_100_SSIM) + 1e-10)/(len(collection_100_SSIM) + 1e-7),
                (sum(collections_SSIM) + 1e-10)/(len(collections_SSIM) + 1e-7),
                failures), refresh=True
        )

        flows, disps= None, None

        # if i>10:
        #     break

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config')
#     parser.add_argument('--gpu', default='0')
#     args = parser.parse_args()

#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#     with open(args.config, 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#         print('config loaded.')

#     main(config, args)



# Main functions

def main_UDIS():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/testing/')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)


def main_IHN():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config')
    # parser.add_argument('--gpu', default='0')
    # args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with open(r"Neural_Image_Stitching_main\configs\test\ihn.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
        print(config)

    loader, model = prepare_eval(config)

    with torch.no_grad():
        eval_IHN(loader, model)


def main_NIS():
    global config, log, writer

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    with open(r"Neural_Image_Stitching_main/configs/test/NIS_blending.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    config_ = r"Neural_Image_Stitching_main/configs/test/NIS_blending.yaml"
    save_path = os.path.join('./save', '_' + config_.split('/')[-1][:-len('.yaml')])
    log, writer = nis_utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # loader, model, H_model = prepare_validation(config)
    loader = prepare_validation(config)
    model, H_model = stitch.prepare_validation(config)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    with torch.no_grad():
        valid_psnr = valid_NIS(loader, model, H_model)


def main_REWARP():

    yaml_path = "Residual_Elastic_Warp_main/configs/test/rewarp.yaml"

    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    sv_file = torch.load("Residual_Elastic_Warp_main/pretrained/rewarp.pth")
    model = RE_models.make(sv_file['T_model'], load_sd=True).cuda()
    H_model = RE_models.models.make(sv_file['H_model'], load_sd=True).cuda()

    loader = make_data_loader(config.get('eval_dataset'), tag='eval')

    with torch.no_grad():
        eval_REWARP(loader, H_model, model)


if __name__ == '__main__':

    # datas = glob.glob(os.path.join("testing", '*'))
    # normalized_paths = [path.replace("\\", "/") for path in datas]
    # print(normalized_paths)
    # print(datas)
    main_UDIS()
    main_IHN()
    # main_NIS()
    main_REWARP()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config')
    # parser.add_argument('--gpu', default='0')
    # args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # with open(args.config, 'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    #     print('config loaded.')

    # main(config, args)
