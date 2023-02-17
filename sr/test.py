import argparse
import os, pdb
import torch, cv2
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time, math, glob
import scipy.io as sio
from PIL import Image
from ssim import calculate_ssim_floder
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./checkpoint/real/model_super_l05_200_0.1.pth", type=str, help="model path")
parser.add_argument("--dataset", default="../super_resolution/real_res/test", type=str, help="noisy dataset name")
parser.add_argument("--GT", default="../super_resolution/real_res/test_gt", type=str, help="ground truth dataset name")
parser.add_argument("--save", default="./results_real2", type=str, help="savepath, Default: results")
# parser.add_argument("--save_up", default="./bicubic", type=str, help="savepath, Default: results")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt 
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

model = torch.load(opt.model)["model"]

image_list = glob.glob(opt.dataset+"/*.*") 
p=0
# p2=0
with torch.no_grad():
    for image_name in image_list:
        name = image_name.split('\\')
        print("Processing ", image_name)
        im_n = Image.open(image_name)

        im_n=np.array(im_n)
        im_gt = Image.open(opt.GT+'/'+name[-1])
        im_gt=np.array(im_gt)

        # print(im_n.shape)
     
        #pdb.set_trace()
        # im_n = np.expand_dims(im_n, 0)
        
        im_n = np.transpose(im_n, (2,0,1))
        im_n = np.expand_dims(im_n, 0)
        im_n = torch.from_numpy(im_n).float()/255

        im_gt = np.transpose(im_gt, (2,0,1))
        im_gt = np.expand_dims(im_gt, 0)
        im_gt = torch.from_numpy(im_gt).float()/255

        im_input = Variable(im_n)
        im_gt = Variable(im_gt)

        if cuda:
            model = model.cuda()
            im_gt=im_gt.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()

        height = int(im_input.size()[2])
        width = int(im_input.size()[3])
        M = int(height / 64)  # 行能分成几组
        N = int(width / 64)

        im_input = im_input[:,:, :M * 64, :N * 64]
        im_gt = im_gt[:,:, :M * 64, :N * 64]

        im_output = model(im_input)

        pp=PSNR(im_output,im_gt)
        # pp2=PSNR(im_input_up,im_gt)
        p+=pp
        # p2+=pp2
        
        #HR_4x = HR[:,:,:,:,0].cpu()
        im_output = im_output.cpu()
        # im_input_up = im_input_up.cpu()
        # save_image(im_output.data,'6.png')
        save_image(im_output.data,opt.save+'/'+name[-1])
        # save_image(im_input_up.data,opt.save_up+'/'+name[-1])

ssim=calculate_ssim_floder(opt.GT,opt.save)
# ssim_in=calculate_ssim_floder(opt.GT,opt.save_up)
print("Average PSNR:",p/len(image_list))
# print("Average input PSNR:",p2/len(image_list))
print("Average SSIM:",ssim)
# print("Average input SSIM:",ssim_in)
# print(calculate_ssim_floder(opt.GT,opt.dataset))
