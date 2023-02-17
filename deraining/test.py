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
parser.add_argument("--model", default="./checkpoint/model_denoise_light_50_100.pth", type=str, help="model path")
parser.add_argument("--dataset", default="./rain_data_test_light", type=str, help="noisy dataset name")
parser.add_argument("--GT", default="./norain", type=str, help="ground truth dataset name")
parser.add_argument("--save", default="./results2", type=str, help="savepath, Default: results")
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
p2=0
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
        M = int(height / 16)  # 行能分成几组
        N = int(width / 16)
        im_input = im_input[:,:, :M * 16, :N * 16]
        im_input = torch.transpose(im_input,2,3)
        im_gt = im_gt[:,:, :M * 16, :N * 16]
        # im_output = torch.zeros(3, M * 64, N * 64)
        
        # print(im_input.shape)
        # for i in range(M):
        #     for j in range(N):
        #         im_output[:, i * 64:i * 64 + 64, j * 64:j * 64 + 64] = model(im_input[:, :, i * 64:i * 64 + 64, j * 64:j * 64 + 64]).squeeze()
        # print(im_output.shape)
        im_output = model(im_input)
        # print(im_input.shape,im_output.shape,im_gt.shape)
        im_output = torch.transpose(im_output,2,3)
        im_input = torch.transpose(im_input,2,3)
        pp=PSNR(im_output,im_gt)
        pp2=PSNR(im_input,im_gt)
        p+=pp
        p2+=pp2
        
        #HR_4x = HR[:,:,:,:,0].cpu()
        im_output = im_output.cpu()
        # save_image(im_output.data,'6.png')
        save_image(im_output.data,opt.save+'/'+name[-1])

ssim=calculate_ssim_floder(opt.GT,opt.save)
# ssim_in=calculate_ssim_floder(opt.GT,opt.dataset)
print("Average PSNR:",p/len(image_list))
print("Average input PSNR:",p2/len(image_list))
print("Average SSIM:",ssim)
# print("Average input SSIM:",ssim_in)
# print(calculate_ssim_floder(opt.GT,opt.dataset))
