import os
import sys
sys.path.append(os.path.abspath('/workspace/WRL_ENS_3D_HDAE'))
from inference_model import create_model
from inference_data_utils import *
from monai.utils import misc
from ema_pytorch import EMA
import SimpleITK as sitk 
import torch.nn.parallel
import argparse
import torch
import ast

def args_as_list(s):
    v = ast.literal_eval(s)
    return v

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='HDAE-FGM inference pipeline')
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument('--image_size', default=96, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--infer_epoch', default=10, type=int)
parser.add_argument('--session', default=None, type=str)

def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    misc.set_determinism(seed=2024)

    main_worker_enc_256(args=args)

def save_image_256(args, pred, images_ts, img_path_list, i, times):
    ori_img = sitk.ReadImage(img_path_list[i])

    images_ts = images_ts.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    images_ts = images_ts.clip(min=0.1, max=0.9)
    
    pred_img = pred.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    pred_img = pred_img.clip(min=0.1, max=0.9)

    save_ori = sitk.GetImageFromArray(images_ts)
    save_pred = sitk.GetImageFromArray(pred_img)
    
    save_pred.CopyInformation(ori_img)
    save_ori.CopyInformation(ori_img)

    sitk.WriteImage(save_pred, img_path_list[i].replace("/IMG/", "/IMG_infer/").replace(".nii.gz", f"_256_{str(times)}_pred.nii.gz"))
    sitk.WriteImage(save_ori, img_path_list[i].replace("/IMG/", "/IMG_infer/").replace(".nii.gz", f"_256_ori.nii.gz"))

def main_worker_enc_256(args):
    from generative.networks.schedulers import DDIMScheduler
    from generative.inferers import DiffusionInferer_ae_seg
    
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    args.image_size = 256
    
    loader, img_path_list = get_loader_256(args)

    model = create_model(args)
    model.to(device)

    scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta")
    inferer = DiffusionInferer_ae_seg(scheduler)
    ema_model = EMA(model, 
                    beta=0.995, 
                    update_after_step=15000,
                    update_every=10)

    a_path = f'/workspace/model_10_final.pt'
    weight = torch.load(a_path, map_location='cpu')
    weight['ema']['step'] = torch.ones([1])
    weight['ema']['initted'] = torch.ones([1])
    ema_model.load_state_dict(weight['ema'])
    print(a_path)

    ema_model.ema_model.eval()
    infer_time_list = [500]

    with torch.no_grad():
        a = scheduler.alphas_cumprod**0.5
        b = (1 - scheduler.alphas_cumprod)**0.5
        
        for idx, batch_data in enumerate(loader):
            images_ts = batch_data['image_ts'].to('cuda')
            labels_ts = batch_data['label_ts'].to('cuda')
            latent_ema = ema_model.ema_model.semantic_encoder(images_ts)
            
            _, _, H, W, D = images_ts.shape
            noise = torch.randn((1, 1, H, W, D)).to('cuda')
            
            for times in infer_time_list:
                image = a[times]*images_ts + b[times]*noise
                image = image.to("cuda")

                scheduler.set_timesteps(num_inference_steps=times)

                image_pred_ema = inferer.sample(input_noise=image, 
                                                diffusion_model=ema_model.ema_model.unet, 
                                                scheduler=scheduler, 
                                                save_intermediates=False,
                                                conditioning=labels_ts,
                                                cond=latent_ema)
                
                for batch_save in range(len(images_ts[:,0,0,0,0])):
                    save_image_256(args, image_pred_ema[batch_save:batch_save+1,:,:,:,:], images_ts[batch_save:batch_save+1,:,:,:,:], img_path_list, idx*args.batch_size + batch_save, times)
        
if __name__ == '__main__':
    main()