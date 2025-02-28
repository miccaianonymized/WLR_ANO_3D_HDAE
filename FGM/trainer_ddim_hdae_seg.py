import os
import torch
from tqdm import tqdm
import SimpleITK as sitk 
from pathlib import Path
from einops import reduce
from ema_pytorch import EMA
import torch.nn.functional as F
from generative.inferers import DiffusionInferer_ae_seg
from generative.networks.schedulers import DDIMScheduler

def train_ddim_hdae_epoch_seg(model,
                              ema_model,
                              scheduler,
                              inferer,
                              train_loader,
                              optimizer,
                              epoch,
                              args):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for idx, batch_data in progress_bar:
        images = batch_data["image_tr"].to('cuda')
        labels = batch_data["label_tr"].to('cuda')

        optimizer.zero_grad(set_to_none=True)

        noise = torch.randn_like(images).to('cuda')

        timesteps = torch.randint(0, 550, (images.shape[0],), device=images.device).long()

        latent = model.semantic_encoder(images)
        
        noise_pred = inferer(inputs=images, 
                             diffusion_model=model.unet, 
                             noise=noise, 
                             timesteps=timesteps,
                             condition=labels,
                             cond = latent)
        
        loss = F.mse_loss(noise_pred.float(), noise.float())

        loss.backward()
    
        optimizer.step()
        ema_model.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / (idx + 1)})
            
    return epoch_loss / (idx + 1)


def val_ddim_hdae_epoch_seg(model,
                            ema_model,
                            inferer,
                            val_loader, 
                            scheduler,
                            epoch,
                            args):
    
    model.eval()
    ema_model.ema_model.eval()
    val_epoch_loss = 0
    val_epoch_loss_ema = 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            images = batch_data["image_val"].to('cuda')
            labels = batch_data["label_val"].to('cuda')
            
            timesteps = torch.randint(0, 550, (images.shape[0],), device=images.device).long()
            
            noise = torch.randn_like(images).to('cuda')

            latent = model.semantic_encoder(images)

            noise_pred = inferer(inputs=images, 
                                 diffusion_model=model.unet, 
                                 noise=noise, 
                                 timesteps=timesteps,
                                 condition=labels,
                                 cond = latent)

            val_loss = F.mse_loss(noise_pred.float(), noise.float())

            latent_ema = ema_model.ema_model.semantic_encoder(images)

            noise_pred_ema = inferer(inputs=images, 
                                     diffusion_model=ema_model.ema_model.unet, 
                                     noise=noise, 
                                     timesteps=timesteps,
                                     condition=labels,
                                     cond = latent_ema)

            val_loss_ema = F.mse_loss(noise_pred_ema.float(), noise.float())

        val_epoch_loss += val_loss.item()
        val_epoch_loss_ema += val_loss_ema.item()

    print({"val_loss": val_epoch_loss / (idx + 1)})
    print({"val_loss_ema": val_epoch_loss_ema / (idx + 1)})

    a = scheduler.alphas_cumprod**0.5
    b = (1 - scheduler.alphas_cumprod)**0.5

    _, _, H, W, D = images.shape
    noise = torch.randn((1, 1, H, W, D)).to('cuda')
    image = a[550]*images + b[550]*noise
    image = image.to("cuda")

    scheduler.set_timesteps(num_inference_steps=550)

    image_pred = inferer.sample(input_noise=image,
                                diffusion_model=model.unet, 
                                scheduler=scheduler, 
                                save_intermediates=False,
                                conditioning=labels,
                                cond=latent)

    image_pred_ema = inferer.sample(input_noise=image, 
                                    diffusion_model=ema_model.ema_model.unet, 
                                    scheduler=scheduler, 
                                    save_intermediates=False,
                                    conditioning=labels,
                                    cond=latent_ema)

    save_image(images, 'ori', args, epoch)
    save_image(image_pred, 'val', args, epoch)
    save_image(image_pred_ema, 'val_ema', args, epoch)
    
    return val_epoch_loss / (idx + 1)

def save_checkpoint(model,
                    ema_model,
                    epoch,
                    ckpt_logdir,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    
    state_dict = model.state_dict()
    state_dict_ema = ema_model.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict,
            'ema':state_dict_ema
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(ckpt_logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

def save_image(pred, phase, args, epoch):
    pred_img = pred.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    pred_img = pred_img.clip(min=0.1, max=0.9)
    save_pred = sitk.GetImageFromArray(pred_img)

    sitk.WriteImage(save_pred, f"{args.img_save_dir}/{phase}_{epoch}_pred.nii.gz")

def fix_optimizer(optimizer):
    # Optimizer Error fix...!
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

import json
import shutil
def run_ddim_hdae_training_seg(model,
                               train_loader,
                               val_loader,
                               optimizer,
                               lr_scheduler,
                               args,
                               ):
    val_loss_max = 1.

    scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta")

    inferer = DiffusionInferer_ae_seg(scheduler)
        
    n_epochs = args.max_epochs
    
    ema_model = EMA(model, 
                    beta=0.995, 
                    update_after_step=15000,
                    update_every=10)
    
    for epoch in range(n_epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"lr : {lr}")        

        epoch_tr_loss = train_ddim_hdae_epoch_seg(model,
                                                  ema_model,
                                                  scheduler,
                                                  inferer,
                                                  train_loader,
                                                  optimizer,
                                                  epoch,
                                                  args)

        log_stats = {f'train_{epoch}': epoch_tr_loss}
        with (Path(args.log_dir) / 'log.txt').open('a') as f:
            f.write(json.dumps(log_stats) + "\n")

        lr_scheduler.step()

        b_new_best = False
        epoch_val_loss = val_ddim_hdae_epoch_seg(model,
                                                 ema_model,
                                                 inferer,
                                                 val_loader,
                                                 scheduler,
                                                 epoch,
                                                 args=args
                                                 )
        
        print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), 'val_loss', epoch_val_loss)

        save_checkpoint(model,
                        ema_model,
                        epoch,
                        args.log_dir,
                        best_acc=epoch_val_loss,
                        filename=f'model_{epoch}_final.pt',
                        optimizer=optimizer,
                        scheduler=lr_scheduler)

        if epoch_val_loss < val_loss_max:
            print('new best ({:.6f} --> {:.6f}). '.format(val_loss_max, epoch_val_loss))
            val_loss_max = epoch_val_loss
            b_new_best = True
        
        if b_new_best:
            print('Copying to model.pt new best model!!!!')
            shutil.copyfile(os.path.join(args.log_dir, f'model_{epoch}_final.pt'), os.path.join(args.log_dir, 'model_best.pt'))

        print('Training Finished !, Best Accuracy: ', val_loss_max)

    return val_loss_max