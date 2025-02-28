import torch

def create_model(args): 
    if args.image_size == 96:
        from generative.networks.nets.diffusion_model_aniso_unet_AE_official import DiffusionModelUNet_aniso_AE, DiffusionModelEncoder_ansio
        class DIF_AE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                #96*96*32 -> 48*48*32 -> 24*24*32 -> 12*12*32
                self.unet =  DiffusionModelUNet_aniso_AE(spatial_dims=3,
                                                        in_channels=1,
                                                        out_channels=1,
                                                        num_channels=[32,128,256,512],
                                                        attention_levels=(False,False,False,True),
                                                        num_head_channels=[0,0,0,32],
                                                        norm_num_groups=32,
                                                        use_flash_attention=True,
                                                        iso_conv_down=(False, False, False, None),
                                                        iso_conv_up=(False, False, False, None),
                                                        resblock_updown=False,
                                                        num_res_blocks=2)
                                        
                self.semantic_encoder = DiffusionModelEncoder_ansio(spatial_dims=3,
                                                                    in_channels=1,
                                                                    out_channels=1,
                                                                    num_channels=[32,128,256,512],
                                                                    attention_levels=(False,False,False,False),
                                                                    num_head_channels=[0,0,0,0],
                                                                    norm_num_groups=32,
                                                                    iso_conv_down=(False, False, False, False),
                                                                    resblock_updown=False,
                                                                    num_res_blocks=(2,2,2,2))
            def forward(self, xt, x_cond, t):
                latent = self.semantic_encoder(x_cond)
                noise_pred = self.unet(x=xt, timesteps=t, context=latent.unsqueeze(2))
                return noise_pred, latent
                                                            
        model = DIF_AE()


    return model