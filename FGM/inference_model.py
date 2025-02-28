import torch

def create_model(args): 
    from generative.networks.nets.diffusion_model_aniso_unet_AE_official import DiffusionModelUNet_aniso_AE, DiffusionModelEncoder_ansio
    class DIF_AE_SEG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            #256*256*32 -> 128*128*16 -> 64*64*8 -> 32*32*4 -> 16*16*2 
            self.unet =  DiffusionModelUNet_aniso_AE(spatial_dims=3,
                                                    in_channels=2,
                                                    out_channels=1,
                                                    num_channels=[16,32,64,128,256],
                                                    attention_levels=(False,False,False,False,True),
                                                    num_head_channels=[0,0,0,0,32],
                                                    norm_num_groups=16,
                                                    use_flash_attention=True,
                                                    iso_conv_down=(False, False, False, True, None),
                                                    iso_conv_up=(True, False, False, False, None),
                                                    resblock_updown=False,
                                                    num_res_blocks=2)

            self.semantic_encoder = DiffusionModelEncoder_ansio(spatial_dims=3,
                                                                in_channels=1,
                                                                out_channels=1,
                                                                num_channels=[32,64,128,256],
                                                                attention_levels=(False,False,False,False),
                                                                num_head_channels=[0,0,0,0],
                                                                norm_num_groups=32,
                                                                iso_conv_down=(False, False, False, True),
                                                                resblock_updown=False,
                                                                num_res_blocks=(2,2,2,2))


    model = DIF_AE_SEG()


    return model