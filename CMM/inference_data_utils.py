from monai import data, transforms
import glob
import numpy as np
import os
import re
import natsort
import SimpleITK as sitk

def get_loader_96(args):
    if args.session == "int":
        test_int_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/1_INTERNAL/96/abnormal/*.nii.gz"))[:]
        
        test_int_benign_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/1_INTERNAL/96/benign/*.nii.gz"))[:]
        
        test_int_normal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/1_INTERNAL/96/normal/*.nii.gz"))[:]

        print("Test [int_abnormal]  number = ", len(test_int_abnormal_img))
        print("Test [int_benign]  number = ", len(test_int_benign_img))
        print("Test [int_normal]  number = ", len(test_int_normal_img))
        
        total_img_list = test_int_abnormal_img + test_int_benign_img + test_int_normal_img

    elif args.session == "ext":
        test_ext_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/2_EXTERNAL/96/abnormal/*.nii.gz"))[:]
        
        test_ext_benign_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/2_EXTERNAL/96/benign/*.nii.gz"))[:]
        
        test_ext_normal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/2_EXTERNAL/96/normal/*.nii.gz"))[:]

        print("Test [ext_abnormal]  number = ", len(test_ext_abnormal_img))
        print("Test [ext_benign]  number = ", len(test_ext_benign_img))
        print("Test [ext_normal]  number = ", len(test_ext_normal_img))
        
        total_img_list = test_ext_abnormal_img + test_ext_benign_img + test_ext_normal_img
        
    elif args.session == "hemo":
        test_hemo_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/3_HEMO/96/abnormal/*.nii.gz"))[:]

        print("Test [hemo_abnormal]  number = ", len(test_hemo_abnormal_img))
    
        total_img_list = test_hemo_abnormal_img
        
    elif args.session == "stroke":
        test_stroke_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/10_STROKE/96/abnormal/*.nii.gz"))[:]

        print("Test [stroke_abnormal]  number = ", len(test_stroke_img))
    
        total_img_list = test_stroke_img

    elif args.session == "RSNA":
        test_rsna_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/7_RSNA/96/abnormal/*.nii.gz"))[:]

        print("Test [hemo_abnormal]  number = ", len(test_rsna_abnormal_img))
    
        total_img_list = test_rsna_abnormal_img

    total_list = total_img_list
    
    print("Test [total]  number = ", len(total_list))

    files_ts = [img_ts for img_ts in zip(total_list)]

    ts_transforms = transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="LPS"),
            transforms.ScaleIntensityRange(a_min=-10.0, a_max=90.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.EnsureType(),
            transforms.ToTensor(track_meta=False)
        ]
    )

    # new_dataset -> Cachenew_dataset
    test_ds = data.Dataset(data = files_ts, transform = ts_transforms)

    test_loader = data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
    )

    print("loader is ver (train)")

    loader = test_loader

    return loader, total_list
