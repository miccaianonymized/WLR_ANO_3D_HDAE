import glob
import json
import natsort
from monai import data, transforms

def get_loader_256(args):
    if args.session == "int":
        test_int_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/1_INTERNAL/256/abnormal/*.nii.gz"))[:]
        test_int_abnormal_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/1_INTERNAL/256/abnormal/*.nii.gz"))[:]
        
        test_int_benign_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/1_INTERNAL/256/benign/*.nii.gz"))[:]
        test_int_benign_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/1_INTERNAL/256/benign/*.nii.gz"))[:]
        
        test_int_normal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/1_INTERNAL/256/normal/*.nii.gz"))[:]
        test_int_normal_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/1_INTERNAL/256/normal/*.nii.gz"))[:]

        print("Test [int_abnormal]  number = ", len(test_int_abnormal_img), len(test_int_abnormal_mask))
        print("Test [int_benign]  number = ", len(test_int_benign_img), len(test_int_benign_mask))
        print("Test [int_normal]  number = ", len(test_int_normal_img), len(test_int_normal_mask))
        
        total_img_list = test_int_abnormal_img + test_int_benign_img + test_int_normal_img
        total_mask_list = test_int_abnormal_mask + test_int_benign_mask + test_int_normal_mask

    elif args.session == "ext":
        test_ext_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/2_EXTERNAL/256/abnormal/*.nii.gz"))[:]
        test_ext_abnormal_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/2_EXTERNAL/256/abnormal/*.nii.gz"))[:]
        
        test_ext_benign_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/2_EXTERNAL/256/benign/*.nii.gz"))[:]
        test_ext_benign_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/2_EXTERNAL/256/benign/*.nii.gz"))[:]
        
        test_ext_normal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/2_EXTERNAL/256/normal/*.nii.gz"))[:]
        test_ext_normal_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/2_EXTERNAL/256/normal/*.nii.gz"))[:]

        print("Test [ext_abnormal]  number = ", len(test_ext_abnormal_img), len(test_ext_abnormal_mask))
        print("Test [ext_benign]  number = ", len(test_ext_benign_img), len(test_ext_benign_mask))
        print("Test [ext_normal]  number = ", len(test_ext_normal_img), len(test_ext_normal_mask))
        
        total_img_list = test_ext_abnormal_img + test_ext_benign_img + test_ext_normal_img
        total_mask_list = test_ext_abnormal_mask + test_ext_benign_mask + test_ext_normal_mask

    elif args.session == "hemo":
        test_hemo_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/3_HEMO/256/abnormal/*.nii.gz"))[:]
        test_hemo_abnormal_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/3_HEMO/256/abnormal/*.nii.gz"))[:]

        print("Test [hemo_abnormal]  number = ", len(test_hemo_abnormal_img), len(test_hemo_abnormal_mask))
    
        total_img_list = test_hemo_abnormal_img
        total_mask_list = test_hemo_abnormal_mask

    elif args.session == "hemo_n":
        test_hemo_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/3_HEMO/256/normal/*.nii.gz"))[:]
        test_hemo_abnormal_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/3_HEMO/256/normal/*.nii.gz"))[:]

        print("Test [hemo_normal]  number = ", len(test_hemo_abnormal_img), len(test_hemo_abnormal_mask))
    
        total_img_list = test_hemo_abnormal_img
        total_mask_list = test_hemo_abnormal_mask
        
    elif args.session == "stroke":
        test_stroke_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/10_STROKE/256/abnormal/*.nii.gz"))[:]
        test_stroke_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/10_STROKE/256/abnormal/*.nii.gz"))[:]

        print("Test [stroke_abnormal]  number = ", len(test_stroke_img), len(test_stroke_mask))
    
        total_img_list = test_stroke_img
        total_mask_list = test_stroke_mask

    elif args.session == "RSNA":
        test_rsna_abnormal_img = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/IMG/7_RSNA/256/abnormal/*.nii.gz"))[:]
        test_rsna_abnormal_mask = natsort.natsorted(glob.glob("/workspace/dataset/BRAIN_TEST_FINAL/MASK/7_RSNA/256/abnormal/*.nii.gz"))[:]

        print("Test [hemo_abnormal]  number = ", len(test_rsna_abnormal_img), len(test_rsna_abnormal_mask))
    
        total_img_list = test_rsna_abnormal_img
        total_mask_list = test_rsna_abnormal_mask

    print("Test [total]  number = ", len(total_img_list), len(total_mask_list))


    files_ts = [{"image_ts": img_ts, "label_ts": label_ts} for img_ts, label_ts in zip(total_img_list, total_mask_list)]

    ts_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image_ts", "label_ts"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image_ts", "label_ts"]),
            transforms.Orientationd(keys=["image_ts", "label_ts"], axcodes="LPS"),
            transforms.ScaleIntensityRanged(keys=["image_ts"], a_min=-10.0, a_max=90.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.EnsureTyped(keys=["image_ts", "label_ts"]),
            transforms.ToTensord(keys=["image_ts", "label_ts"], track_meta=False)
        ]
    )

    # new_dataset -> Cachenew_dataset
    test_ds = data.Dataset(data = files_ts, transform = ts_transforms)

    # else:
    test_loader = data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False
    )

    print("loader is ver (train)")

    loader = test_loader

    return loader, total_img_list