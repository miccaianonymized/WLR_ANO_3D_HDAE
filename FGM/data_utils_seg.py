import glob
import natsort
from monai import data, transforms

def get_loader(args):
    train_image_list = natsort.natsorted(glob.glob(f'/workspace/dataset/BRAIN_256/normal_final/SET_1/BRAIN_256/images_Tr/train_*/*.nii.gz'))[:]
    train_mask_list = natsort.natsorted(glob.glob(f'/workspace/dataset/BRAIN_256/normal_final/SET_1/BRAIN_256/labels_Tr/train_*/*.nii.gz'))[:]
    
    valid_image_list = natsort.natsorted(glob.glob(f'/workspace/dataset/BRAIN_256/normal_final/SET_1/BRAIN_256/images_Tr/train_*/*.nii.gz'))[-100:]
    valid_mask_list = natsort.natsorted(glob.glob(f'/workspace/dataset/BRAIN_256/normal_final/SET_1/BRAIN_256/labels_Tr/train_*/*.nii.gz'))[-100:]
    
    print("Train [Total]  number = ", len(train_image_list))

    files_tr = [{"image_tr": img_tr, "label_tr": label_tr} for img_tr, label_tr in zip(train_image_list, train_mask_list)]
    files_val = [{"image_val": img_val, "label_val": label_val} for img_val, label_val in zip(valid_image_list, valid_mask_list)]

    tr_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image_tr", "label_tr"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image_tr", "label_tr"]),
            transforms.Orientationd(keys=["image_tr", "label_tr"], axcodes="LPS"),
            transforms.ScaleIntensityRanged(keys=["image_tr"], a_min=-10.0, a_max=90.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.EnsureTyped(keys=["image_tr", "label_tr"]),
            transforms.ToTensord(keys=["image_tr", "label_tr"], track_meta=False)
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image_val", "label_val"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image_val", "label_val"]),
            transforms.Orientationd(keys=["image_val", "label_val"], axcodes="LPS"),
            transforms.ScaleIntensityRanged(keys=["image_val"], a_min=-10.0, a_max=90.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.EnsureTyped(keys=["image_val", "label_val"]),
            transforms.ToTensord(keys=["image_val", "label_val"], track_meta=False)
        ]
    )

    # new_dataset -> Cachenew_dataset
    train_ds = data.Dataset(data = files_tr, transform = tr_transforms)
    val_ds = data.Dataset(data = files_val, transform = val_transforms)

    # else:
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=False
    )
    print("loader is ver (train)")

    loader = [train_loader, val_loader]

    return loader
