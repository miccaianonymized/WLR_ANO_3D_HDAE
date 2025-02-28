import glob
import natsort
from monai import data, transforms

def get_loader(args):
    train_real = natsort.natsorted(glob.glob(f'/workspace/dataset/BRAIN_256/normal_final/SET_1/BRAIN_96/images_Tr/train_*/*.nii.gz'))[:]
    valid_real = natsort.natsorted(glob.glob(f'/workspace/dataset/BRAIN_256/normal_final/SET_1/BRAIN_96/images_Tr/train_*/*.nii.gz'))[-100:]

    print("Train [Total]  number = ", len(train_real))

    files_tr = [img_tr for img_tr in zip(train_real)]

    files_val = [img_val for img_val in zip(valid_real)]

    tr_transforms = transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="LPS"),
            transforms.ScaleIntensityRange(a_min=-10.0, a_max=90.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.EnsureType(),
            transforms.ToTensor(track_meta=False)
        ]
    )
    val_transforms = transforms.Compose(
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
        shuffle=True,
        pin_memory=False
    )
    print("loader is ver (train)")

    loader = [train_loader, val_loader]

    return loader, train_real
