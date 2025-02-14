import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from monai import transforms
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from diffusers.models.vq_gan_3d import CNN_3D

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--output_dir", type=str, default=None, required=True, help="output dir where files saved.")
# parser.add_argument("--pretrained_cls_path", type=str, default=None, required=True, help="CNN_3D path.")
# parser.add_argument("--resolution", "-r", type=str, required=True, default=0, help="Generation resolution.") # SET
args = parser.parse_args()

class UKB_Dataset(Dataset):
    """
    A PyTorch Dataset for loading UKB images and labels.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): CSV file containing image IDs and labels.
        transform (callable, optional): Optional transform to apply to samples.
    """
    def __init__(self, image_dir, label_dir, transform=None, axis="c"):
        super().__init__()
        self.data_dir = image_dir
        data_csv = pd.read_csv(label_dir)
        self.image_names = list(data_csv['rel_path'])
        self.transform = transform
        self.axis = axis
        self.image_paths = [os.path.join(self.data_dir, name) for name in self.image_names]

        # load conditioning variable: age
        self.ages = data_csv['age'].values.astype(np.float16)
        self.min_age, self.max_age = 45, 81 # self.ages.min(), self.ages.max()
        self.norm_age = (self.ages - self.min_age) / (self.max_age - self.min_age)
        
        # load conditioning variable: sex
        self.gender = data_csv['sex'].values.astype(np.float16)

        # load conditioning variable: ventricular cerebrospinal fluid
        self.vcf = data_csv['p25004_i2'].values.astype(np.float32)
        self.min_vcf, self.max_vcf = self.vcf.min(), self.vcf.max()
        self.norm_vcf = (self.vcf - self.min_vcf) / (self.max_vcf - self.min_vcf)

        # load conditioning variable: brain volume normalized for head size
        self.bv = data_csv['p25009_i2'].values.astype(np.float32)
        self.min_bv, self.max_bv = self.bv.min(), self.bv.max()
        self.norm_bv = (self.bv - self.min_bv) / (self.max_bv - self.min_bv)

        # load conditioning variable: gender
        #self.genders = data_csv['sex'].values.astype(np.float16)
        # self.gender_encoded = []
        # for gender_int in self.genders:
        #     if gender_int == 1.0: # male
        #         self.gender_encoded.append([1.0])
        #     elif gender_int == 0.0: # female
        #         self.gender_encoded.append([0.0])
        #     else:
        #         self.gender_encoded.append([0.0, 0.0])  # Handle unknown gender
        # self.gender_encoded = np.array(self.gender_encoded, dtype=np.float16)
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
        try:
            image = nib.load(image_path) # (182, 218, 182)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = image.get_fdata()
        image = torch.from_numpy(image) # Stays on CPU # (182, 218, 182)

        axes_mapping = {
            's': (0, 1, 2),
            'c': (1, 0, 2),
            'a': (2, 1, 0)
        }

        try:
            image = image.permute(*axes_mapping[self.axis]).unsqueeze(0) # (1, 218, 182, 182)
        except KeyError:
            raise ValueError("axis must be one of 'a', 'c', or 's'.")

        # Define target size
        # target_size = (128, 128, 128)  # (D₂, H₂, W₂)
        # Resize the volume using trilinear interpolation
        # image = F.interpolate(
        #     image,
        #     size=target_size,
        #     mode='trilinear',
        #     align_corners=False
        # )  # Shape: (1, 1, D₂, H₂, W₂)
        # Remove the batch dimension
        # image = image.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1, 128, 128, 128)

        age = torch.tensor([self.norm_age[index]], dtype=torch.float16) # Shape: [1]
        sex = torch.tensor([self.gender[index]], dtype=torch.float16)  # Shape: [1]
        vcf = torch.tensor([self.norm_vcf[index]], dtype=torch.float16)  # Shape: [1]
        bv = torch.tensor([self.norm_bv[index]], dtype=torch.float16)  # Shape: [1]
        cond_tensor = torch.cat([age, sex, vcf, bv], dim=-1)  # Shape: [4]

        sample = {
            "pixel_values": image.to(torch.float16),
            "condition": cond_tensor,
            "eid": image_path, ###
        }

        if self.transform:
            sample = self.transform(sample)
        del image

        return sample

#args.pretrained_cls_path = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/M0_AGE_TRANS/checkpoint-110000" 
args.pretrained_cls_path = "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/M0_AGE_NAIVE/checkpoint-130000"
args.data_dir = "/leelabsg/data/20252_unzip"
args.test_label_dir = "/shared/s1/lab06/wonyoung/diffusers/sd3/data/test_cls.csv"
args.resolution = "128,128,128"
args.output_dir = ""
args.axis = "c"

resolution = args.resolution # SET #"128,64,64" "224,40,40"
input_size = tuple(int(x) for x in resolution.split(","))
test_transforms = transforms.Compose(
        [
            transforms.Resized(keys=["pixel_values"], spatial_size=input_size, size_mode="all"),
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0), # SET
            # transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0), # SET
            # transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0), # SET
            transforms.ToTensord(keys=["pixel_values"]),
        ]
    )

test_dataset = UKB_Dataset(args.data_dir, args.test_label_dir, transform=test_transforms, axis=args.axis)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=1,
    num_workers=4,
    pin_memory=True
)

pretrained_cls_path = args.pretrained_cls_path
mixed_precision = "fp16"
weight_dtype = torch.float16
accelerator = Accelerator(mixed_precision=mixed_precision)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls_model = CNN_3D.from_pretrained(pretrained_cls_path, subfolder="cls_model",).to(device)
cls_model.eval()
cls_model = accelerator.prepare(cls_model)
cls_model = accelerator.unwrap_model(cls_model)

loss_fn = nn.L1Loss()

print("Inference starts")
loss_dict = dict()
with torch.no_grad():
    test_loss = 0.0
    for step, batch in enumerate(test_dataloader): # 2525
        #with torch.autocast(accelerator.device.type, dtype=weight_dtype):
        x = batch["pixel_values"].to(weight_dtype).to(device)
        target = batch["condition"][:, 0].to(weight_dtype).to(device)
        
        pred = cls_model(x)
        loss = loss_fn(pred.squeeze(dim=1), target)

        key = batch["eid"][0].split("/")[4][:7]
        print(f"Test {step}, eid: {key}, age: {int(target*(81-45)+45)}")        

        loss_dict[key] = loss.item()
        test_loss += loss

    print(test_loss)
    print("Test MAE:", test_loss.item()*(81-45) / len(test_dataloader)) 
    # NAIVE 2.4949
    # TRANS 2.7752


# match loss and eid
df = pd.DataFrame.from_dict(loss_dict, orient="index", columns=["Value"])
df.reset_index(inplace=True)
df.columns = ["eid", "test_loss"]  # Rename columns
df["eid"] = df["eid"].astype(str)
df["test_loss_real"] = df["test_loss"]*(81-45)

test_df = pd.read_csv("/shared/s1/lab06/wonyoung/diffusers/sd3/data/test_cls.csv")
test_df["eid"] = test_df["eid"].astype(str)
test_df["age"] = test_df["age"].astype(int)
merge_df = pd.merge(
    test_df,
    df,
    on="eid",
    how="inner"
)
result1 = merge_df.groupby("age")["test_loss"].mean().reset_index()
result2 = merge_df.groupby("age")["test_loss_real"].mean().reset_index() 
# NAIVE: large loss in 46-49 & 74-81 "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/M0_AGE_NAIVE/checkpoint-130000"
# TRANS: large loss in 46-49 & 72-81 "/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/M0_AGE_TRANS/checkpoint-110000" 
