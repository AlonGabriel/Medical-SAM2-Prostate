import os
import numpy as np
import torch
import pydicom
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from func_3d.utils import random_click, generate_bbox


class ProstateMRI(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None, variation=0):
        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'images'))

        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        self.video_length = 60  # As each patient has 60 frames

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)  # Ensure this matches the model's expected input size

        # Get the images and masks
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        mask_path = os.path.join(self.data_path, self.mode, 'masks', name)

        # Get all .dcm files in the image path
        img_files = sorted([f for f in os.listdir(img_path) if f.endswith('.dcm')])
        mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.dcm')])

        # Make sure we have the correct number of frames (or handle fewer frames)
        num_frames = min(self.video_length, len(img_files))

        # Load all frames for the patient
        img_tensor = torch.zeros(num_frames, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(num_frames):
            # Load the DICOM image
            img_dcm = pydicom.dcmread(os.path.join(img_path, img_files[frame_index]))
            img = img_dcm.pixel_array.astype(np.uint8)

            # Convert to PIL image and resize
            img = Image.fromarray(img).convert('RGB')
            img = img.resize(newsize)

            # Convert to tensor and adjust dimensions
            img = torch.tensor(np.array(img)).float().permute(2, 0, 1)
            
            img_tensor[frame_index, :, :, :] = img

        # Load and handle the segmentation masks
        for mask_file in mask_files:
            # Load the DICOM mask
            mask_dcm = pydicom.dcmread(os.path.join(mask_path, mask_file))
            mask = mask_dcm.pixel_array.astype(np.uint8)

            # Check if it's a prostate or lesion mask (both have 60 slices)
            if mask.shape == (60, 256, 256):
                # Resize each slice to match [H*16, W*16]
                for slice_idx in range(num_frames):
                    mask_slice = mask[slice_idx, :, :]
                    obj_mask = Image.fromarray(mask_slice).resize((self.img_size * 16, self.img_size * 16))
                    obj_mask = torch.tensor(np.array(obj_mask)).float().unsqueeze(0)  # Add channel dimension [1, H*16, W*16]
                    mask_dict[slice_idx] = {'mask': obj_mask}

                    # Generate prompts (click or bbox)
                    if self.prompt == 'click':
                        point_label, pt = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                        pt_dict[slice_idx] = {'pt': pt}
                        point_label_dict[slice_idx] = {'point_label': point_label}
                    elif self.prompt == 'bbox':
                        bbox = generate_bbox(np.array(mask_slice).squeeze(), variation=self.variation, seed=self.seed)
                        if bbox is not None:
                            bbox_dict[slice_idx] = {'bbox': bbox}
                    else:
                        raise ValueError('Prompt not recognized')

        image_meta_dict = {'filename_or_obj': name}
        if self.prompt == 'bbox':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict': image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image': img_tensor,
                'label': mask_dict,
                'p_label': point_label_dict,
                'pt': pt_dict,
                'image_meta_dict': image_meta_dict,
            }