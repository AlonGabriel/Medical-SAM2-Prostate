import os
import numpy as np
import torch
import pydicom
from PIL import Image
from torch.utils.data import Dataset

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

        # Get the images and masks
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        mask_path = os.path.join(self.data_path, self.mode, 'masks', name)

        # Load all frames for the patient
        img_tensor = torch.zeros(self.video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(self.video_length):
            # Load the DICOM image
            img_dcm = pydicom.dcmread(os.path.join(img_path, f'{frame_index + 1}.dcm'))
            img = img_dcm.pixel_array
            img = Image.fromarray(img).convert('RGB')
            if self.transform:
                img = self.transform(img)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
            img_tensor[frame_index, :, :, :] = img

            # Load the segmentation masks (3 types)
            diff_obj_mask_dict = {}
            for mask_type in ['lesion', 'prostate', 'aimi']:
                mask_dcm = pydicom.dcmread(os.path.join(mask_path, f'{mask_type}_{frame_index + 1}.dcm'))
                mask = mask_dcm.pixel_array
                obj_mask = Image.fromarray(mask)
                if self.transform_msk:
                    obj_mask = self.transform_msk(obj_mask)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                diff_obj_mask_dict[mask_type] = obj_mask

                # Generate prompts (click or bbox)
                if self.prompt == 'click':
                    point_label, pt = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=self.seed)
                    pt_dict[frame_index] = pt
                    point_label_dict[frame_index] = point_label
                elif self.prompt == 'bbox':
                    bbox_dict[frame_index] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
                else:
                    raise ValueError('Prompt not recognized')

            mask_dict[frame_index] = diff_obj_mask_dict

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
