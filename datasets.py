import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        # Eye-gaze settings
        self.use_eye_gaze = args.use_eye_gaze
        if self.use_eye_gaze:
            self.max_fixations = args.max_fixations
            self.heatmap_size = (224, 224)  # 与图像大小一致
            self.sigma_pixels = 15  # 高斯模糊的标准差(像素)
            self.fixation_data = self._load_fixations(args.fixation_path)
            print(f"Loaded {len(self.fixation_data)} fixation records for heatmap generation")

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def _load_fixations(self, fixation_path):
        """Load and preprocess fixation data"""
        df = pd.read_csv(fixation_path)

        # Group by DICOM_ID
        fixation_dict = {}
        for dicom_id, group in df.groupby('DICOM_ID'):
            # Extract key features: X, Y coordinates and duration
            fixations = group[['FPOGX', 'FPOGY', 'FPOGD']].values
            fixation_dict[dicom_id] = fixations

        return fixation_dict

    def _generate_heatmap_sequence(self, image_id):
        """
        Generate temporal heatmap sequence from fixations
        Returns: (T, H, W) tensor of heatmaps
        """
        H, W = self.heatmap_size

        if image_id in self.fixation_data:
            fixations = self.fixation_data[image_id]
        else:
            # If no fixation data, return zeros
            return torch.zeros(self.max_fixations, H, W), torch.zeros(self.max_fixations)

        heatmaps = []
        num_fix = min(len(fixations), self.max_fixations)

        for i in range(num_fix):
            x, y, duration = fixations[i]

            # 创建单个fixation的热图
            heatmap = np.zeros((H, W))

            # 归一化坐标 [0,1] → 像素坐标
            x_pixel = int(x * W)
            y_pixel = int(y * H)

            if 0 <= x_pixel < W and 0 <= y_pixel < H:
                # 持续时间作为权重
                weight = min(duration / 500.0, 2.0)  # 限制最大权重为2
                heatmap[y_pixel, x_pixel] = weight

                # 高斯模糊
                heatmap = gaussian_filter(heatmap, sigma=self.sigma_pixels)

                # 归一化到 [0, 1]
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()

            heatmaps.append(heatmap)

        # Pad to max_fixations
        while len(heatmaps) < self.max_fixations:
            heatmaps.append(np.zeros((H, W)))

        # Convert to tensor (T, H, W)
        heatmap_tensor = torch.FloatTensor(np.stack(heatmaps[:self.max_fixations]))

        # Create mask
        heatmap_mask = torch.zeros(self.max_fixations)
        heatmap_mask[:num_fix] = 1

        return heatmap_tensor, heatmap_mask

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        if self.use_eye_gaze:
            heatmaps, heatmap_mask = self._generate_heatmap_sequence(image_id)
            sample = (image_id, image, report_ids, report_masks, seq_length, heatmaps, heatmap_mask)
        else:
            sample = (image_id, image, report_ids, report_masks, seq_length)

        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path'][0]

        # 处理路径:如果路径包含p14/p14xxx这样的结构,去掉第一层p14
        # 例如: p14/p14681474/s56673853/xxx.jpg -> p14681474/s56673853/xxx.jpg
        path_parts = image_path.split('/')
        if len(path_parts) > 1 and path_parts[0].startswith('p') and path_parts[1].startswith('p'):
            # 去掉第一个p14层级
            image_path = '/'.join(path_parts[1:])

        full_path = os.path.join(self.image_dir, image_path)
        image = Image.open(full_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        if self.use_eye_gaze:
            heatmaps, heatmap_mask = self._generate_heatmap_sequence(image_id)
            sample = (image_id, image, report_ids, report_masks, seq_length, heatmaps, heatmap_mask)
        else:
            sample = (image_id, image, report_ids, report_masks, seq_length)

        return sample