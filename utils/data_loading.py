import logging
import numpy as np
import torch
from PIL import Image
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename, is_mask=False):
    ext = splitext(filename)[1]
    if ext == '.npy':
        img = Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        img = Image.fromarray(torch.load(filename).numpy())
    else:
        img = Image.open(filename)

    # 修改部分：如果是掩码，转换为灰度模式
    if is_mask:
        img = img.convert('L')
    else:
        img = img.convert('RGB')  # 保持输入图像为彩色
    return img

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    # 修改部分：在加载掩码时，指定 is_mask=True
    mask = np.asarray(load_image(mask_file, is_mask=True))
    # 由于掩码是灰度图像，返回唯一值
    return np.unique(mask)

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
        # 添加调试信息
        #for i, u in enumerate(unique):
            #print(f"Mask {i}: shape = {u.shape}")

        # 修改部分：由于所有的 unique 都是一维数组，直接连接即可
        self.mask_values = list(sorted(np.unique(np.concatenate(unique)).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = 256, 256  # Resize to 256x256
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                mask[img == v] = i
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        
        # 修改部分：指定 is_mask 参数
        mask = load_image(mask_file[0], is_mask=True)
        img = load_image(img_file[0], is_mask=False)
        
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'name': name
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

# utils/data_loading.py

# utils/data_loading.py

from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class BasicDataset_test(Dataset):
    def __init__(self, images_dir, masks_dir, img_size):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        
        # 定义图像和掩码的转换
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.images_dir, self.images[idx])
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # 假设mask是灰度图

            image = self.transform(image)
            mask = self.mask_transform(mask)
            mask = mask.long().squeeze(0)  # 转换为整数类型并移除通道维度

            return {"image": image, "mask": mask}
        except Exception as e:
            logging.error(f'Error loading image {self.images[idx]}: {e}')
            # 你可以选择返回一个空的样本，或是跳过这个样本
            # 这里选择跳过，返回下一个有效的样本
            return self.__getitem__((idx + 1) % len(self))

