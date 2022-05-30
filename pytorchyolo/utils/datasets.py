from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:  # 打开一个txt文件
            self.img_files = file.readlines()  # 把txt文件中的内容读入进来，这个txt中的内容，每一行是一个图片的地址

        self.label_files = []  # 每一个元素都是一个标签地址
        for path in self.img_files:  # 这个循环的目的是由每个图片的地址找到每个图片的groundtruth的地址
            image_dir = os.path.dirname(path)  # 这里是图片所在的文件夹
            label_dir = "labels".join(image_dir.rsplit("images", 1))  # 这里表示的是标签所在的文件夹把image_dir字符串用images分隔开成一个列表（这个列表中存放的是被images分割开来的部分）之后，用lables连接起来
            assert label_dir != image_dir, \
                "Image path must contain a folder named 'images'! \n{}".format(image_dir)  # 结果为非0时候不进行任何操作。
            label_file = os.path.join(label_dir, os.path.basename(path))  # 这里得出的是每一个标签的
            label_file = os.path.splitext(label_file)[0] + '.txt'  # 本行的前部分是把后缀去掉，最终得到的就是我们目前处理的这个图片的groundtruth文件的地址
            self.label_files.append(label_file)  # 每一个存放标签的txt文件的地址加到label_files这个列表中（这里也向我们揭示了，labels中的txt文件是一个txt文件对应一张图片的，而且名字应该对应还有后缀不一样）

        self.img_size = img_size  # 是后面要resize成这个尺寸么？
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):  # 想通过索引来访问对象中的元素的时候，会调用这个函数

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()  # 获得标签所在位置

            # Ignore warning if file is empty
            with warnings.catch_warnings():  # 如果要是没有这个警告的话那就不执行了么？
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
