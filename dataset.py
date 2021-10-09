import cv2
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import glob
import numpy as np
import traceback
from shutil import move
"""
自定义数据集要点：
1.构建transforms.Compose，以便预处理数据集
2.利用torch的Dataset类，构建数据集。构建的过程需要注意：
    1.实现__len__和__getitem__方法，其中，__getitem__(self, index)按索引映射到对应的数据， __len__(self)则会返回这个数据集的长度
    2.可以选择先对图片数据集进行resize操作
"""

tran = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def redefine_dataset(content_dir, style_dir, re_size, source_dir):
    content_imgs = glob.glob(os.path.join(content_dir, "*.jpg"))
    style_imgs = glob.glob(os.path.join(style_dir, "*.jpg"))
    style_imgs += glob.glob(os.path.join(style_dir, "*.png"))

    reset_content_dir = os.path.join(content_dir, "content_images")
    reset_style_dir = os.path.join(style_dir, "style_images")

    if not (os.path.exists(reset_content_dir) and
            os.path.exists(reset_style_dir)):
        os.mkdir(reset_content_dir)
        os.mkdir(reset_style_dir)
    print(len(content_imgs), len(style_imgs))
    if len(content_imgs) <= re_size or len(style_imgs) <= re_size:
        print("your images are less")
        return

    np.random.shuffle(content_imgs)
    np.random.shuffle(style_imgs)

    zips = zip(content_imgs[:re_size], style_imgs[:re_size])

    for c, s in tqdm(zips):
        c_fname = os.path.basename(c)
        s_fname = os.path.basename(s)
        move(os.path.join(content_dir, c_fname), reset_content_dir)
        move(os.path.join(style_dir, s_fname), reset_style_dir)

    move(reset_content_dir, source_dir)
    move(reset_style_dir, source_dir)


def rebuiltDataset(train_n, test_n, source_dir):
    train_dir = os.path.join(source_dir, "train")
    test_dir = os.path.join(source_dir, "test")

    train_content_dir = os.path.join(train_dir, "content")
    train_style_dir = os.path.join(train_dir, "style")
    test_content_dir = os.path.join(test_dir, "content")
    test_style_dir = os.path.join(test_dir, "style")

    if not (os.path.exists(train_dir) and os.path.exists(test_dir)):
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        os.mkdir(train_content_dir)
        os.mkdir(train_style_dir)
        os.mkdir(test_content_dir)
        os.mkdir(test_style_dir)
    # train dir
    imgs = glob.glob(source_dir+"/*.jpg")
    np.random.shuffle(imgs)
    half = train_n // 2
    train_content_imgs = imgs[:half]
    train_style_imgs = imgs[half:]
    for c, s in tqdm(zip(train_content_imgs, train_style_imgs)):
        cn = os.path.basename(c)
        sn = os.path.basename(s)
        move(os.path.join(source_dir, cn), train_content_dir)
        move(os.path.join(source_dir, sn), train_style_dir)
    # test dir
    imgs = glob.glob(source_dir+"/*.jpg")
    np.random.shuffle(imgs)
    half = test_n // 2
    test_content_imgs = imgs[:half]
    test_style_imgs = imgs[half:]
    for c, s in tqdm(zip(test_content_imgs, test_style_imgs)):
        cn = os.path.basename(c)
        sn = os.path.basename(s)
        move(os.path.join(source_dir, cn), test_content_dir)
        move(os.path.join(source_dir, sn), test_style_dir)


class PreprocessDataSet(data.Dataset):
    def __init__(self, content_dir, style_dir, transformer=tran):
        super(PreprocessDataSet, self).__init__()
        content_dir_resized = content_dir + "_resize"
        style_dir_resized = style_dir + "_resize"
        if not (os.path.exists(content_dir_resized) and os.path.exists(style_dir_resized)):
            os.mkdir(content_dir_resized)
            os.mkdir(style_dir_resized)
            self.img_resize(source_dir=content_dir, target_dir=content_dir_resized)
            self.img_resize(source_dir=style_dir, target_dir=style_dir_resized)

        content_images = glob.glob(content_dir_resized + "/*.jpg")  # 返回匹配路径下的所有文件路径列表
        content_images += glob.glob(content_dir_resized + "/*.png")
        np.random.shuffle(content_images)
        style_images = glob.glob(style_dir_resized + "/*.jpg")
        style_images += glob.glob(style_dir_resized + "/*.png")
        np.random.shuffle(style_images)
        self.image_stream = list(zip(content_images, style_images))
        self.transformer = transformer

    @staticmethod
    def img_resize(source_dir, target_dir):
        print(f"start resize images from {source_dir} to {target_dir}")
        source_imgs = os.listdir(source_dir)
        for s in tqdm(source_imgs):    # tqdm是python的进度条库
            imgname = os.path.basename(s)
            try:
                img = cv2.imread(os.path.join(source_dir, s))
                if img.shape[-1] == 3 and len(img.shape) == 3:
                    h, w, c = img.shape
                    r = w / h
                    if h < w:
                        h = 128
                        w = int(h * r)
                    else:
                        w = 128
                        h = int(w // r)
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(target_dir, imgname), img)
            except:
                traceback.print_exc()
                continue

    def __getitem__(self, item):
        content_img, style_img = self.image_stream[item]
        content_img = Image.open(content_img)
        style_img = Image.open(style_img)
        if self.transformer:
            content_img = self.transformer(content_img)
            style_img = self.transformer(style_img)
        return content_img, style_img

    def __len__(self):
        return len(self.image_stream)

