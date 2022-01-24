# 分patch的例子
import os

import torch
from torchvision import transforms

train_images_path = "/data/Pathology_Experiment/MIL_dataset/MARS_MIL_split/train/data"
train_labels_path = "/data/Pathology_Experiment/MIL_dataset/MARS_MIL_split/train/mask"


# 给data与mask进行同步变化
class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class to_patch:
    """
    将图片数据分为patch，集成给Pytorch Transform
    """

    def __init__(self, patch_size=(16, 16)):
        self.patch_h = patch_size[0]
        self.patch_w = patch_size[1]

    def __call__(self, x):
        c, h, w = x.shape

        num_patches = (h // self.patch_h) * (w // self.patch_w)

        # patch encoding
        # (c, h, w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, self.patch_h, self.patch_w, c)
        # -> (n_patches, patch_size^2*c)
        patches = x.view(
            c,
            h // self.patch_h,
            self.patch_h,
            w // self.patch_w,
            self.patch_w).permute(1, 3, 2, 4, 0).reshape(num_patches, -1)  # 这个可以直接给transformer编码
        '''
        # compose the patch encoding directly for Transformer
        recons_img = patches.view(h // self.patch_h, w // self.patch_w,
                                  self.patch_h, self.patch_w, c
                                  ).permute(4, 0, 2, 1, 3).reshape(c, h, w)

        '''

        # patch split
        # (n_patches, patch_size^2*c)
        # -> (num_patches, self.patch_h, self.patch_w, c)
        # -> (num_patches, c, self.patch_h, self.patch_w)
        patches = patches.view(num_patches,
                               self.patch_h,
                               self.patch_w,
                               c).permute(0, 3, 1, 2)
        '''
        # patch compose to image
        # (num_patches, c, self.patch_h, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, c, self.patch_h, self.patch_w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (c, h, w)
        patches = patches.view(h // self.patch_h,
                               w // self.patch_w,
                               c,
                               self.patch_h,
                               self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        '''
        # patch compose to image
        # (num_patches, c, self.patch_h, self.patch_w)
        # -> (h // self.patch_h, w // self.patch_w, c, self.patch_h, self.patch_w)
        # -> (c, h // self.patch_h, self.patch_h, w // self.patch_w, self.patch_w)
        # -> (c, h, w)
        composed_patches = patches.view(h // self.patch_h,
                                        w // self.patch_w,
                                        c,
                                        self.patch_h,
                                        self.patch_w).permute(2, 0, 3, 1, 4).reshape(c, h, w)



        return patches,composed_patches


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读入图像并缩放到适合模型输入的尺寸
    img_raw = Image.open(os.path.join('./ROSE_play', '165.jpg'))
    h, w = img_raw.height, img_raw.width
    ratio = h / w
    print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

    img_size, patch_size = (224, 224), (112, 112)
    img = img_raw.resize(img_size)
    rh, rw = img.height, img.width
    print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')
    img.save(os.path.join('./ROSE_play', 'resized_target.jpg'))

    # 将图像转换成张量
    from torchvision.transforms import ToTensor, ToPILImage

    img_ts = ToTensor()(img).to(device)
    print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

    trs = transforms.Lambda(to_patch(patch_size))
    patches,composed_patches = trs(img_ts)

    # check patches
    for i in range(len(patches)):
        recons_img = ToPILImage()(patches[i])
        recons_img.save(os.path.join('./patch_play', 'target_patch' + str(i) + '.jpg'))

    composed_img = ToPILImage()(composed_patches)
    composed_img.save(os.path.join('./patch_play', 'recons_target.jpg'))
