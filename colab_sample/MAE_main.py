"""
MAE 简单复现  ver: 2021.3.23
本实验是单个图片的MAE复现，意在说明MAE的过程。
本身没有进行代码结构性与逻辑细节的优化。
这个里面用的vit和我们习惯的timm版本略有不同，后续需要做匹配
总的来说，这个脚本作为一个流程例子用来学习MAE流程和部分细节还是不错的，
但是需要匹配性的重新做才可以使用到论文中。
这个脚本离真正论文能用的还有很远距离。
来自：https://zhuanlan.zhihu.com/p/439554945
我加了一些内容和注释
"""
import os

import matplotlib.pyplot as plt
import timm
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter

from Transformers import *


class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, mask_ratio=0.75, decoder_depth=1,
                 num_decoder_heads=8, decoder_dim_per_head=64):
        """
        MAE 框架，自带decoder模块
        :param encoder: 输入Transformer模型
        :param decoder_dim: Encoder 输出的维度可能和 Decoder 要求的输入维度不一致
        :param mask_ratio: paper提倡这个比例最好是 75%
        :param decoder_depth: 实质就是多层堆叠的 Transformer（这个也有一个gap问题，前后结构是否一样值得研究） TODO
        :param num_decoder_heads:
        :param decoder_dim_per_head:
        """
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        # Encoder(这里 CW 用 ViT 实现)
        self.encoder = encoder
        self.patch_h = encoder.patch_embed.patch_size[0]
        self.patch_w = encoder.patch_embed.patch_size[1]

        encoder_dim = encoder.embed_dim

        channels = 3
        patch_dim = channels * self.patch_h * self.patch_w

        # 由于原生的 ViT 有 cls_token，因此其 position embedding 的倒数第2个维度是：
        # 实际划分的 patch 数量加上 1个 cls_token
        # num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]  fixme 之前不匹配
        num_patches_plus_cls_token = encoder.patch_embed.num_patches + 1

        # Patch embedding 后续考虑做成使用encoder的，但是原文使用的是MLP映射而不是CNN映射
        self.patch_embed = nn.Linear(patch_dim, encoder_dim)

        # Input channels of encoder patch embedding: patch size**2 x 3
        # 这个用作预测头部的输出通道，从而能够对 patch 中的所有像素值进行预测
        num_pixels_per_patch = encoder_dim  # encoder.patch_embed.weight.size(1)  fixme 之前不匹配
        # print(num_pixels_per_patch)

        # Encoder-Decoder：Encoder 输出的维度可能和 Decoder 要求的输入维度不一致，因此需要转换
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # Mask token
        # paper提倡这个比例最好是 75%
        self.mask_ratio = mask_ratio
        # mask token 的实质：1个可学习的共享向量
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))

        # Decoder：实质就是多层堆叠的 Transformer
        self.decoder = Transformer(
            decoder_dim,
            decoder_dim * 4,
            depth=decoder_depth,
            num_heads=num_decoder_heads,
            dim_per_head=decoder_dim_per_head,
        )
        # 在 Decoder 中用作对 mask tokens 的 position embedding
        # Filter out cls_token 注意第1个维度去掉 cls_token
        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls_token - 1, decoder_dim)

        # Prediction head 输出的维度数等于1个 patch 的像素值数量
        self.head = nn.Linear(decoder_dim, num_pixels_per_patch)

    def forward(self, x):
        device = x.device
        b, c, h, w = x.shape

        '''i. Patch partition'''
        # 首先需要将图像划分成 patch，划分方式实质就是维度的变换
        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

        '''ii. Divide into masked & un-masked groups'''

        # 根据 mask 比例计算需要 mask 掉的 patch 数量
        # num_patches = (h // self.patch_h) * (w // self.patch_w)
        num_masked = int(self.mask_ratio * num_patches)

        # Shuffle:生成对应 patch 的随机索引
        # torch.rand() 服从均匀分布(normal distribution)
        # torch.rand() 只是生成随机数，argsort() 是为了获得成索引
        # (b, n_patches)
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        # mask 和 unmasked patches 对应的索引
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

        # 对应 batch 维度的索引：(b,1)
        batch_ind = torch.arange(b, device=device).unsqueeze(-1)
        # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组 TODO
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

        '''iii. Encode'''

        # 将 patches 通过 emebdding 转换成 tokens
        unmask_tokens = self.patch_embed(unmask_patches)
        # unmask_tokens = self.encoder.patch_embed(unmask_patches)  fixme 之前不匹配

        # 为 tokens 加入 position embeddings
        # 注意这里索引加1是因为索引0对应 ViT 的 cls_token
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        # 真正的编码过程
        encoded_tokens = self.encoder.blocks(unmask_tokens)
        # encoded_tokens = self.encoder.transformer(unmask_tokens)  fixme 之前不匹配

        '''iv. Decode'''

        # 对编码后的 tokens 维度进行转换，从而符合 Decoder 要求的输入维度
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # 由于 mask token 实质上只有1个，因此要对其进行扩展，从而和 masked patches 一一对应
        # (decoder_dim)->(b, n_masked, decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        # 为 mask tokens 加入位置信息
        mask_tokens += self.decoder_pos_embed(mask_ind)

        # 将 mask tokens 与 编码后的 tokens 拼接起来
        # (b, n_patches, decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # Un-shuffle：恢复原先 patches 的次序
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
        # 将全量 tokens 喂给 Decoder 解码
        decoded_tokens = self.decoder(dec_input_tokens)

        '''v. Mask pixel Prediction'''

        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
        # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens)

        return pred_mask_pixel_values, mask_patches

    # @torch.no_grad
    def predict(self, x):
        self.eval()

        device = x.device
        b, c, h, w = x.shape

        '''i. Patch partition'''

        num_patches = (h // self.patch_h) * (w // self.patch_w)
        # (b, c=3, h, w)->(b, n_patches, patch_size**2*c)
        patches = x.view(
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

        '''ii. Divide into masked & un-masked groups'''

        num_masked = int(self.mask_ratio * num_patches)

        # Shuffle
        # (b, n_patches)
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()
        # .argsort()返回能使得数据有序的索引。比如argsort([2,1])，结果是[1,0],即：最小的元素是索引为0的元素，其次是1.
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

        # (b, 1)
        batch_ind = torch.arange(b, device=device).unsqueeze(-1)
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]

        '''iii. Encode'''

        # unmask_tokens = self.encoder.patch_embed(unmask_patches)  fixme 之前不匹配
        unmask_tokens = self.patch_embed(unmask_patches)

        # Add position embeddings
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        # encoded_tokens = self.encoder.transformer(unmask_tokens)  fixme 之前不匹配
        encoded_tokens = self.encoder.blocks(unmask_tokens)
        '''iv. Decode'''

        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)

        # (decoder_dim)->(b, n_masked, decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)
        # Add position embeddings
        mask_tokens += self.decoder_pos_embed(mask_ind)

        # (b, n_patches, decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)
        # dec_input_tokens = concat_tokens
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)
        # Un-shuffle
        dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
        decoded_tokens = self.decoder(dec_input_tokens)

        '''v. Mask pixel Prediction'''

        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]
        # (b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens)

        # 比较下预测值和真实值
        mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        mse_all_patches = mse_per_patch.mean()

        print(f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} '
              f'total {num_masked} masked patches')
        print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')

        '''vi. Reconstruction'''

        recons_patches = patches.detach()
        # Un-shuffle (b, n_patches, patch_size**2 * c)
        recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values
        # 模型重建的效果图
        # Reshape back to image
        # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)
        # mask 效果图
        patches[batch_ind, mask_ind] = mask_patches
        patches_to_img = patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return recons_img, patches_to_img


def train(mae, dataloader, optimizer, criterion, epoch=1000, writer=None):
    for i in range(epoch):
        img_ts = dataloader  # TODO 这个只是假的dataloader
        pred_mask_pixel_values, mask_patches = mae(img_ts)
        optimizer.zero_grad()
        loss = criterion(pred_mask_pixel_values, mask_patches)

        loss.backward()
        optimizer.step()
        # 比较下预测值和真实值
        mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        mse_all_patches = mse_per_patch.mean()
        print(f'Epoch: {i + 1} ')
        print(f'mse all (masked)patches: {mse_all_patches} ')
        print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')

        if writer is not None:
            # ...log the running loss
            writer.add_scalar('train' + ' minibatch loss',
                              float(mse_all_patches),
                              i)
    # 结束记录内容给tensorboard
    if writer is not None:
        writer.close()


def main():
    BASE_DIR = r'./'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    draw_path = os.path.join(BASE_DIR, 'runs')
    if not os.path.exists(draw_path):
        os.makedirs(draw_path)

    writer = SummaryWriter(draw_path)

    # 读入图像并缩放到适合模型输入的尺寸
    img_raw = Image.open(os.path.join(BASE_DIR, 'BUAA.jpg'))
    h, w = img_raw.height, img_raw.width
    ratio = h / w
    print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

    img_size, patch_size = (224, 224), (16, 16)
    img = img_raw.resize(img_size)
    rh, rw = img.height, img.width
    print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')
    img.save(os.path.join(BASE_DIR, 'resized_BUAA.jpg'))

    # 将图像转换成张量
    from torchvision.transforms import ToTensor, ToPILImage

    img_ts = ToTensor()(img).unsqueeze(0).to(device)
    print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

    # 实例化模型
    # encoder = ViT(img_size, patch_size, dim=512, mlp_dim=1024, dim_per_head=64)
    encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
    decoder_dim = 512
    mae = MAE(encoder, decoder_dim, decoder_depth=6)
    # weight = torch.load(os.path.join(BASE_DIR, 'mae.pth'), map_location='cpu')  # 加载训练好的权重
    mae.to(device)

    # 训练
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mae.parameters(), lr=0.0001, weight_decay=0.01)

    train(mae, img_ts, optimizer, criterion, epoch=1000, writer=writer)  # TODO 这个用的假dataloader

    # 推理
    # 模型重建的效果图，mask 效果图
    recons_img_ts, masked_img_ts = mae.predict(img_ts)
    recons_img_ts, masked_img_ts = recons_img_ts.cpu().squeeze(0), masked_img_ts.cpu().squeeze(0)

    # 将结果保存下来以便和原图比较
    recons_img = ToPILImage()(recons_img_ts)
    recons_img.save(os.path.join(BASE_DIR, 'recons_BUAA.jpg'))

    masked_img = ToPILImage()(masked_img_ts)
    masked_img.save(os.path.join(BASE_DIR, 'masked_BUAA.jpg'))

    # 画图
    img = Image.open(os.path.join(BASE_DIR, 'BUAA.jpg'))
    plt.figure("BUAA")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('BUAA')  # 图像题目
    plt.show()

    img = Image.open(os.path.join(BASE_DIR, 'masked_BUAA.jpg'))
    plt.figure("masked_BUAA")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('masked_BUAA')  # 图像题目
    plt.show()

    img = Image.open(os.path.join(BASE_DIR, 'recons_BUAA.jpg'))
    plt.figure("recons_BUAA")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('recons_BUAA')  # 图像题目
    plt.show()


if __name__ == '__main__':
    main()
