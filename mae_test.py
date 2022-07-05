import os
from PIL import Image

import torch
import torch.nn as nn

from torchvision.transforms import ToTensor, ToPILImage


def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads  # 8
        self.scale = dim_per_head ** -0.5  # 64 ** -0.5 = 0.125

        inner_dim = dim_per_head * num_heads  # 64 * 8 = 512
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 512-->1536

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)  # True
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 512-->512

    def forward(self, x):
        b, l, d = x.shape  # 1, 49, 512

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)  # [1, 49, 512]-->[1, 49, 1536]
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()  # [1, 49, 1536]-->[3, 1, 8, 49, 64]
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)  # [3, 1, 8, 49, 64]-->[1, 1, 8, 49, 64], [1, 1, 8, 49, 64], [1, 1, 8, 49, 64]
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)  # [1, 1, 8, 49, 64]-->[1, 8, 49, 64]

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )  # [1, 8, 49, 64]*[1, 8, 64, 49]-->[1, 8, 49, 49]

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)  # [1, 8, 49, 49]*[1, 8, 49, 64]-->[1, 8, 49, 64]
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)  # [1, 8, 49, 64]-->[1, 49, 512]
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)  # [1, 49, 512]-->[1, 49, 512]
        # assert out.size(-1) == d

        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 512-->1024
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),  # 1024-->512
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=6, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ffn(x)

        return x


class ViT(nn.Module):
    def __init__(
            self, image_size, patch_size,
            num_classes=1000, dim=1024, depth=6, num_heads=8, mlp_dim=2048,
            pool='cls', channels=3, dim_per_head=64, dropout=0., embed_dropout=0.
    ):
        super().__init__()

        img_h, img_w = to_pair(image_size)  # 224,224
        self.patch_h, self.patch_w = to_pair(patch_size)  # 16,16
        assert not img_h % self.patch_h and not img_w % self.patch_w, \
            f'Image dimensions ({img_h},{img_w}) must be divisible by the patch size ({self.patch_h},{self.patch_w}).'
        num_patches = (img_h // self.patch_h) * (img_w // self.patch_w)

        assert pool in {'cls', 'mean'}, f'pool type must be either cls (cls token) or mean (mean pooling), got: {pool}'  # 'cls'

        patch_dim = channels * self.patch_h * self.patch_w  # 3*16*16=768
        self.patch_embed = nn.Linear(patch_dim, dim)  # 768-->512

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # shape:[1, 1, 512]
        # Add 1 for cls_token
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # shape:[1, 197, 512]

        self.dropout = nn.Dropout(p=embed_dropout)
        self.transformer = Transformer(
            dim, mlp_dim, depth=depth, num_heads=num_heads,
            dim_per_head=dim_per_head, dropout=dropout
        )  # 512, 1024, 6, 8, 64, 0.0

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # def forward(self, x):
    #     b, c, img_h, img_w = x.shape
    #     assert not img_h % self.patch_h and not img_w % self.patch_w, \
    #         f'Input image dimensions ({img_h},{img_w}) must be divisible by the patch size ({self.patch_h},{self.patch_w}).'
    #
    #     '''i. Patch partition'''
    #     num_patches = (img_h // self.patch_h) * (img_w // self.patch_w)
    #     # (b,c,h,w)->(b,n_patches,patch_h*patch_w*c)
    #     patches = x.view(
    #         b, c,
    #         img_h // self.patch_h, self.patch_h,
    #         img_w // self.patch_w, self.patch_w
    #     ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)
    #
    #     '''ii. Patch embedding'''
    #     # (b,n_patches,dim)
    #     tokens = self.patch_embed(patches)
    #     # (b,n_patches+1,dim)
    #     tokens = torch.cat([self.cls_token.repeat(b, 1, 1), tokens], dim=1)
    #     tokens += self.pos_embed[:, :(num_patches + 1)]
    #     tokens = self.dropout(tokens)
    #
    #     '''iii. Transformer Encoding'''
    #     enc_tokens = self.transformer(tokens)
    #
    #     '''iv. Pooling'''
    #     # (b,dim)
    #     pooled = enc_tokens[:, 0] if self.pool == 'cls' else enc_tokens.mean(dim=1)
    #
    #     '''v. Classification'''
    #     # (b,n_classes)
    #     logits = self.mlp_head(pooled)
    #
    #     return logits


class MAE(nn.Module):
    def __init__(
            self, encoder, decoder_dim,
            mask_ratio=0.75, decoder_depth=1,
            num_decoder_heads=8, decoder_dim_per_head=64
    ):
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        # Encoder(这里 CW 用 ViT 实现)
        self.encoder = encoder
        self.patch_h, self.patch_w = encoder.patch_h, encoder.patch_w  # 16,16

        # 由于原生的 ViT 有 cls_token，因此其 position embedding 的倒数第2个维度是：
        # self.pos_embed==nn.Parameter(torch.randn(1, num_patches + 1, dim)),实际划分的 patch 数量加上 1个 cls_token
        num_patches_plus_cls_token, encoder_dim = encoder.pos_embed.shape[-2:]  # 197, 512
        # Input channels of encoder patch embedding: patch size**2 x 3
        # 这个用作预测头部的输出通道，从而能够对 patch 中的所有像素值进行预测
        num_pixels_per_patch = encoder.patch_embed.weight.size(1)  # 768

        # Encoder-Decoder：Encoder 输出的维度可能和 Decoder 要求的输入维度不一致，因此需要转换
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # Mask token
        # 社会提倡这个比例最好是 75%
        self.mask_ratio = mask_ratio
        # mask token 的实质：1个可学习的共享向量
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))  # shape:[512]

        # Decoder：实质就是多层堆叠的 Transformer
        self.decoder = Transformer(
            decoder_dim,  # 512
            decoder_dim * 4,  # 2048
            depth=decoder_depth,  # 6
            num_heads=num_decoder_heads,  # 8
            dim_per_head=decoder_dim_per_head,  # 64
        )
        # 在 Decoder 中用作对 mask tokens 的 position embedding
        # Filter out cls_token 注意第1个维度去掉 cls_token
        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls_token - 1, decoder_dim)  # shape:[196,512]

        # Prediction head 输出的维度数等于1个 patch 的像素值数量
        self.head = nn.Linear(decoder_dim, num_pixels_per_patch)  # [512]-->[768]

    @torch.no_grad()
    def predict(self, x):
        self.eval()

        device = x.device
        b, c, h, w = x.shape  # b, c, h, w == 1, 3, 224, 224

        '''i. Patch partition'''
        # 将图像划分成 patch，划分方式实质就是维度的变换
        num_patches = (h // self.patch_h) * (w // self.patch_w)  # num_patches==196
        # 将图像划分成 patches：(b, c=3, h, w)->(b, n_patches, patch_size**2*c)
        patches = x.view(  # view等价reshape，permute转置纬度
            b, c,
            h // self.patch_h, self.patch_h,
            w // self.patch_w, self.patch_w
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)  # shape:[1, 196, 768]

        '''ii. Divide into masked & un-masked groups'''

        num_masked = int(self.mask_ratio * num_patches)  # num_masked==147

        # Shuffle
        # (b, n_patches)，torch.rand()服从均匀分布(normal distribution)生成随机数，argsort()是为了获得成索引
        shuffle_indices = torch.rand(b, num_patches, device=device).argsort()  # shape:[1, 196]
        mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]  # shape:[1, 147],[1,49]

        # (b, 1)
        batch_ind = torch.arange(b, device=device).unsqueeze(-1)  # batch_ind==tensor([[0],])
        mask_patches, unmask_patches = patches[batch_ind, mask_ind], patches[batch_ind, unmask_ind]  # shape:[1, 147, 768], [1, 49, 768]

        '''iii. Encode, 当然，我们得先对 unmasked patches 进行 emebdding 转换成 tokens，
        并且加上 position embeddings，从而为它们添加位置信息，然后才能是真正的编码过程。
        至于编码过程，实质上就是扔给 Transformer 玩(query 和 key 玩一玩，玩出个 attention 后再和 value 一起玩~)：'''
        # 将 patches 通过 emebdding 转换成 tokens
        unmask_tokens = self.encoder.patch_embed(unmask_patches)  # self.encoder==VIT, shape:[1, 49, 768]-->[1, 49, 512]
        # 为 tokens 加入 position embeddings, 注意这里索引加1是因为索引0对应 ViT 的 cls_token
        # self.encoder.pos_embed==nn.Parameter(torch.randn(1, num_patches + 1, dim)) shape:[1, 197, 512], unmask_tokens shape:[1, 49, 512]
        unmask_tokens += self.encoder.pos_embed.repeat(b, 1, 1)[batch_ind, unmask_ind + 1]
        encoded_tokens = self.encoder.transformer(unmask_tokens)  # shape:[1, 49, 512]

        '''iv. Decode, 解码后取出mask tokens对应的部分送入到全连接层，对masked patches的像素值进行预测，
        最后将预测结果与masked patches进行比较，计算MSE loss'''
        # 对编码后的 tokens 维度进行转换，从而符合 Decoder 要求的输入维度
        # self.enc_to_dec==nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        enc_to_dec_tokens = self.enc_to_dec(encoded_tokens)  # shape:[1, 49, 512]

        # 由于mask token实质上只有1个纬度，因此要对其进行扩展，从而和masked patches一一对应,(decoder_dim)->(b, n_masked, decoder_dim)
        # mask token 的实质：1个可学习的共享向量，self.mask_embed==nn.Parameter(torch.randn(decoder_dim))
        mask_tokens = self.mask_embed[None, None, :].repeat(b, num_masked, 1)  # [None, None, :]在前面添加两个纬度,[512]-->[1, 147, 512]
        # Add position embeddings， self.decoder_pos_embed==Embedding(196, 512)
        mask_tokens += self.decoder_pos_embed(mask_ind)  # shape:[1, 147]-->[1, 147, 512]

        # 将mask tokens与编码后的tokens拼接起来, (b, n_patches, decoder_dim)
        concat_tokens = torch.cat([mask_tokens, enc_to_dec_tokens], dim=1)  # shape:[1, 147, 512]+[1, 49, 512]-->[1, 196, 512]
        # dec_input_tokens = concat_tokens
        dec_input_tokens = torch.empty_like(concat_tokens, device=device)  # empty_like定义一个随机初始化的矩阵, shape:[1, 196, 512]
        # Un-shuffle，恢复原先patches的次序
        dec_input_tokens[batch_ind, shuffle_indices] = concat_tokens
        decoded_tokens = self.decoder(dec_input_tokens)  # shape:[1, 196, 512]-->[1, 196, 512]

        '''v. Mask pixel Prediction'''
        # 取出解码后的 mask tokens
        dec_mask_tokens = decoded_tokens[batch_ind, mask_ind, :]  # shape:[1, 147, 512]
        # 预测masked patches的像素值，(b, n_masked, n_pixels_per_patch=patch_size**2 x c)
        pred_mask_pixel_values = self.head(dec_mask_tokens)  # shape:[1, 147, 512]-->[1, 147, 768]

        # 比较下预测值和真实值
        mse_per_patch = (pred_mask_pixel_values - mask_patches).abs().mean(dim=-1)
        mse_all_patches = mse_per_patch.mean()  # 均方差损失

        print(
            f'mse per (masked)patch: {mse_per_patch} mse all (masked)patches: {mse_all_patches} total {num_masked} masked patches')
        print(f'all close: {torch.allclose(pred_mask_pixel_values, mask_patches, rtol=1e-1, atol=1e-1)}')

        '''vi. Reconstruction'''

        recons_patches = patches.detach()  # shape:[1, 196, 768]
        # Un-shuffle (b, n_patches, patch_size**2 * c)
        recons_patches[batch_ind, mask_ind] = pred_mask_pixel_values  # shape:[1, 196, 768]
        # 模型重建的效果图
        # Reshape back to image
        # (b, n_patches, patch_size**2 * c)->(b, c, h, w)
        recons_img = recons_patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)  # shape:[1, 3, 224, 224]

        mask_patches = torch.randn_like(mask_patches, device=mask_patches.device)  # randn_like均值为0，方差为1的正态分布, shape:[1, 147, 768]
        # mask 效果图
        patches[batch_ind, mask_ind] = mask_patches  # shape:[1, 196, 768]
        patches_to_img = patches.view(
            b, h // self.patch_h, w // self.patch_w,
            self.patch_h, self.patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)  # shape:[1, 3, 224, 224]

        return recons_img, patches_to_img

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读入图像并缩放到适合模型输入的尺寸
    img_raw = Image.open(os.path.join('test.jpeg'))
    h, w = img_raw.height, img_raw.width
    ratio = h / w
    print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

    img_size, patch_size = (224, 224), (16, 16)
    img = img_raw.resize(img_size)
    rh, rw = img.height, img.width
    print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')
    img.save(os.path.join('resized_test.jpg'))

    img_ts = ToTensor()(img).unsqueeze(0).to(device)
    print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

    # 实例化模型并加载训练好的权重
    encoder = ViT(img_size, patch_size, dim=512, mlp_dim=1024, dim_per_head=64)
    decoder_dim = 512
    mae = MAE(encoder, decoder_dim, decoder_depth=6)
    # weight = torch.load(os.path.join('mae.pth'), map_location='cpu')
    mae.to(device)

    # 推理
    # 模型重建的效果图，mask 效果图
    recons_img_ts, masked_img_ts = mae.predict(img_ts)
    recons_img_ts, masked_img_ts = recons_img_ts.cpu().squeeze(0), masked_img_ts.cpu().squeeze(0)

    # 将结果保存下来以便和原图比较
    recons_img = ToPILImage()(recons_img_ts)
    recons_img.save(os.path.join('recons_test.jpg'))

    masked_img = ToPILImage()(masked_img_ts)
    masked_img.save(os.path.join('masked_test.jpg'))