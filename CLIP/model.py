from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from clip.adaptor import Adapter
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        print("resblocks: ", len(self.resblocks))
    def forward(self, x: torch.Tensor, feature_layers=None, visual_prompt=None):
        """
        Args:
            x (torch.Tensor): 输入张量
            feature_layers (list, optional): 需要提取特征的层索引列表，默认为None
            visual_prompt (torch.Tensor, optional):控制是否拼接视觉提示张量，默认为None
        Returns:
            torch.Tensor 或 list: 如果fearure_layers为None，返回最终输出张量；否则返回指定层的特征列表
        """
        out = []
        prefix_len = len(visual_prompt) if visual_prompt is not None else 0
        # 遍历所有残差块
        for i in range(len(self.resblocks)):
            # 如果存在视觉提示，则将其与输入张量拼接
            if i < prefix_len:
                x = torch.cat([visual_prompt[i:i+1].repeat(x.size(0), 1, 1), x], dim=1)
            # 通过第i个残差块
            x = self.resblocks[i](x)
            # 如果之前添加了视觉提示，现在需要移除
            if i < prefix_len:
                x = x[:, visual_prompt[i:i+1].size(1):]
            # 如果指定了特征层，则保存对应层的输出
            if feature_layers is not None and i+1 in feature_layers:
                out.append(x)
        # 根据是否指定特征层返回相应结果
        if feature_layers is None:
            return x
        else:
            return out


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        print(self.positional_embedding.size())
    def forward(self, x: torch.Tensor, feature_layers=[24], visual_prompt=None, class_embeds=None):
        """
        Args:
            x (torch.Tensor): 输入的图像张量，形状为 [batch_size, channels, height, width]
            feature_layers (list, optional): 指定要提取特征的Transformer层索引，默认为[24]
            visual_prompt (torch.Tensor, optional): 可选的视觉提示张量，默认为None
            
        Returns:
            list: 指定层的输出特征列表，每个元素形状为 [batch_size, seq_length, width（embed_dim）]
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # 添加class token
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # 当输入尺寸变化时更新位置编码
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(x.shape[-1], new_side * new_side).transpose(0, 1)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos], 0)
            
        x = x + self.positional_embedding.to(x.dtype)
         
        # 如果提供了视觉提示，则将其添加到输入序列中
        if visual_prompt is not None:
            x = torch.cat([x, visual_prompt[:1].repeat(x.size(0), 1, 1)], dim=1)    #x.shape = [B, N+1, D]
        x = self.ln_pre(x)  #NLD: [B, T, D]
        

        used_adapter = hasattr(self, "adaptor") and (class_embeds is not None)
        if used_adapter:
            out = [ self.adaptor(x, class_embeds, hw=(new_side, new_side)) ]  # 假定返回 NLD
        else:
             x = x.permute(1, 0, 2)  # NLD -> LND
             out = self.transformer(x, feature_layers)
             if feature_layers is None:
                out = [out]
      
        for i, o in enumerate(out):
            if o.dim() == 3 and not used_adapter:   
                out[i] = o.permute(1, 0, 2)
            else:
                out[i] = o
            if visual_prompt is not None:
                out[i] = out[i][:, :-visual_prompt.size(1), :]
        return out


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

        # few-shot memory bank init
        self.memorybank = None


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def insert(self, args, tokenizer, device):
        # 初始化模型中的状态提示相关参数和组件
        self.normal_cls_prompt = f'without defect.'
        self.anomaly_cls_prompt = f'with defect.'
        self.state_prompt_tokens = tokenizer([self.normal_cls_prompt, self.anomaly_cls_prompt]).to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_len = args.prompt_len
        # 初始化状态提示嵌入向量参数
        self.state_prompt_embedding = nn.Parameter(torch.empty(1, args.prompt_len, self.token_embedding.weight.shape[-1]).to(device))
        nn.init.normal_(self.state_prompt_embedding, std=0.01)
        self.state_prompt_embedding.requires_grad_(True)
        self.tokenizer = tokenizer
        

        if isinstance(self.visual, VisionTransformer):
            vit_blocks = nn.ModuleList(list(self.visual.transformer.resblocks))
            self.adaptor = Adapter(
                blocks=vit_blocks,
                d=self.visual.transformer.width,
                num_classes=2,
            ).to(device)
            # 挂到视觉模型上，供 VisionTransformer.forward 调用
            self.visual.adaptor = self.adaptor
            # 文本→视觉维度映射（text_width -> vision_width）
            self.txt2vis = nn.Linear(self.transformer.width, self.visual.transformer.width, bias=False).to(device)

    def encode_state_prompt(self):
        """
        编码状态提示文本，生成对应的状态提示嵌入向量
        Returns:
            torch.Tensor: 编码后的状态提示特征向量
        """
        # 获取状态提示tokens的嵌入表示
        state_x = self.token_embedding(self.state_prompt_tokens).type(self.dtype)
        # 将可学习的状态提示嵌入与token嵌入拼接，并截取前77个token
        state_x = torch.cat([self.state_prompt_embedding.repeat(2, 1, 1), state_x], dim=1)[:, :77, :]
        # 添加位置编码
        state_x = state_x + self.positional_embedding.type(self.dtype)
        state_x = state_x.permute(1, 0, 2)  # NLD -> LND
        # 通过Transformer处理
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)  # LND -> NLD
        # 最终归一化处理
        state_x = self.ln_final(state_x).type(self.dtype)
        # 提取特定位置的特征并投影到文本特征空间
        state_x = state_x[torch.arange(state_x.shape[0]), self.prompt_len + self.state_prompt_tokens.argmax(dim=-1)] @ self.text_projection
        return state_x
    
    def encode_state_prompt_for_adaptor(self):

        state_x = self.token_embedding(self.state_prompt_tokens).type(self.dtype)
        state_x = torch.cat([self.state_prompt_embedding.repeat(2, 1, 1), state_x], dim=1)[:, :77, :]
        state_x = state_x + self.positional_embedding.type(self.dtype)
        state_x = state_x.permute(1, 0, 2)  # NLD -> LND
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)  # LND -> NLD
        state_x = self.ln_final(state_x).type(self.dtype)
        token_feat = state_x[torch.arange(state_x.shape[0]), self.prompt_len + self.state_prompt_tokens.argmax(dim=-1)]
        if hasattr(self, "txt2vis"):
            token_feat = self.txt2vis(token_feat)
        return token_feat  # [2, vision_width]

    def get_trainable_parameters(self):
        params = [self.state_prompt_embedding]
        if hasattr(self, "adaptor"):
            params += list(self.adaptor.parameters())
        if hasattr(self, "txt2vis"):
            params += list(self.txt2vis.parameters())
        return params
    
    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_image(self, image, feature_layers=None):
        return self.visual(image.type(self.dtype), feature_layers)

    def detect_encode_image(self, image, args):
        class_embeds = self.encode_state_prompt_for_adaptor()
        tokens = self.visual(image.type(self.dtype),
                             feature_layers=None,
                             visual_prompt=None,
                             class_embeds=class_embeds)[0]
        # 3) 正常的 ln_post 与 proj
        tokens = self.visual.ln_post(tokens) @ self.visual.proj
        return [tokens]
    
    def store_memory(self, image, args):
        memory_layers = getattr(args, 'memory_layers', None) or [24]
        img_tokens = self.encode_image(image, memory_layers)
        b, l, c = img_tokens[0].size()
        self.memorybank = [torch.nn.functional.normalize(img_token[:, 1:], dim=-1).reshape(-1, c) for img_token in img_tokens]
    # @torch.no_grad()
    # def store_memory(self, image, args):
    #     memory_layers = getattr(args, 'memory_layers', None) or [getattr(self.visual.transformer, "layers", 24)]
    #     img_tokens = self.encode_image(image, memory_layers)   # list of [B, 1+N, C]
    #     banks = []
    #     for t in img_tokens:
    #         if t.ndim != 3 or t.size(1) <= 1:
    #             banks.append(torch.empty(0, t.size(-1), device=t.device, dtype=t.dtype))
    #             continue
    #         B, T, C = t.size()
    #         tokens = torch.nn.functional.normalize(t[:, 1:], dim=-1).reshape(-1, C)  # 去 CLS
    #         banks.append(tokens)
    #     self.memorybank = banks
    #     # 方便一次性确认是否真的写入了支持向量
    #     print([b.shape for b in self.memorybank])  # 形如 [(B*(H*W), C), ...]

    def detect_forward_seg(self, image, args):
        text_features = self.encode_state_prompt()
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        img_tokens = self.detect_encode_image(image, args)
        scores = 0
        for img_token in img_tokens:
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = torch.matmul(img_token, text_features.permute(1, 0)) / 0.07
            scores += score
        # prob = torch.softmax(scores, dim=-1) 
        prob = torch.sigmoid(scores) 
        cls_label = prob[:, 0, 1].view(-1)  #[B]   CLS token 的异常概率
        logits_img = scores[:, 0, :]  # [B, 2] ——CLS token 的原始 logits   
       
        patch_logits = scores[:, 1:, 1] - scores[:, 1:, 0]   # [B, N]
        predict_mask = torch.sigmoid(patch_logits)           # [B, N]
        B = predict_mask.shape[0]
        _, T, _ = img_tokens[0].shape   # img_tokens 由 detect_encode_image 返回
        N = max(T - 1, 1)               # 去掉 CLS
        H = W = int(N ** 0.5)           # 对 518 输入应为 37
        predict_mask = predict_mask.view(B, 1, H, W)
        return cls_label, predict_mask, img_tokens, logits_img
    
    def detect_forward_memorybank(self, image, args):
        scores = 0
        memory_layers = getattr(args, 'memory_layers', None) or [24]
        img_tokens = self.encode_image(image, memory_layers)
        for i, img_token in enumerate(img_tokens):
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = (1 - torch.matmul(img_token, self.memorybank[i].T)).min(dim=-1)[0] / 2
            scores += score[:, 1:]
        scores = scores / len(img_tokens)
        cls_label = torch.max(scores, dim=-1)[0]
        b, l = scores.size()
        h = w = int(math.sqrt(l))
        predict_mask = scores.reshape(b, 1, h, w)
        return cls_label, predict_mask

    # def detect_forward_memorybank(self, image, args):
    #     scores = 0
    #     valid = False
    #     memory_layers = getattr(args, 'memory_layers', None) or [24]
    #     img_tokens = self.encode_image(image, memory_layers)   # list of [B, 1+N, C]

    #     for i, img_token in enumerate(img_tokens):
    #         mem = None if (self.memorybank is None or i >= len(self.memorybank)) else self.memorybank[i]
    #         if mem is None or mem.ndim != 2 or mem.shape[0] == 0:
    #             continue  # 跳过空库层
    #         img_token = torch.nn.functional.normalize(img_token, dim=-1)
    #         score = (1 - torch.matmul(img_token, mem.T)).min(dim=-1)[0] / 2  # [B, 1+N]
    #         scores = scores + score[:, 1:] if valid else score[:, 1:]        # 累加去掉 CLS 的 patch 分数
    #         valid = True

    #     if not valid:
    #         # 所有层都没有有效支持向量——兜底为 0 图（等价于“不开 few-shot”）
    #         B, L, _ = img_tokens[0].size()
    #         L = max(L - 1, 1)
    #         H = W = int(L ** 0.5)
    #         predict_map = torch.zeros(B, 1, H, W, device=img_tokens[0].device, dtype=img_tokens[0].dtype)
    #         cls_label   = torch.zeros(B, device=img_tokens[0].device, dtype=img_tokens[0].dtype)
    #         return cls_label, predict_map

    #     scores = scores / len(img_tokens)          # [B, N]
    #     cls_label = torch.max(scores, dim=-1)[0]
    #     B, T, C = img_tokens[0].shape             # [B, 1+N, C]  —— NLD 已保证
    #     N = max(T - 1, 1)
    #     H = W = int(N ** 0.5)                     # 518 输入 → 37×37
    #     predict_map = scores.view(B, 1, H, W)
    #     return cls_label, predict_map

    
    def detect_forward(self, image, args):
        cls_label, predict_mask, _, _= self.detect_forward_seg(image, args)
        if self.memorybank is not None:
            cls_label_memory, predict_map_memory = self.detect_forward_memorybank(image, args)
            predict_mask = predict_map_memory + args.alpha * predict_mask
            cls_label = cls_label_memory + args.alpha * cls_label
        return cls_label, predict_mask

    # def detect_forward(self, image, args):
    #     cls_label, predict_mask, _, _ = self.detect_forward_seg(image, args)

    #     mb = getattr(self, "memorybank", None)
    #     has_mem = (
    #         isinstance(mb, list) and len(mb) > 0
    #         and all((m is not None) and (m.ndim == 2) and (m.shape[0] > 0) for m in mb)  # 至少有一条支持向量
    #     )
    #     if has_mem:
    #         alpha = getattr(args, "alpha", 1.0)
    #         cls_label_mb, predict_map_mb = self.detect_forward_memorybank(image, args)
    #         predict_mask = predict_map_mb + alpha * predict_mask
    #         cls_label    = cls_label_mb + alpha * cls_label
    #     return cls_label, predict_mask

    
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()