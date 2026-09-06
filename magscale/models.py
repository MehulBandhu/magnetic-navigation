import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d, heads, mup=False):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.heads = heads
        dh = d // heads
        self.scale = 1.0 / dh if mup else dh ** -0.5   # muP: 1/d not 1/sqrt(d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, heads, mup=False):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads, mup)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    """MAE-style encoder on all tokens (masked pixels zeroed, mask given as a channel),
    linear head back to pixels. Altitude enters through FiLM on the token embeddings."""

    def __init__(self, n=64, patch=4, width=128, depth=4, heads=4, altitude=True,
                 mup=False, base_width=64):
        super().__init__()
        self.n, self.p, self.width, self.mup = n, patch, width, mup
        T = (n // patch) ** 2
        self.embed = nn.Linear(2 * patch * patch, width)
        self.pos = nn.Parameter(torch.randn(1, T, width) * 0.02)
        self.film = None
        if altitude:
            self.film = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, 2 * width))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        self.blocks = nn.ModuleList(Block(width, heads, mup) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, patch * patch)
        # muP output multiplier: keeps the head's contribution O(1) as width grows
        self.out_mult = base_width / width if mup else 1.0
        self.base_width = base_width

    def patchify(self, x):
        B, C, n, _ = x.shape
        g, p = n // self.p, self.p
        return x.view(B, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * p * p)

    def unpatchify(self, y):
        B = y.shape[0]
        g, p = self.n // self.p, self.p
        return y.view(B, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(B, self.n, self.n)

    def forward(self, x, hnorm):
        t = self.embed(self.patchify(x)) + self.pos
        if self.film is not None:
            gamma, beta = self.film(hnorm.view(-1, 1)).chunk(2, dim=-1)
            t = t * (1 + gamma[:, None]) + beta[:, None]
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        # fp32 head even under bf16 autocast: floors are ~0.1 nT^2 on a ~50 nT signal,
        # and bf16 only carries three significant digits
        with torch.autocast(device_type=t.device.type, enabled=False):
            y = self.head(t.float()) * self.out_mult
        return self.unpatchify(y)

    def param_groups(self, lr, wd):
        # under muP with Adam, hidden and output matrices get lr * base/width
        mult = self.base_width / self.width if self.mup else 1.0
        scaled, plain_w, plain_nw = [], [], []
        for name, p in self.named_parameters():
            if self.mup and p.ndim == 2 and (name.startswith("blocks") or name == "head.weight"):
                scaled.append(p)
            elif p.ndim >= 2:
                plain_w.append(p)
            else:
                plain_nw.append(p)
        groups = [dict(params=plain_w, lr=lr, weight_decay=wd), dict(params=plain_nw, lr=lr, weight_decay=0.0)]
        if scaled:
            groups.append(dict(params=scaled, lr=lr * mult, weight_decay=wd))
        return groups


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    """Stationary-convolution control. The optimal predictor is a fixed linear
    filter, so this architecture has exactly the right inductive bias."""

    def __init__(self, n=64, width=16, levels=3, altitude=True):
        super().__init__()
        self.n, self.altitude, self.mup = n, altitude, False
        cin = 3 if altitude else 2
        ch = [width * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList([conv_block(cin, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(levels)])
        self.dec = nn.ModuleList(conv_block(ch[i + 1] + ch[i], ch[i]) for i in range(levels))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x, hnorm):
        if self.altitude:
            x = torch.cat([x, hnorm.view(-1, 1, 1, 1).expand(-1, 1, self.n, self.n)], dim=1)
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for d in reversed(self.dec):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = d(torch.cat([x, skips.pop()], dim=1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.head(x.float())[:, 0]

    def param_groups(self, lr, wd):
        w = [p for p in self.parameters() if p.ndim >= 2]
        nw = [p for p in self.parameters() if p.ndim < 2]
        return [dict(params=w, lr=lr, weight_decay=wd), dict(params=nw, lr=lr, weight_decay=0.0)]


def build(kind, n, width, depth, heads, altitude, mup=False, base_width=64, patch=4):
    if kind == "vit":
        return ViT(n, patch, width, depth, heads, altitude, mup, base_width)
    if kind == "unet":
        return UNet(n, width, depth, altitude)
    raise ValueError(kind)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
