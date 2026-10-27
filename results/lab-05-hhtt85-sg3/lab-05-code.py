# An R3GAN (Huang et al. 2024) for CIFAR-10 horses: relativistic pairing loss with R1 + R2
# zero-centred gradient penalties, normalization-free ConvNeXt-style residual networks, and a
# StyleGAN3-style generator (mapping network, Fourier input, modulated convs, filtered activations,
# style mixing) trained with blurred D inputs and adaptive discriminator augmentation (ADA).
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import time
import zipfile
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_fidelity import calculate_metrics  # pip install torch-fidelity
from torch.nn import functional as F

STUDENT_ID = "hhtt85"
DATA_DIR = Path("data")

# %% Model and training settings

CONFIG = dict(task="dcgan", steps=10000, batch_size=64,
              learning_rate=2e-4, final_learning_rate=5e-5, betas=(0.0, 0.99), weight_decay=0.0,
              gamma=0.05, final_gamma=0.005, ema_kimg=10.0, ema_rampup=0.05, xflip=True,
              blur_sigma=2.0, blur_fade_kimg=100.0, truncation_psi=1.0, style_mixing_prob=0.5,
              ada_target=0.6, ada_kimg=100.0, ada_interval=4,
              log_every=100, sample_every=1000)

LEAK = 0.2
GAIN = math.sqrt(2/(1 + LEAK**2))  # keeps activation variance through leaky ReLU
MAGNITUDE_BETA = 0.5**(64/20000)    # StyleGAN3's input-magnitude EMA, half-life of 20k images


def conv(c_in, c_out, kernel=1, groups=1, gain=1.0, bias=True):
    """Convolution with magnitude-preserving init; gain=0 gives a zero-initialised layer."""
    layer = nn.Conv2d(c_in, c_out, kernel, padding=kernel//2, groups=groups, bias=bias)
    nn.init.normal_(layer.weight, 0, gain/math.sqrt(layer.weight[0].numel()))
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class EqualLinear(nn.Module):
    """Linear layer with StyleGAN's equalized learning rate: weights are stored at unit scale and
    rescaled at runtime, so `lr_mult` sets this layer's effective learning rate."""
    def __init__(self, c_in, c_out, bias_init=0.0, lr_mult=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c_out, c_in)/lr_mult)
        self.bias = nn.Parameter(torch.full((c_out,), float(bias_init)))
        self.weight_gain = lr_mult/math.sqrt(c_in)
        self.lr_mult = lr_mult

    def forward(self, x):
        return F.linear(x, self.weight*self.weight_gain, self.bias*self.lr_mult)


def lowpass_taps(numtaps, cutoff, width, fs):
    """Kaiser-windowed sinc low-pass filter with unit DC gain (scipy.signal.firwin equivalent)."""
    atten = 2.285*(numtaps - 1)*math.pi*(width/(fs/2)) + 7.95
    if atten > 50:
        beta = 0.1102*(atten - 8.7)
    elif atten > 21:
        beta = 0.5842*(atten - 21)**0.4 + 0.07886*(atten - 21)
    else:
        beta = 0.0
    taps = np.sinc(2*cutoff/fs*(np.arange(numtaps) - (numtaps - 1)/2))*np.kaiser(numtaps, beta)
    return torch.tensor(taps/taps.sum(), dtype=torch.float32)


def fir_resample(x, taps, up=1, down=1):
    """Separable FIR filter with zero-insertion upsampling or decimation (StyleGAN3's upfirdn2d)."""
    n, c, h, w = x.shape
    if up > 1:
        x = F.pad(x[:, :, :, None, :, None], (0, up - 1, 0, 0, 0, up - 1)).reshape(n, c, h*up, w*up)
    k = len(taps)
    p0, p1 = ((k + up - 1)//2, (k - up)//2) if up > 1 else ((k - down + 1)//2, (k - down)//2)
    x = F.pad(x, (p0, p1, p0, p1))
    taps = (taps*up).to(x.dtype)  # zero insertion loses a factor of `up` per dimension
    x = F.conv2d(x, taps.view(1, 1, 1, k).expand(c, 1, 1, k), stride=(1, down), groups=c)
    return F.conv2d(x, taps.view(1, 1, k, 1).expand(c, 1, k, 1), stride=(down, 1), groups=c)


class FilteredLeakyReLU(nn.Module):
    """Alias-suppressed leaky ReLU (StyleGAN3): upsample 2x, apply the nonlinearity, low-pass at
    this layer's cutoff and downsample, so the high frequencies it creates are mostly removed
    rather than aliased. The filter's transition band ends at max(stopband, Nyquist)."""
    def __init__(self, sampling_rate, cutoff, stopband, numtaps=12):
        super().__init__()
        half_width = max(stopband, sampling_rate/2) - cutoff
        self.register_buffer("taps", lowpass_taps(numtaps, cutoff, 2*half_width, 2*sampling_rate))

    def forward(self, x):
        x = F.leaky_relu(fir_resample(x, self.taps, up=2), LEAK)
        return fir_resample(x, self.taps, down=2)


class MagnitudeNorm(nn.Module):
    """Divides by the running RMS of its input (StyleGAN3's magnitude_ema input normalization)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("magnitude_ema", torch.ones([]))

    def forward(self, x):
        if self.training:
            self.magnitude_ema.copy_(x.detach().square().mean().lerp(self.magnitude_ema, MAGNITUDE_BETA))
        return x*self.magnitude_ema.rsqrt()


class Block(nn.Module):
    """ConvNeXt-style inverted bottleneck: 1x1 expand, grouped 3x3, 1x1 project, no normalization.

    Fix-up style init: the last layer starts at zero and the others are scaled by
    L^(-1/(2m-2)) for L residual blocks with m=3 layers each, as in R3GAN. With `w_dim` set (the
    generator), the branch input is magnitude-normalized, the expansion is a StyleGAN
    modulated/demodulated conv driven by w, and the activations are alias-suppressed with
    `filter_spec` = (sampling rate, cutoff, stopband)."""
    def __init__(self, width, total_blocks, expansion=2, group_width=16, w_dim=None, filter_spec=None):
        super().__init__()
        hidden = width*expansion
        self.gain = GAIN*total_blocks**(-1/4)
        self.expand = conv(width, hidden, 1, gain=self.gain)
        self.spatial = conv(hidden, hidden, 3, groups=hidden//group_width, gain=self.gain)
        self.project = conv(hidden, width, 1, gain=0, bias=False)
        self.style = EqualLinear(w_dim, width, bias_init=1) if w_dim else None
        self.normalize = MagnitudeNorm() if w_dim else None
        self.activation = (FilteredLeakyReLU(*filter_spec) if filter_spec
                           else partial(F.leaky_relu, negative_slope=LEAK))

    def forward(self, x, w=None):
        if self.style is None:
            y = self.expand(x)
        else:
            styles = self.style(w)                                           # N x C
            weight = self.expand.weight[:, :, 0, 0]                          # H x C
            demod = (weight[None]*styles[:, None]).square().sum(2).add(1e-8).rsqrt()  # N x H
            y = F.conv2d(self.normalize(x)*styles[:, :, None, None], self.expand.weight)
            y = y*(demod*self.gain)[:, :, None, None] + self.expand.bias[:, None, None]
        y = self.activation(y)
        y = self.activation(self.spatial(y))
        return x + self.project(y)


class Upsample(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class Downsample(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 2)


class FourierInput(nn.Module):
    """StyleGAN3-style input: sinusoids with random frequencies on a size x size grid, rotated and
    translated per sample by an affine map of w, then mixed into channels by a learned matrix.

    Frequencies are band-limited to `bandwidth` cycles per unit; those between the bandwidth and
    Nyquist (size/2) are faded out to limit aliasing on the coarse grid."""
    def __init__(self, w_dim, channels, features=64, size=4, bandwidth=1.0):
        super().__init__()
        freqs = torch.randn(features, 2)
        radii = freqs.norm(dim=1, keepdim=True)
        freqs = freqs/(radii*radii.square().exp().pow(0.25))*bandwidth
        nyquist = size/2
        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", torch.rand(features) - 0.5)
        self.register_buffer("amplitudes",
                             (1 - (freqs.norm(dim=1) - bandwidth)/(nyquist - bandwidth)).clamp(0, 1))
        coords = (torch.arange(size) + 0.5)/size - 0.5  # pixel centres spanning one unit
        self.register_buffer("grid", torch.stack(torch.meshgrid(coords, coords, indexing="xy"), -1))
        # Rotation (cos, sin) and translation (x, y); starts as the identity transform.
        self.transform = EqualLinear(w_dim, 4)
        with torch.no_grad():
            self.transform.weight.zero_()
            self.transform.bias.copy_(torch.tensor([1., 0., 0., 0.]))
        self.mix = nn.Parameter(torch.randn(channels, features)/math.sqrt(features))

    def forward(self, w):
        t = self.transform(w)
        rotation = t[:, :2]/t[:, :2].norm(dim=1, keepdim=True)
        cos, sin = rotation.unbind(1)
        matrix = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
        freqs = self.freqs@matrix                               # N x F x 2
        phases = self.phases + (freqs@t[:, 2:, None]).squeeze(-1)  # N x F
        angles = torch.einsum("hwc,nfc->nfhw", self.grid, freqs) + phases[:, :, None, None]
        features = torch.sin(2*math.pi*angles)*self.amplitudes[:, None, None]
        return torch.einsum("cf,nfhw->nchw", self.mix, features)


class MappingNetwork(nn.Module):
    """z -> w: normalize z, then an MLP at 1/100 learning rate; tracks the mean w for truncation."""
    def __init__(self, latent, w_dim, layers=2, lr_mult=0.01, w_avg_beta=0.998):
        super().__init__()
        self.layers = nn.ModuleList(EqualLinear(latent if i == 0 else w_dim, w_dim, lr_mult=lr_mult)
                                    for i in range(layers))
        self.register_buffer("w_avg", torch.zeros(w_dim))
        self.w_avg_beta = w_avg_beta

    def forward(self, z, psi=1.0):
        w = z*z.square().mean(1, keepdim=True).add(1e-8).rsqrt()
        for layer in self.layers:
            w = F.leaky_relu(layer(w), LEAK)*GAIN
        if self.training:
            self.w_avg.lerp_(w.detach().mean(0), 1 - self.w_avg_beta)
        return w if psi == 1 else self.w_avg.lerp(w, psi)


class ToRGB(nn.Module):
    """Modulated 1x1 conv without demodulation, as in StyleGAN."""
    def __init__(self, width, w_dim):
        super().__init__()
        self.style = EqualLinear(w_dim, width, bias_init=1)
        self.weight = nn.Parameter(torch.randn(3, width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(3))
        self.weight_gain = 1/math.sqrt(width)
        self.normalize = MagnitudeNorm()

    def forward(self, x, w):
        styles = self.style(w)*self.weight_gain
        return F.conv2d(self.normalize(x)*styles[:, :, None, None], self.weight) + self.bias[:, None, None]


def filter_schedule(rates, first_cutoff, critical=2):
    """StyleGAN3's per-layer filters: cutoffs rise geometrically from `first_cutoff` to the final
    Nyquist frequency, reached `critical` layers before the end; the stopband sits 2^2.1/2 above
    the cutoff at first and 2^0.3 above it by then. Cutoffs never exceed a layer's own Nyquist."""
    last_cutoff = rates[-1]/2
    specs = []
    for i, rate in enumerate(rates):
        e = min(i/max(1, len(rates) - critical), 1)
        cutoff = min(first_cutoff*(last_cutoff/first_cutoff)**e, rate/2)
        specs.append((rate, cutoff, cutoff*2**(1.1*(1 - e) + 0.3*e)))
    return specs


class Generator(nn.Module):
    """StyleGAN3-flavoured R3GAN generator: mapping network, Fourier-feature input, then
    style-modulated residual stages at 4, 8, 16 and 32 pixels with per-layer anti-aliasing
    filters, and a modulated RGB output. Each block, the input and the output take their own w,
    so styles from two latents can be mixed during training."""
    def __init__(self, latent=64, w_dim=64, widths=(96, 96, 96, 64), blocks_per_stage=2,
                 first_cutoff=1.0):
        super().__init__()
        self.latent = latent
        self.truncation_psi = 1.0
        self.style_mixing_prob = 0.0
        total = len(widths)*blocks_per_stage
        rates = [4*2**i for i in range(len(widths)) for _ in range(blocks_per_stage)]
        specs = iter(filter_schedule(rates, first_cutoff))
        self.mapping = MappingNetwork(latent, w_dim)
        self.input = FourierInput(w_dim, widths[0], bandwidth=first_cutoff)
        self.layers = nn.ModuleList()
        for i, width in enumerate(widths):
            if i:
                self.layers.append(Upsample())
                if width != widths[i - 1]:
                    self.layers.append(conv(widths[i - 1], width, 1))
            self.layers.extend(Block(width, total, w_dim=w_dim, filter_spec=next(specs))
                               for _ in range(blocks_per_stage))
        self.to_rgb = ToRGB(widths[-1], w_dim)
        self.num_ws = total + 2

    def forward(self, z):
        psi = 1.0 if self.training else self.truncation_psi
        ws = self.mapping(z, psi)[:, None].expand(-1, self.num_ws, -1)
        if self.training and self.style_mixing_prob > 0:
            cutoff = torch.randint(1, self.num_ws, (len(z),), device=z.device)
            cutoff = torch.where(torch.rand(len(z), device=z.device) < self.style_mixing_prob,
                                 cutoff, self.num_ws)
            mixed = torch.arange(self.num_ws, device=z.device)[None] >= cutoff[:, None]
            ws = torch.where(mixed[:, :, None], self.mapping(torch.randn_like(z), psi)[:, None], ws)
        x = self.input(ws[:, 0])
        index = 1
        for layer in self.layers:
            if isinstance(layer, Block):
                x = layer(x, ws[:, index])
                index += 1
            else:
                x = layer(x)
        return self.to_rgb(x, ws[:, index])


class Discriminator(nn.Module):
    """Mirror of the generator, ending in a depthwise 4x4 basis projection and a linear logit."""
    def __init__(self, widths=(64, 96, 96, 96), blocks_per_stage=2):
        super().__init__()
        total = len(widths)*blocks_per_stage
        layers = [conv(3, widths[0], 1)]
        for i, width in enumerate(widths):
            if i:
                if width != widths[i - 1]:
                    layers.append(conv(widths[i - 1], width, 1))
                layers.append(Downsample())
            layers += [Block(width, total) for _ in range(blocks_per_stage)]
        self.stages = nn.Sequential(*layers)
        self.basis = conv(widths[-1], widths[-1], 4, groups=widths[-1], bias=False)
        self.basis.padding = 0
        self.logit = nn.Linear(widths[-1], 1)

    def forward(self, x):
        return self.logit(self.basis(self.stages(x)).flatten(1)).squeeze(-1)


class Model(nn.Module):
    def __init__(self, latent=64, generator_widths=(96, 96, 96, 64),
                 discriminator_widths=(64, 96, 96, 96)):
        super().__init__()
        self.generator = Generator(latent=latent, widths=generator_widths)
        self.discriminator = Discriminator(widths=discriminator_widths)


def build_model():
    model = Model()
    model.generator.truncation_psi = CONFIG["truncation_psi"]
    model.generator.style_mixing_prob = CONFIG["style_mixing_prob"]
    return model

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = (1000000, 10000, 64)

# %% Data, evaluation and submission
def program_source():
    if "__file__" in globals():
        source = Path(__file__).read_text()
    else:
        source = get_ipython().history_manager.input_hist_raw[-1].rstrip() + "\n"
    tree = ast.parse(source)
    names = {getattr(node, "name", "") for node in tree.body}
    if not {"build_model", "run", "save_submission"} <= names:
        raise ValueError("Paste the complete program into one notebook cell and run that cell.")
    return source


def model_size(model):
    parameters = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    buffers = sum(b.numel()*(2 if b.is_complex() else 1) for b in model.buffers())
    if parameters + buffers > MAX_PARAMETERS:
        raise ValueError(f"Model has {parameters + buffers} values; limit is {MAX_PARAMETERS}.")
    return parameters, buffers


def save_submission(folder, source, history, metrics, size, seed, split):
    stem = "lab-05"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.png" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"generator": CONFIG["steps"], "discriminator": CONFIG["steps"]}
    result = dict(format=6, lab=5, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-cifar10-horses-kid1000-v1-" + split, code_sha256=digest(code),
                  metrics=metrics, loss=history,
                  samples_sha256={name: digest(folder/"samples"/name) for name in names})
    record = folder / (stem + "-results.json")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = folder.with_name(folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in [code, record] + [folder/"samples"/name for name in names]:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2))
    print("Submission:", archive.resolve())
    return result


def show_results(history, folder):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 2.5))
    curves = np.array(history)
    plt.plot(curves[:, 0], curves[:, 1], label="Generator")
    plt.plot(curves[:, 0], curves[:, 2], label="Discriminator")
    plt.legend()
    plt.xlabel("GAN iteration"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()
    from PIL import Image
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(folder/"samples/batch-01.png"))
    plt.axis("off"); plt.tight_layout(); plt.show()


HF_CIFAR10 = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def hf_horses(train):
    """CIFAR-10 horses from the Hugging Face mirror, in the original dataset order."""
    import io
    import urllib.request
    import pyarrow.parquet as pq
    from PIL import Image
    path = DATA_DIR/"hf"/("train.parquet" if train else "test.parquet")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF_CIFAR10.format(path.stem), path)
    table = pq.read_table(path).to_pydict()
    return np.stack([np.asarray(Image.open(io.BytesIO(image["bytes"])).convert("RGB"))
                     for image, label in zip(table["img"], table["label"]) if label == 7])


def image_data(split="train"):
    from torchvision.datasets import CIFAR10
    try:  # The official archive when it is already on disk; its server is often very slow.
        dataset = CIFAR10(root=DATA_DIR, train=split != "test", download=False)
        pixels = dataset.data[np.asarray(dataset.targets) == 7]
    except RuntimeError:
        pixels = hf_horses(train=split != "test")
    index = torch.arange(len(pixels))
    if split != "test":
        order = torch.randperm(len(index), generator=torch.Generator().manual_seed(3802))
        index = index[order[:4000] if split == "train" else order[4000:]]
    images = torch.from_numpy(pixels).permute(0, 3, 1, 2)[index].float()/127.5 - 1
    return images, torch.zeros(len(images), dtype=torch.long)


def image_bytes(images):
    return ((images.clamp(-1, 1) + 1)*127.5 + .5).floor().to(torch.uint8)


class Images(torch.utils.data.Dataset):
    """Give the KID library one uint8 RGB image at a time."""
    def __init__(self, pixels): self.pixels = pixels
    def __len__(self): return len(self.pixels)
    def __getitem__(self, index): return self.pixels[index]


@torch.no_grad()
def image_score(model, folder, device, split):
    from PIL import Image
    model.eval()
    rng = torch.Generator(device=device).manual_seed(921)
    fake = torch.cat([generate(model, 50, device, rng).clamp(-1, 1).cpu() for _ in range(20)])
    real, _ = image_data(split)
    pixels = image_bytes(fake)
    # One full set of 1,000 images, using the same pixels as the saved grids.
    metrics = calculate_metrics(
        input1=Images(image_bytes(real)), input2=Images(pixels),
        cuda=torch.device(device).type == "cuda", batch_size=32,
        kid=True, kid_subset_size=1000, kid_subsets=1,
        feature_extractor="inception-v3-compat", feature_layer_kid="2048",
        rng_seed=921, save_cpu_ram=True, verbose=False)
    score = metrics["kernel_inception_distance_mean"]
    for i, batch in enumerate(pixels.split(100), 1):
        grid = batch.reshape(10, 10, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(320, 320, 3)
        Image.fromarray(grid.numpy()).save(folder/f"batch-{i:02d}.png")
    print(f"KID x 1000: {1000*score:.3f}")
    return {"kid": score, "samples": 1000}


def discriminator_loss(real_logits, fake_logits):
    """Relativistic pairing loss: softplus(D(G(z)) - D(x))."""
    return F.softplus(fake_logits - real_logits).mean()

def generator_loss(fake_logits, real_logits):
    return F.softplus(real_logits - fake_logits).mean()

def gradient_penalty(logits, inputs):
    """Zero-centred squared gradient norm of D at each input (R1 on reals, R2 on fakes)."""
    grad, = torch.autograd.grad(logits.sum(), inputs, create_graph=True)
    return grad.square().sum(dim=(1, 2, 3))

def blur(images, sigma):
    """Gaussian blur applied to everything D sees early in training (StyleGAN3's blur_init_sigma)."""
    if sigma <= 0:
        return images
    radius = max(1, round(3*sigma))
    taps = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    taps = taps.div(sigma).square().neg().exp2()
    taps = (taps/taps.sum()).expand(images.size(1), 1, -1)
    images = F.pad(images, (radius,)*4, mode="reflect")
    images = F.conv2d(images, taps[:, :, None, :], groups=images.size(1))
    return F.conv2d(images, taps[:, :, :, None], groups=images.size(1))

def batch_eye(n, size, device):
    return torch.eye(size, device=device).repeat(n, 1, 1)

def translate2d(tx, ty):
    m = batch_eye(len(tx), 3, tx.device)
    m[:, 0, 2], m[:, 1, 2] = tx, ty
    return m

def scale2d(sx, sy):
    m = batch_eye(len(sx), 3, sx.device)
    m[:, 0, 0], m[:, 1, 1] = sx, sy
    return m

def rotate2d(theta):
    m = batch_eye(len(theta), 3, theta.device)
    m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = theta.cos(), -theta.sin(), theta.sin(), theta.cos()
    return m


class AdaptiveAugment:
    """ADA (Karras et al. 2020), 'bgc' pipeline: per-sample geometric and colour transforms, each
    applied with probability p, on everything D sees (reals and fakes alike, so the augmentations
    do not leak into G while p stays below ~0.8). p is raised when D separates reals from fakes
    too reliably, measured here as r_t = E[sign(D(x) - D(G(z)))] since relativistic logits have no
    absolute zero, and lowered otherwise. Transforms are differentiable, so R1/R2 stay w.r.t. the
    un-augmented images."""
    def __init__(self, target, speed_kimg, interval, batch):
        self.p, self.target, self.interval = 0.0, target, interval
        self.step_size = batch*interval/(speed_kimg*1000)
        self.signs = []
        self.rt = 0.0

    def update(self, real_logits, fake_logits):
        self.signs.append(torch.sign(real_logits - fake_logits).mean().detach())
        if len(self.signs) == self.interval:
            self.rt = torch.stack(self.signs).mean().item()
            self.p = min(1.0, max(0.0, self.p + math.copysign(self.step_size, self.rt - self.target)))
            self.signs.clear()

    def __call__(self, images):
        if self.p <= 0:
            return images
        n, _, h, w = images.shape
        device = images.device
        def chance(p=self.p):
            return torch.rand(n, device=device) < p
        def uniform(low, high):
            return torch.rand(n, device=device)*(high - low) + low
        one, zero = torch.ones(n, device=device), torch.zeros(n, device=device)
        p_rot = 1 - math.sqrt(1 - self.p)  # pre- and post-rotation together fire with chance p

        # Geometric transforms, composed in normalized [-1, 1] coordinates and applied in one go.
        g = batch_eye(n, 3, device)
        flip = 1 - 2*torch.randint(2, (n,), device=device)
        g = g@scale2d(torch.where(chance(), flip.float(), one), one)
        quarter = torch.randint(4, (n,), device=device)*(math.pi/2)
        g = g@rotate2d(torch.where(chance(), quarter, zero))
        shift = lambda size: torch.where(chance(), (uniform(-0.125, 0.125)*size).round()/size*2, zero)
        g = g@translate2d(shift(w), shift(h))
        g = g@scale2d(*[torch.where(chance(), 2**(torch.randn(n, device=device)*0.2), one)]*2)
        g = g@rotate2d(torch.where(chance(p_rot), uniform(-math.pi, math.pi), zero))
        aniso = torch.where(chance(), 2**(torch.randn(n, device=device)*0.2), one)
        g = g@scale2d(aniso, 1/aniso)
        g = g@rotate2d(torch.where(chance(p_rot), uniform(-math.pi, math.pi), zero))
        g = g@translate2d(*[torch.where(chance(), torch.randn(n, device=device)*0.125*2, zero)
                            for _ in range(2)])
        grid = F.affine_grid(g[:, :2], list(images.shape), align_corners=False)
        images = F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection",
                               align_corners=False)

        # Colour transforms as 4x4 affine matrices on RGB, around the luma axis v.
        v = torch.tensor([1., 1., 1.], device=device)/math.sqrt(3)
        vv = torch.outer(v, v)
        eye = torch.eye(3, device=device)
        c = batch_eye(n, 4, device)
        def colour(matrix=None, offset=None):
            m = batch_eye(n, 4, device)
            if matrix is not None:
                m[:, :3, :3] = matrix
            if offset is not None:
                m[:, :3, 3] = offset
            return m
        brightness = torch.where(chance(), torch.randn(n, device=device)*0.2, zero)
        c = colour(offset=brightness[:, None].expand(n, 3))@c
        contrast = torch.where(chance(), 2**(torch.randn(n, device=device)*0.5), one)
        c = colour(matrix=contrast[:, None, None]*eye)@c
        lumaflip = torch.where(chance(), torch.randint(2, (n,), device=device).float(), zero)
        c = colour(matrix=eye - 2*vv*lumaflip[:, None, None])@c
        theta = torch.where(chance(), uniform(-math.pi, math.pi), zero)[:, None, None]
        cross = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=device)
        c = colour(matrix=theta.cos()*eye + theta.sin()*cross + (1 - theta.cos())*vv)@c
        saturation = torch.where(chance(), 2**torch.randn(n, device=device), one)[:, None, None]
        c = colour(matrix=vv + (eye - vv)*saturation)@c
        pixels = c[:, :3, :3]@images.reshape(n, 3, -1) + c[:, :3, 3:]
        return pixels.reshape(images.shape)


def cosine(start, end, progress):
    return end + (start - end)*0.5*(1 + math.cos(math.pi*progress))

def start_logging(model):
    """W&B run if wandb is installed; offline unless an API key or WANDB_MODE says otherwise."""
    try:
        import wandb
    except ImportError:
        return None
    import netrc
    from urllib.parse import urlparse
    try:
        host = urlparse(wandb.Settings().base_url).hostname
        logged_in = netrc.netrc().authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        logged_in = False
    logged_in = logged_in or bool(os.environ.get("WANDB_API_KEY"))
    mode = os.environ.get("WANDB_MODE") or ("online" if logged_in else "offline")
    parameters = {name: sum(p.numel() for p in part.parameters())
                  for name, part in model.named_children()}
    return wandb.init(project=os.environ.get("WANDB_PROJECT", "r3gan-cifar-horses"), mode=mode,
                      config=dict(CONFIG, student=STUDENT_ID, parameters=parameters))

@torch.no_grad()
def log_samples(run, generator, noise, step):
    pixels = image_bytes(generator(noise)).cpu()
    grid = pixels.reshape(8, 8, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(256, 256, 3).numpy()
    import wandb
    run.log({"samples": wandb.Image(grid, caption=f"EMA generator, step {step}")}, step=step)

# %% Training
def train(model, device):
    images, _ = image_data()
    images = images.to(device)
    generator, discriminator = model.generator, model.discriminator
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=CONFIG["learning_rate"],
                             betas=CONFIG["betas"], weight_decay=CONFIG["weight_decay"])
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["learning_rate"],
                             betas=CONFIG["betas"], weight_decay=CONFIG["weight_decay"])
    steps, batch = CONFIG["steps"], CONFIG["batch_size"]
    augment = AdaptiveAugment(CONFIG["ada_target"], CONFIG["ada_kimg"], CONFIG["ada_interval"], batch)
    run = start_logging(model)
    fixed_noise = torch.randn(64, generator.latent, device=device,
                              generator=torch.Generator(device=device).manual_seed(0))
    history = []
    model.train()
    clock = time.perf_counter()
    for step in range(1, steps + 1):
        progress = (step - 1)/max(1, steps - 1)
        lr = cosine(CONFIG["learning_rate"], CONFIG["final_learning_rate"], progress)
        gamma = cosine(CONFIG["gamma"], CONFIG["final_gamma"], progress)
        for group in opt_g.param_groups + opt_d.param_groups:
            group["lr"] = lr
        sigma = CONFIG["blur_sigma"]*max(0.0, 1 - step*batch/(CONFIG["blur_fade_kimg"]*1000))
        critic = lambda images: discriminator(augment(blur(images, sigma)))

        real = images[torch.randint(len(images), (batch,), device=device)]
        if CONFIG["xflip"]:
            flip = torch.rand(batch, 1, 1, 1, device=device) < 0.5
            real = torch.where(flip, real.flip(3), real)
        noise = torch.randn(batch, generator.latent, device=device)
        fake = generator(noise)

        # Discriminator: adversarial loss + gamma/2 (R1 + R2), reals and fakes in one pass.
        pair = torch.cat([real, fake.detach()]).requires_grad_(True)
        logits = critic(pair)
        r1, r2 = gradient_penalty(logits, pair).chunk(2)
        real_logits, fake_logits = logits.chunk(2)
        adversarial = discriminator_loss(real_logits, fake_logits)
        d_loss = adversarial + gamma/2*(r1 + r2).mean()
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward(); opt_d.step()
        augment.update(real_logits, fake_logits)

        # Generator: the same relativistic loss with the roles swapped.
        discriminator.requires_grad_(False)
        with torch.no_grad():
            real_logits = critic(real)
        g_loss = generator_loss(critic(fake), real_logits)
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward(); opt_g.step()
        discriminator.requires_grad_(True)

        # Exponential moving average of generator weights, with a half-life of ema_kimg images.
        ema_images = min(CONFIG["ema_kimg"]*1000, step*batch*CONFIG["ema_rampup"])
        beta = 0.5**(batch/max(ema_images, 1e-8))
        torch._foreach_lerp_(list(ema.parameters()), list(generator.parameters()), 1 - beta)
        torch._foreach_copy_(list(ema.buffers()), list(generator.buffers()))  # e.g. w_avg

        if step == 1 or step % max(1, steps//40) == 0 or step == steps:
            history.append([step, round(g_loss.item(), 6), round(d_loss.item(), 6)])
        if step == 1 or step % CONFIG["log_every"] == 0 or step == steps:
            stats = {"loss/generator": g_loss.item(), "loss/discriminator": d_loss.item(),
                     "loss/adversarial": adversarial.item(), "penalty/r1": r1.mean().item(),
                     "penalty/r2": r2.mean().item(), "logits/real": real_logits.mean().item(),
                     "logits/fake": fake_logits.mean().item(), "schedule/lr": lr,
                     "schedule/gamma": gamma, "schedule/ema_beta": beta,
                     "schedule/blur_sigma": sigma, "ada/p": augment.p, "ada/rt": augment.rt}
            rate = step/(time.perf_counter() - clock)
            print(f"{step:>5} / {steps}: G {stats['loss/generator']:.4f}  "
                  f"D {stats['loss/discriminator']:.4f}  R1 {stats['penalty/r1']:.3f}  "
                  f"R2 {stats['penalty/r2']:.3f}  D(x)-D(G(z)) "
                  f"{stats['logits/real'] - stats['logits/fake']:+.3f}  "
                  f"lr {lr:.2e}  gamma {gamma:.4f}  blur {sigma:.2f}  ada p {augment.p:.3f} "
                  f"(r_t {augment.rt:+.2f})  {rate:.1f} it/s", flush=True)
            if run:
                run.log(dict(stats, **{"speed/it_per_s": rate}), step=step)
        if run and (step == 1 or step % CONFIG["sample_every"] == 0 or step == steps):
            log_samples(run, ema, fixed_noise, step)

    generator.load_state_dict(ema.state_dict())
    return history, run


@torch.no_grad()
def generate(model, count, device, generator):
    return model.generator(torch.randn(count, model.generator.latent, device=device, generator=generator))

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] != "dcgan":
        raise ValueError("Keep task='dcgan' in CONFIG.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-05-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history, wandb_run = train(model, device)
    metrics = image_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if wandb_run:
        import wandb
        wandb_run.log({"kid_x1000": 1000*metrics["kid"],
                       "final_samples": wandb.Image(str(folder/"samples/batch-01.png"))})
        wandb_run.finish()
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()

