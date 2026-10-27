# An R3GAN (Huang et al. 2024) for CIFAR-10 horses: relativistic pairing loss with R1 + R2
# zero-centred gradient penalties, and normalization-free ConvNeXt-style residual networks.
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
              aug_p=0.0, initial_beta2=None,
              architecture=dict(latent=64, generator_widths=(96, 96, 96, 64),
                                discriminator_widths=(64, 96, 96, 96), generator_blocks=2,
                                discriminator_blocks=2, expansion=2, group_width=16),
              log_every=100, sample_every=1000)

LEAK = 0.2
GAIN = math.sqrt(2/(1 + LEAK**2))  # keeps activation variance through leaky ReLU


def conv(c_in, c_out, kernel=1, groups=1, gain=1.0, bias=True):
    """Convolution with magnitude-preserving init; gain=0 gives a zero-initialised layer."""
    layer = nn.Conv2d(c_in, c_out, kernel, padding=kernel//2, groups=groups, bias=bias)
    nn.init.normal_(layer.weight, 0, gain/math.sqrt(layer.weight[0].numel()))
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class Block(nn.Module):
    """ConvNeXt-style inverted bottleneck: 1x1 expand, grouped 3x3, 1x1 project, no normalization.

    Fix-up style init: the last layer starts at zero and the others are scaled by
    L^(-1/(2m-2)) for L residual blocks with m=3 layers each, as in R3GAN."""
    def __init__(self, width, total_blocks, expansion=2, group_width=16):
        super().__init__()
        hidden = width*expansion
        gain = GAIN*total_blocks**(-1/4)
        self.expand = conv(width, hidden, 1, gain=gain)
        self.spatial = conv(hidden, hidden, 3, groups=hidden//group_width, gain=gain)
        self.project = conv(hidden, width, 1, gain=0, bias=False)

    def forward(self, x):
        y = F.leaky_relu(self.expand(x), LEAK)
        y = F.leaky_relu(self.spatial(y), LEAK)
        return x + self.project(y)


class Upsample(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class Downsample(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 2)


class Generator(nn.Module):
    """4x4 learned basis modulated by z, then residual stages at 4, 8, 16 and 32 pixels."""
    def __init__(self, latent=64, widths=(96, 96, 96, 64), blocks_per_stage=2, **block):
        super().__init__()
        self.latent = latent
        total = len(widths)*blocks_per_stage
        self.basis = nn.Parameter(torch.randn(widths[0], 4, 4))
        self.modulate = nn.Linear(latent, widths[0], bias=False)
        nn.init.normal_(self.modulate.weight, 0, 1/math.sqrt(latent))
        layers = []
        for i, width in enumerate(widths):
            if i:
                layers.append(Upsample())
                if width != widths[i - 1]:
                    layers.append(conv(widths[i - 1], width, 1))
            layers += [Block(width, total, **block) for _ in range(blocks_per_stage)]
        self.stages = nn.Sequential(*layers)
        self.to_rgb = conv(widths[-1], 3, 1)

    def forward(self, z):
        x = self.basis*self.modulate(z)[:, :, None, None]
        return self.to_rgb(self.stages(x))


class Discriminator(nn.Module):
    """Mirror of the generator, ending in a depthwise 4x4 basis projection and a linear logit."""
    def __init__(self, widths=(64, 96, 96, 96), blocks_per_stage=2, **block):
        super().__init__()
        total = len(widths)*blocks_per_stage
        layers = [conv(3, widths[0], 1)]
        for i, width in enumerate(widths):
            if i:
                if width != widths[i - 1]:
                    layers.append(conv(widths[i - 1], width, 1))
                layers.append(Downsample())
            layers += [Block(width, total, **block) for _ in range(blocks_per_stage)]
        self.stages = nn.Sequential(*layers)
        self.basis = conv(widths[-1], widths[-1], 4, groups=widths[-1], bias=False)
        self.basis.padding = 0
        self.logit = nn.Linear(widths[-1], 1)

    def forward(self, x):
        return self.logit(self.basis(self.stages(x)).flatten(1)).squeeze(-1)


class Model(nn.Module):
    def __init__(self, latent=64, generator_widths=(96, 96, 96, 64),
                 discriminator_widths=(64, 96, 96, 96), generator_blocks=2, discriminator_blocks=2,
                 expansion=2, group_width=16):
        super().__init__()
        block = dict(expansion=expansion, group_width=group_width)
        self.generator = Generator(latent=latent, widths=generator_widths,
                                   blocks_per_stage=generator_blocks, **block)
        self.discriminator = Discriminator(widths=discriminator_widths,
                                           blocks_per_stage=discriminator_blocks, **block)


def build_model():
    return Model(**CONFIG["architecture"])

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

def augment(images, p, colour=0.5, cutout=1.0, cutout_size=0.5):
    """StyleGAN2-ADA augmentation as configured by R3GAN's CIFAR-10 preset: geometric transforms
    (x-flip, 90-degree rotation, integer translation, isotropic scaling, rotation, anisotropic
    scaling, fractional translation) each applied with probability p, colour transforms
    (brightness, contrast, luma flip, hue, saturation) with probability colour*p, and cutout
    with probability cutout*p. Applied to everything D sees, and differentiable, so R1/R2 are
    still taken w.r.t. the un-augmented images."""
    if p <= 0:
        return images
    n, device = len(images), images.device
    def chance(q=p):
        return torch.rand(n, device=device) < q
    def uniform(low, high):
        return torch.rand(n, device=device)*(high - low) + low
    one, zero = torch.ones(n, device=device), torch.zeros(n, device=device)
    p_rot = 1 - math.sqrt(1 - p)  # pre- and post-rotation together fire with chance p

    # Geometric transforms, composed in normalized [-1, 1] coordinates and applied in one go.
    h, w = images.shape[-2:]
    g = batch_eye(n, 3, device)
    flip = 1 - 2*torch.randint(2, (n,), device=device).float()
    g = g@scale2d(torch.where(chance(), flip, one), one)
    g = g@rotate2d(torch.where(chance(), torch.randint(4, (n,), device=device)*(math.pi/2), zero))
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
    q = colour*p
    v = torch.tensor([1., 1., 1.], device=device)/math.sqrt(3)
    vv, eye = torch.outer(v, v), torch.eye(3, device=device)
    def affine(matrix=None, offset=None):
        m = batch_eye(n, 4, device)
        if matrix is not None:
            m[:, :3, :3] = matrix
        if offset is not None:
            m[:, :3, 3] = offset
        return m
    c = batch_eye(n, 4, device)
    brightness = torch.where(chance(q), torch.randn(n, device=device)*0.2, zero)
    c = affine(offset=brightness[:, None].expand(n, 3))@c
    contrast = torch.where(chance(q), 2**(torch.randn(n, device=device)*0.5), one)
    c = affine(matrix=contrast[:, None, None]*eye)@c
    lumaflip = torch.where(chance(q), torch.randint(2, (n,), device=device).float(), zero)
    c = affine(matrix=eye - 2*vv*lumaflip[:, None, None])@c
    theta = torch.where(chance(q), uniform(-math.pi, math.pi), zero)[:, None, None]
    cross = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=device)
    c = affine(matrix=theta.cos()*eye + theta.sin()*cross + (1 - theta.cos())*vv)@c
    saturation = torch.where(chance(q), 2**torch.randn(n, device=device), one)[:, None, None]
    c = affine(matrix=vv + (eye - vv)*saturation)@c
    images = (c[:, :3, :3]@images.reshape(n, 3, -1) + c[:, :3, 3:]).reshape(images.shape)

    # Cutout: zero (mid-grey) a cutout_size x cutout_size rectangle at a random centre.
    centre = torch.rand(n, 2, 1, 1, device=device)
    size = torch.where(chance(cutout*p), cutout_size, 0.0)[:, None, None, None]
    ys = ((torch.arange(h, device=device) + 0.5)/h)[None, None, :, None]
    xs = ((torch.arange(w, device=device) + 0.5)/w)[None, None, None, :]
    keep = ((xs - centre[:, :1]).abs() >= size/2) | ((ys - centre[:, 1:]).abs() >= size/2)
    return images*keep

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
        aug_p = cosine(0.0, CONFIG["aug_p"], progress)
        beta2 = CONFIG["betas"][1]
        if CONFIG["initial_beta2"] is not None:  # R3GAN ramps Adam's beta2 up over training
            beta2 = cosine(CONFIG["initial_beta2"], beta2, progress)
            for group in opt_g.param_groups + opt_d.param_groups:
                group["betas"] = (CONFIG["betas"][0], beta2)
        critic = lambda images: discriminator(augment(images, aug_p))

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

        if step == 1 or step % max(1, steps//40) == 0 or step == steps:
            history.append([step, round(g_loss.item(), 6), round(d_loss.item(), 6)])
        if step == 1 or step % CONFIG["log_every"] == 0 or step == steps:
            stats = {"loss/generator": g_loss.item(), "loss/discriminator": d_loss.item(),
                     "loss/adversarial": adversarial.item(), "penalty/r1": r1.mean().item(),
                     "penalty/r2": r2.mean().item(), "logits/real": real_logits.mean().item(),
                     "logits/fake": fake_logits.mean().item(), "schedule/lr": lr,
                     "schedule/gamma": gamma, "schedule/ema_beta": beta, "schedule/aug_p": aug_p,
                     "schedule/beta2": beta2}
            rate = step/(time.perf_counter() - clock)
            print(f"{step:>5} / {steps}: G {stats['loss/generator']:.4f}  "
                  f"D {stats['loss/discriminator']:.4f}  R1 {stats['penalty/r1']:.3f}  "
                  f"R2 {stats['penalty/r2']:.3f}  D(x)-D(G(z)) "
                  f"{stats['logits/real'] - stats['logits/fake']:+.3f}  "
                  f"lr {lr:.2e}  gamma {gamma:.4f}  aug {aug_p:.3f}  {rate:.1f} it/s", flush=True)
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

