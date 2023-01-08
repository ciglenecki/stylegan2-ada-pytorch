# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image

import torch
import torch.nn.functional as F
from pathlib import Path
import dnnlib
import legacy
from utils import stdout_to_file, random_codeword
from training.networks import Generator, SynthesisNetwork
import json
import lovely_tensors as lt

lt.monkey_patch()


def print_generator(generator):
    generator_dict_ = {
        "z_dim": generator.z_dim,
        "c_dim": generator.c_dim,
        "w_dim": generator.w_dim,
        "img_resolution": generator.img_resolution,
        "img_channels": generator.img_channels,
    }

    print(json.dumps(generator_dict_, sort_keys=True, indent=2))
    mapping = generator.mapping
    mapping_dict_ = {
        "z_dim": mapping.z_dim,
        "c_dim": mapping.c_dim,
        "w_dim": mapping.w_dim,
        "num_ws": mapping.num_ws,
        "num_layers": mapping.num_layers,
        "w_avg_beta": mapping.w_avg_beta,
    }
    print("Mapping")
    print(json.dumps(mapping_dict_, sort_keys=True, indent=2))

    print("sythesis")
    sy = generator.synthesis
    sy_dict = {
        "w_dim": sy.w_dim,
        "img_resolution": sy.img_resolution,
        "img_resolution_log2": sy.img_resolution_log2,
        "img_channels": sy.img_channels,
        "block_resolutions": sy.block_resolutions,
        "num_ws": sy.num_ws,
    }
    print(sy_dict)


def project(
    generator: Generator,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match generator output resolution
    *,
    num_steps=1000,
    dlatent_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    vgg16_url="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt",
    verbose=False,
    device: torch.device,
):
    assert target.shape == (
        generator.img_channels,
        generator.img_resolution,
        generator.img_resolution,
    )

    def logprint(*args):
        if verbose:
            print(*args)

    generator = copy.deepcopy(generator).eval().requires_grad_(False).to(device)  # type: ignore
    print_generator(generator)
    # Compute w stats.
    logprint(f"Computing W midpoint and stddev using {dlatent_avg_samples} samples...")
    z_samples_npy = np.random.RandomState(123).randn(
        dlatent_avg_samples, generator.z_dim
    )

    z_samples = torch.from_numpy(z_samples_npy).to(device)
    # mapping is a mapping network
    w_samples = generator.mapping(
        z_samples,
        None,
    )  # [N, L, C]

    # exit(1)
    print("wsamples", w_samples.shape)
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]

    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / dlatent_avg_samples) ** 0.5
    print("w_avg", w_avg.shape)
    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in generator.synthesis.named_buffers()
        if "noise_const" in name
    }

    # Load VGG16 feature detector.
    with dnnlib.util.open_url(vgg16_url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode="area")
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)
    print("target features", target_features)
    w_opt = torch.tensor(
        w_avg, dtype=torch.float32, device=device, requires_grad=True
    )  # pylint: disable=not-callable
    w_out = torch.zeros(
        [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
    )

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, generator.mapping.num_ws, 1])
        synth_images = generator.synthesis(ws, noise_mode="const")

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(
            f"step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}"
        )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    print("WOUT end", w_out.shape)
    return w_out.repeat([1, generator.mapping.num_ws, 1])


# ----------------------------------------------------------------------------


def latent_to_image(synthesis: SynthesisNetwork, latent_w, **synthesis_kwargs):

    synth_image = synthesis(latent_w.unsqueeze(0), **synthesis_kwargs)
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = (
        synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    )

    return synth_image


def pil_to_torch(target_pil: PIL.Image, resize_to_size=None):
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
    )

    if resize_to_size:
        # When ANTIALIAS was initially added, it was the only high-quality filter based on convolutions. It’s name was supposed to reflect this. Starting from Pillow 2.7.0 all resize method are based on convolutions. All of them are antialias from now on. And the real name of the ANTIALIAS filter is Lanczos filter.
        resampling = PIL.Image.ANTIALIAS
        target_pil = target_pil.resize((resize_to_size, resize_to_size), resampling)

    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_torch = torch.tensor(target_uint8.transpose([2, 0, 1]))
    return target_torch


@click.command()
@click.option("--network", "network_pkl", help="Network pickle filename", required=True)
@click.option(
    "--target",
    "target_fname",
    help="Target image file to project to",
    required=True,
    metavar="FILE",
)
@click.option(
    "--num-steps",
    help="Number of optimization steps",
    type=int,
    default=1000,
    show_default=True,
)
@click.option("--seed", help="Random seed", type=int, default=303, show_default=True)
@click.option(
    "--save-video",
    help="Save an mp4 video of optimization progress",
    type=bool,
    default=True,
    show_default=True,
)
@click.option(
    "--outdir", help="Where to save the output images", required=True, metavar="DIR"
)
@click.option(
    "--num-frames", help="Number of interpolation frames to save", type=int, default=10
)
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    num_frames: int,
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:
    
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    device = torch.device("cuda")

    mouth = torch.from_numpy(np.load("data/stylegan2directions/mouth_open.npy")).to(
        device
    )
    print(mouth)
    print(mouth.shape)

    codeword = random_codeword()

    np.random.seed(seed)
    torch.manual_seed(seed)

    experiment_name = (
        f"{Path(target_fname).stem}_seed_{seed}_steps_{num_steps}_{codeword}"
    )
    stdout_to_file(Path("reports", experiment_name + ".txt"))

    # Load networks.
    network_pkl_path = Path(network_pkl)
    print(f'Loading networks from "{str(network_pkl)}"...')

    with dnnlib.util.open_url(network_pkl) as fp:
        generator = legacy.load_network_pkl(fp)["G"].requires_grad_(False).to(device)

        # generator = (
        #     legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)
        # )

    target_pil = PIL.Image.open(target_fname).convert("RGB")
    target = pil_to_torch(target_pil, generator.img_resolution).to(device)

    start_time = perf_counter()
    projected_w_steps = project(
        generator,
        target=target,
        num_steps=num_steps,
        device=device,
        verbose=True,
    )

    print(projected_w_steps[-1])
    print(projected_w_steps[-1, 1])
    print(projected_w_steps[-1, 2])
    print(f"Elapsed: {(perf_counter()-start_time):.1f} s")

    # Render debug output: optional video and projected image and W vector.
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if save_video:
        video = imageio.get_writer(
            f"{outdir}/proj_{seed}.mp4",
            mode="I",
            fps=10,
            codec="libx264",
            bitrate="16M",
        )
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')

        num_of_frames = np.min([len(projected_w_steps), num_frames])
        pick_fewer_indices = np.linspace(
            0, len(projected_w_steps) - 1, num_of_frames
        ).astype(int)

        for projected_w in projected_w_steps[pick_fewer_indices]:
            synth_image = generator.synthesis(
                projected_w.unsqueeze(0), noise_mode="const"
            )
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = (
                synth_image.permute(0, 2, 3, 1)
                .clamp(0, 255)
                .to(torch.uint8)[0]
                .cpu()
                .numpy()
            )
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f"{outdir}/target_{seed}.png")
    projected_w = projected_w_steps[-1]

    synth_image = latent_to_image(generator.synthesis, projected_w, noise_mode="const")
    PIL.Image.fromarray(synth_image, "RGB").save(f"{outdir}/proj_{seed}.png")

    for boost in [3, 5, 10, 20, 30, 40, 50, 90, 100, 120, 130, 150, 160, 200, 300, 400]:
        projected_w_mouth = projected_w + boost * mouth

        synth_image_mouth = latent_to_image(
            generator.synthesis, projected_w_mouth, noise_mode="const"
        )

        PIL.Image.fromarray(synth_image_mouth, "RGB").save(
            f"{outdir}/proj_mouth({boost})_{seed}.png"
        )

    np.savez(
        f"{outdir}/projected_w_{seed}.npy", w=projected_w.unsqueeze(0).cpu().numpy()
    )


if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter
