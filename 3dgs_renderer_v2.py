"""Warp port of the Unit 5 parallel 3DGS raster stage.

Usage:
    python 3dgs_renderer_v2.py point_cloud.ply render.png --device cpu

Projection and global near-to-far ordering deliberately reuse the sequential reference. Warp
owns persistent screen-space arrays and launches one parallel work item per output pixel.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
from PIL import Image
import warp as wp # concurrently run CPU/GPU

# Reuse the functions provided by v1
_reference = importlib.import_module("3dgs_renderer_v1")
Camera = _reference.Camera
GaussianSet = _reference.GaussianSet
project_gaussians = _reference.project_gaussians
SUPPORT_RADIUS_SQUARED = _reference.SUPPORT_RADIUS_SQUARED
compact_support = _reference.compact_support
ALPHA_CUTOFF = _reference.ALPHA_CUTOFF
TRANSMITTANCE_CUTOFF = 1.0e-4


@wp.kernel # indicate that this is a compuation function that can be concurrently ran by Wrap
def rasterize(
    centres: wp.array(dtype=wp.vec2), # the center (x, y) for each Gaussian
    conics: wp.array(dtype=wp.vec3), # the shape of each Gaussian
    colours: wp.array(dtype=wp.vec3), # the color of each Gaussian
    opacities: wp.array(dtype=wp.float32), # the opacity of each Gaussian
    supports: wp.array(dtype=wp.float32), # the effective range of each Gaussian
    count: int, # the number of visible Gaussians
    width: int, # width of image
    background: wp.vec3, # background color, set to white by default
    image: wp.array(dtype=wp.vec3), # the image that kernel will write to
):
    # make sure which pixel should be handled by which task
    pixel = wp.tid()
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5

    # use wp instead of numpy in kernels
    pixel_color = wp.vec3(0.0, 0.0, 0.0)
    transmittance = 1.0

    for i in range(count):
        dx = px - centres[i][0]
        dy = py - centres[i][1]

        a = conics[i][0]
        b = conics[i][1]
        c = conics[i][2]

        mahalanobis_squared = (
            a * dx * dx
            + 2.0 * b * dx * dy
            + c * dy * dy
        )

        if mahalanobis_squared > supports[i]:
            continue

        alpha = opacities[i] * wp.exp(
            -0.5 * mahalanobis_squared
        )
        alpha = wp.min(alpha, 0.99)

        if alpha < ALPHA_CUTOFF:
            continue

        pixel_color = (
            pixel_color
            + transmittance * alpha * colours[i]
        )

        transmittance = transmittance * (1.0 - alpha)

        if transmittance < TRANSMITTANCE_CUTOFF:
            break

    image[pixel] = pixel_color + transmittance * background

# manage Wrap memory
# it does 2 things:
# 1. apply for memory in CPU/GPU in advance
# 2. put data into the memory and then start kernel
class WarpRenderer:
    """Persistent Warp storage for the screen-space records and rendered pixels."""

    def __init__(self, width: int, height: int, maximum_splats: int, device: str):
        self.width, self.height, self.maximum_splats = width, height, maximum_splats
        self.device = wp.get_device(device)
        self.centres = wp.zeros(maximum_splats, dtype=wp.vec2, device=self.device)
        self.conics = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.colours = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.opacities = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.supports = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.image = wp.zeros(width * height, dtype=wp.vec3, device=self.device)

    # render
    def render(self, splats: GaussianSet, camera: Camera,
               background: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        # project 3D Gaussian to 2D Gaussian
        projected = project_gaussians(splats, camera)
        count = len(projected.opacities)
        if count > self.maximum_splats:
            raise ValueError(f"Renderer capacity {self.maximum_splats:,} is below {count:,} visible splats.")

        # copy data to Wrap array
        self.centres.assign(wp.array(projected.centres, dtype=wp.vec2, device=self.device))
        self.conics.assign(wp.array(projected.conics, dtype=wp.vec3, device=self.device))
        self.colours.assign(wp.array(projected.colors, dtype=wp.vec3, device=self.device))
        self.opacities.assign(wp.array(projected.opacities, dtype=wp.float32, device=self.device))

        # calculate the effective range for each Gaussian
        supports = compact_support(projected.opacities)
        self.supports.assign(wp.array(supports, dtype=wp.float32, device=self.device))

        # launch the concurrent pipeline
        wp.launch(rasterize, dim=self.width * self.height,
                  inputs=[self.centres, self.conics, self.colours, self.opacities, self.supports,
                          count, self.width, wp.vec3(*background), self.image],
                  device=self.device)
        return self.image.numpy().reshape(self.height, self.width, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--focal-length", type=float, default=350.0)
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0")
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--look-at", nargs=3, type=float, default=(0.0, 0.0, 1.0))
    parser.add_argument("--up", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    args = parser.parse_args()

    wp.init()
    camera = Camera.from_look_at(
        args.width, args.height, args.focal_length,
        args.camera_position, args.look_at, args.up,
    )
    splats = GaussianSet.from_ply(args.ply)
    renderer = WarpRenderer(args.width, args.height, len(splats.means), args.device)
    image = renderer.render(splats, camera)
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0)).save(args.output)


if __name__ == "__main__":
    main()
