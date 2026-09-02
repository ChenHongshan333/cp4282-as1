# CP4282 Gaussian Splatting — Assignment 1 (Rendering)

Starter code and data for the CP4282 Gaussian Splatting **rendering** assignment.

This repository intentionally contains incomplete teaching implementations. The missing sections
are the work: read the corresponding unit in the course notes, implement the marked functions,
and use the supplied checks before moving on.

## Setup

```bash
git clone https://github.com/weitsang/cp4282-as1.git
cd cp4282-as1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository contains a small Lego dataset under `data/lego/`:

- `init.ply` is a small point cloud you can render.
- `transforms_train.json` / `transforms_test.json` hold the camera poses.

The renderers take a PLY on the command line, so any `point_cloud.ply` works:

```bash
python 3dgs_renderer_v1.py data/lego/init.ply render.png
```

## The assignment

1. `3dgs_renderer_v1.py`: calculate the RGB of a pixel on the CPU
2. `3dgs_renderer_v2.py`: calculate the RGB of a pixel with one Warp work item per pixel
3. `3dgs_renderer_v3.py`: calculate the RGB from Gaussian-first tile records

For Version 3, the tile-list builder (`shared/tile_builder.py`, `GaussianFirstTileBuilder`) is
supplied. Implement the `rasterize_tiles` stage using the ordered records for each pixel's tile.

Annotated walkthroughs explain the provided skeleton code without showing the TODO solution:

- `3dgs_renderer_v1_annotated.md`
- `3dgs_renderer_v2_annotated.md`
- `3dgs_renderer_v3_annotated.md`

Read them in version order.

## Running checks

```bash
python -m compileall scripts shared 3dgs_renderer_v1.py 3dgs_renderer_v2.py 3dgs_renderer_v3.py gaussian_first_tile_workspace_gpu.py
python scripts/check_setup.py
```

Use `--help` on each renderer for its command-line arguments. Start with a low resolution while
debugging.

## Running test cases provided from course website unit 4
### v1
```bash
python .\3dgs_renderer_v1.py .\5splats.ply .\five.png `
  --width 400 --height 400 --focal-length 350 `
  --camera-position 0 0 -2 --look-at 0 0 0 --up 0 -1 0
```
