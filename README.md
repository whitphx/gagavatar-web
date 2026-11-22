# gagavatar-web

Renders [GAGAvatar](https://github.com/xg-chu/GAGAvatar) avatars in the browser on WebGPU. Each frame, the avatar's Gaussians are splatted at that frame's head vertices into a feature image on the GPU, and a refiner network on ONNX Runtime Web turns the feature image into the picture, with no readback in between. The head can turn on the neck with the shoulders kept still.

The repository holds two packages:

| Package | What it does |
|---|---|
| `@whitphx/gagavatar-web` (npm, the repository root) | Reads avatar bundles, renders them on WebGPU, turns the head on the neck, runs the refiner on ONNX Runtime Web, and paces frames that arrive ahead of time |
| `gagavatar-web` (PyPI, [`python/`](python)) | Exports avatar bundles and GAGAvatar's upsampler from a GAGAvatar installation, and reads and writes bundles with numpy |

The package draws. Where each frame's head vertices come from is up to the caller: a motion model, a recording, or a network stream. The splatting itself is [`webgpu-gaussian-rasterizer`](https://github.com/whitphx/webgpu-gaussian-rasterizer), a separate library for any number of feature channels.

This is not an official release of GAGAvatar. The packages contain code only: no model weights, no avatars, and nothing derived from FLAME. You export avatars and refiners yourself from the upstream releases, under their licences.

## Install

```bash
npm install @whitphx/gagavatar-web onnxruntime-web
```

ONNX Runtime Web is an optional peer dependency, needed for the refiner only. The exporter goes into the Python environment GAGAvatar runs in:

```bash
pip install gagavatar-web
```

## Usage

```ts
import * as ort from 'onnxruntime-web/webgpu';
import { createPresenter } from 'webgpu-gaussian-rasterizer';
import { createAvatarRenderer, createRefiner, parseBundle } from '@whitphx/gagavatar-web';

const bundle = parseBundle(await (await fetch('avatar.gsb')).arrayBuffer());
const refiner = await createRefiner({
  ort,
  model: new Uint8Array(await (await fetch('refiner.onnx')).arrayBuffer()),
  spec: await (await fetch('refiner.json')).json(),
});
const device = refiner.device;
const renderer = await createAvatarRenderer(device, { bundle, image: refiner.input, outputFormat: refiner.spec.dataType });
const presenter = createPresenter(device, {
  source: refiner.output, width: 512, height: 512, sourceFormat: refiner.spec.dataType,
  targetFormat: navigator.gpu.getPreferredCanvasFormat(), sharpen: 0.5,
});

// Per frame:
renderer.setFrame({ vertices, camera, bodyStill: true });
const encoder = device.createCommandEncoder();
renderer.encode(encoder);
device.queue.submit([encoder.finish()]);
await refiner.run();
const present = device.createCommandEncoder();
presenter.encode(present, context.getCurrentTexture());
device.queue.submit([present.finish()]);
```

`vertices` are the frame's head vertices (x, y, z per vertex, in the order of the bundle's `head_ref`) and `camera` is GAGAvatar's 3x4 camera transform for the frame, with the head pose in its rotation. `bodyStill` takes the head pose off the camera and turns the head on the neck instead, which needs the neck parameters the exporter writes into the bundle.

The renderer has to render on ONNX Runtime's device, `refiner.device`, so that the refiner can read its image.

### Avatar bundles and refiners

A bundle (`.gsb`) holds one avatar's Gaussians at its neutral head, the neutral head vertices, and optionally the neck parameters and the avatar's own camera. `src/bundle.ts` documents the format. A bundle names its feature array by its channel count: `feat32` for GAGAvatar's raw features, `feat16` for a 16-channel projection.

GAGAvatar's own upsampler, exported as is, is a 32-channel network at 512 x 512. It is heavy for a browser. A smaller network trained to stand in for it also works, as long as the bundle's features match its input (`--projection` on the exporter).

The refiner checks itself at load: the exporter stores the output for a fixed input, and `refiner.check` reports whether ONNX Runtime Web reproduces it.

### Graph capture

`createRefiner({ graphCapture: true })` captures the network's GPU work and replays it each frame, which saves the per-operator dispatch overhead.

ONNX Runtime Web up to at least 1.31.1 binds a captured session's pre-allocated gpu-buffer outputs on its first run only. Later runs take the path for outputs it allocated itself, and look the GPU buffer up through a handle that was freed after the first run. Whether that throws depends on whether the allocator hands the freed address back. `patchOrtGraphCapture` fixes a copy of `ort.webgpu.bundle.min.mjs`: it makes the output check in `run()` treat every gpu-buffer output of a captured session as pre-allocated. It knows the minified bundles of the versions in `GRAPH_CAPTURE_PATCHED_VERSIONS` and refuses any other file.

```ts
import { patchOrtGraphCapture } from '@whitphx/gagavatar-web';

const source = await (await fetch('/ort/ort.webgpu.bundle.min.mjs')).text();
const ort = await import(URL.createObjectURL(new Blob([patchOrtGraphCapture(source).source], { type: 'text/javascript' })));
ort.env.wasm.wasmPaths = '/ort/';
```

`createRefiner` also probes a captured session at load, with three runs on two inputs, and falls back to running without capture when the probe fails.

### Frame pacing

`createPacer` schedules frames that arrive ahead of time. It renders each into a free slot of a ring (an `ImageBitmap` or a texture per slot), and on each animation frame presents the one whose time has come. It makes no DOM or GPU calls, so it runs in a simulation too.

## Exporting an avatar

The exporter runs where GAGAvatar runs. You need:

- GAGAvatar's code, installed following [its README](https://github.com/xg-chu/GAGAvatar), as the `gagavatar` package with its `gagavatar.runtime` module;
- GAGAvatar's released weights, downloaded from upstream;
- your own copy of FLAME 2020, which you register for and download from the [FLAME website](https://flame.is.tue.mpg.de) under its licence;
- a tracked avatars file of your own photos, made as GAGAvatar's README describes.

Then:

```bash
pip install gagavatar-web

# One avatar, from the tracked avatars file (<assets>/tracked.pt unless --tracked names another).
gagavatar-web bundle --assets path/to/GAGAvatar-assets --avatar my_photo.jpg --out avatar.gsb

# GAGAvatar's upsampler as ONNX, with the spec the browser loads it with.
gagavatar-web refiner --assets path/to/GAGAvatar-assets --out refiner --precision float16
```

The second command writes `refiner.onnx` and `refiner.json`. A float16 graph needs a GPU with WebGPU's `shader-f16` feature; export `--precision float32` for one without. The example page loads both files and a bundle from your disk, and they stay in the browser.

From Python, `export_bundle(runtime, out)` and `export_refiner(runtime.model.upsampler, out)` do the same with a `GAGAvatarRuntime` you set up. `read_bundle` and `write_bundle` read and write bundles with numpy only.

A bundle is computed from the FLAME model you supply and its features from your photo, so it comes under the FLAME licence as well as whatever applies to the photo. It is not for redistribution unless those allow it.

## Models and data

The packages ship code only, and neither needs any of these artefacts to build or to run its tests. The example runs on a procedural avatar.

| Artefact | Used by | Licence and source | In this repository |
|---|---|---|---|
| ONNX Runtime Web | the refiner | MIT ([npm](https://www.npmjs.com/package/onnxruntime-web)) | a peer dependency, never bundled. The graph-capture fix is applied at run time, not shipped as a patched copy |
| GAGAvatar code | the exporter | MIT ([LICENSE](https://github.com/xg-chu/GAGAvatar/blob/main/LICENSE)) | not vendored; you install it |
| GAGAvatar released weights | the exporter | no licence stated on the [model repository](https://huggingface.co/xg-chu/GAGAvatar) | not shipped; you download them from upstream |
| FLAME 2020 | GAGAvatar, at export time | [FLAME licence](https://flame.is.tue.mpg.de/modellicense.html): non-commercial research, no redistribution in whole or in part | not shipped, nor anything computed from it, such as template or neutral vertices, or a bundle's `head_ref` and neck |
| Avatar bundles | the renderer | GAGAvatar outputs for one photo, and neutral FLAME vertices | not shipped; you export your own |
| Refiners trained to stand in for GAGAvatar's upsampler | the refiner | the terms of their training data | not shipped |

## Development

```bash
git clone --recurse-submodules https://github.com/whitphx/gagavatar-web.git
cd gagavatar-web
pnpm install
pnpm --filter webgpu-gaussian-rasterizer build
pnpm build
pnpm test
```

`pnpm dev` serves the example page, which renders a made-up avatar generated in `examples/synthetic/avatar.ts` and can load a bundle and a refiner you exported.

`pnpm test` runs unit tests in Node, and GPU tests through Dawn (the `webgpu` package). The GPU tests skip without a WebGPU adapter unless `REQUIRE_WEBGPU=1` is set. On Linux without a GPU, Mesa's lavapipe (`apt install mesa-vulkan-drivers`) provides one, as in CI. The refiner tests run ONNX Runtime Web in Node on the same adapter. One test has the Python package write a bundle and reads it back, through `uv`; it skips without `uv` unless `REQUIRE_PYTHON=1` is set. The Python package is a uv project: `uv run pytest` in `python/`.

### The rasterizer dependency and the publishing order

`webgpu-gaussian-rasterizer` is not on npm yet, so it is a git submodule in `vendor/`, pinned to a commit of its public repository, and a member of this pnpm workspace. The package depends on it through `workspace:^`, which `pnpm publish` rewrites to the rasterizer's version at the time. The rasterizer is therefore published to npm first. The submodule then moves to the released commit, and `@whitphx/gagavatar-web` is published after it. Once the rasterizer is on npm, the submodule can give way to an ordinary versioned dependency.

The Python package has no dependency on the JavaScript one, and is published on its own.

## Research behind this library

### GAGAvatar: Generalizable and Animatable Gaussian Head Avatar

Xuangeng Chu¹, Tatsuya Harada¹,² · ¹The University of Tokyo, ²RIKEN AIP · NeurIPS 2024

[Project page](https://xg-chu.site/project_gagavatar/) · [arXiv](https://arxiv.org/abs/2410.07971) · [Code](https://github.com/xg-chu/GAGAvatar)

```bibtex
@inproceedings{
    chu2024gagavatar,
    title={Generalizable and Animatable Gaussian Head Avatar},
    author={Xuangeng Chu and Tatsuya Harada},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=gVM2AZ5xA6}
}
```

### ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model

Xuangeng Chu¹, Nabarun Goswami¹, Ziteng Cui¹, Hanqin Wang¹, Tatsuya Harada¹,² · ¹The University of Tokyo, ²RIKEN AIP · SIGGRAPH Asia 2025

[Project page](https://xg-chu.site/project_artalk/) · [arXiv](https://arxiv.org/abs/2502.20323) · [Code](https://github.com/xg-chu/ARTalk)

```bibtex
@misc{
    chu2025artalk,
    title={ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model},
    author={Xuangeng Chu and Nabarun Goswami and Ziteng Cui and Hanqin Wang and Tatsuya Harada},
    year={2025},
    eprint={2502.20323},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2502.20323},
}
```

### FLAME

GAGAvatar builds on [FLAME](https://flame.is.tue.mpg.de), the parametric head model by the Max Planck Institute for Intelligent Systems: T. Li, T. Bolkart, M. J. Black, H. Li and J. Romero, *Learning a model of facial shape and expression from 4D scans*, ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 2017.

FLAME 2020, which GAGAvatar uses, is licensed for non-commercial research only and may not be redistributed. Avatar bundles you export are computed from it and come under its [licence](https://flame.is.tue.mpg.de/modellicense.html).

## Licence

MIT, see `LICENSE`. `THIRD_PARTY_NOTICES.md` credits the projects whose work these packages use.
