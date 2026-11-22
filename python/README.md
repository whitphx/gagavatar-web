# gagavatar-web

Exports a [GAGAvatar](https://github.com/xg-chu/GAGAvatar) avatar and GAGAvatar's upsampler for the browser renderer [`@whitphx/gagavatar-web`](https://github.com/whitphx/gagavatar-web).

It runs where GAGAvatar runs, with GAGAvatar's weights, the FLAME model and avatars you tracked yourself. Install GAGAvatar (the `gagavatar` package, with its `gagavatar.runtime` module) into that environment first, following its README, then this package:

```bash
pip install gagavatar-web
```

## Usage

```bash
# One avatar, from the tracked avatars file in the asset directory.
gagavatar-web bundle --assets path/to/GAGAvatar-assets --avatar my_photo.jpg --out avatar.gsb

# GAGAvatar's upsampler as ONNX, with the spec the browser loads it with.
gagavatar-web refiner --assets path/to/GAGAvatar-assets --out refiner --precision float16
```

The second writes `refiner.onnx` and `refiner.json`. A float16 graph needs a GPU with WebGPU's `shader-f16` feature; export `--precision float32` for one without.

From Python, `export_bundle(runtime, out)` and `export_refiner(runtime.model.upsampler, out)` do the same with a `GAGAvatarRuntime` you set up, and `read_bundle` and `write_bundle` read and write bundles with numpy only.

## What a bundle contains, and its licence

A bundle holds GAGAvatar's Gaussians for one avatar at its neutral head, the neutral head vertices, and the neck parameters the browser turns the head with. All of them are computed from the FLAME model you supply, and the features from your photo. A bundle therefore comes under the [FLAME licence](https://flame.is.tue.mpg.de/modellicense.html) as well as under whatever applies to the photo, and is not for redistribution unless those allow it. GAGAvatar's released weights state no licence of their own.
