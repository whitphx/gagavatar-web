"""gagavatar-web: export a GAGAvatar avatar and its upsampler for the browser.

    gagavatar-web bundle --assets DIR --avatar ID --out avatar.gsb
    gagavatar-web refiner --assets DIR --out refiner [--precision float16]

DIR is a GAGAvatar asset directory (GAGAvatar.pt, tracked.pt), as
``gagavatar-assets`` sets it up. --flame points at the FLAME model when it is
not in the default place.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _runtime(args):
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig

    return GAGAvatarRuntime(GAGAvatarRuntimeConfig(
        asset_dir=args.assets, tracked_path=args.tracked, flame_model_path=args.flame, device=args.device,
    ))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gagavatar-web", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("bundle", "refiner"):
        p = sub.add_parser(name)
        p.add_argument("--assets", required=True, help="GAGAvatar asset directory")
        p.add_argument("--flame", help="FLAME model file, if not in the asset directory's default place")
        p.add_argument("--tracked", help="tracked avatars file, if not <assets>/tracked.pt")
        p.add_argument("--device", default="auto")
        p.add_argument("--out", required=True)
    b = sub.choices["bundle"]
    b.add_argument("--avatar", required=True, help="avatar id in the tracked avatars file")
    b.add_argument("--prune", type=float, default=0.01, help="leave out off-head Gaussians less opaque than this")
    b.add_argument("--projection", help=".npz with mean (C,) and basis (C, K) to project the features to K channels")
    r = sub.choices["refiner"]
    r.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args(argv)

    runtime = _runtime(args)
    if args.command == "bundle":
        from .export import export_bundle

        runtime.set_avatar_id(args.avatar)
        projection = None
        if args.projection:
            with np.load(args.projection) as f:
                projection = (f["mean"], f["basis"])
        meta = export_bundle(runtime, args.out, prune=args.prune, projection=projection)
        print(json.dumps({k: meta[k] for k in ("count", "head_count", "feature_channels")}))
    else:
        from .refiner import export_refiner

        spec = export_refiner(runtime.model.upsampler, args.out, precision=args.precision)
        print(json.dumps({k: spec[k] for k in ("dataType", "input", "output")}))


if __name__ == "__main__":
    main()
