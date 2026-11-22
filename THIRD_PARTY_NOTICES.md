# Third-party notices

The packages contain no third-party code beyond what is listed here. They work with the outputs of the projects below, and some of their conventions follow those projects.

## GAGAvatar (MIT)

`@whitphx/gagavatar-web` and the `gagavatar-web` exporter follow GAGAvatar's conventions: its camera (a 3x4 world-to-view transform in PyTorch3D's axes, focal length 12 in units of half the image), its zero-pose camera rotation, the layout of its Gaussians (the FLAME-bound head Gaussians first, then the image-plane ones), and its upsampler's input. The exporter calls GAGAvatar's `gagavatar.runtime` and exports its upsampler; it does not copy its code.

- https://github.com/xg-chu/GAGAvatar/tree/8aa3ddfcd4416aa2434a0ef4c533fd2a6570bbb6

```
MIT License

Copyright (c) 2024 Xuangeng Chu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ONNX Runtime Web (MIT)

`patchOrtGraphCapture` in `@whitphx/gagavatar-web` holds, for each supported version, the one condition of `run()` in ONNX Runtime Web's minified `ort.webgpu.bundle.min.mjs` that it rewrites, and its rewritten form. ONNX Runtime Web is copyright Microsoft Corporation and licensed under the MIT License, as the banner of that file states.

- https://www.npmjs.com/package/onnxruntime-web/v/1.31.1
