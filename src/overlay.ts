export interface OverlayOptions {
  /** Drawn at its own size, with its alpha. */
  image: ImageBitmap | HTMLCanvasElement | OffscreenCanvas | HTMLImageElement;
  width: number;
  height: number;
  /** Top-left corner in the target, in pixels. */
  x: number;
  y: number;
  targetFormat: GPUTextureFormat;
}

export interface Overlay {
  /** Records a render pass that draws the image over what `target` already holds. */
  encode(encoder: GPUCommandEncoder, target: GPUTexture): void;
  destroy(): void;
}

/**
 * Draws an image, such as a label saying the video is AI-generated, over a
 * presented frame. It goes on after the frame is presented, so a sharpening
 * presenter never touches it.
 */
export function createOverlay(device: GPUDevice, options: OverlayOptions): Overlay {
  const { image, width, height, x, y, targetFormat } = options;
  const texture = device.createTexture({
    size: [width, height], format: 'rgba8unorm', label: 'overlay',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  // Premultiplied, which the blend below expects.
  device.queue.copyExternalImageToTexture({ source: image }, { texture, premultipliedAlpha: true }, [width, height]);
  const module = device.createShaderModule({
    label: 'overlay',
    code: /* wgsl */ `
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(image, vec2i(pos.xy) - vec2i(${x}, ${y}), 0);
}`,
  });
  const blend: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' };
  const pipeline = device.createRenderPipeline({
    label: 'overlay', layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format: targetFormat, blend: { color: blend, alpha: blend } }] },
  });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: texture.createView() }] });
  return {
    encode(encoder, target) {
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'load', storeOp: 'store' }] });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      // The viewport limits the full-screen triangle to the image's rectangle.
      pass.setViewport(x, y, width, height, 0, 1);
      pass.draw(3);
      pass.end();
    },
    destroy() { texture.destroy(); },
  };
}
