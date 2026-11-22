import { createGaussianRasterizer, type GaussianRasterizer } from 'webgpu-gaussian-rasterizer';
import { avatarGaussians, type AvatarBundle, type AvatarGaussians } from './bundle.js';
import { rasterizerCamera } from './camera.js';
import { createHeadTurnPass, takeHeadTurnOffCamera } from './head-turn.js';

export interface AvatarRendererOptions {
  bundle: AvatarBundle;
  /** Width and height of the feature image in pixels. */
  size?: number;
  /**
   * Renders into this buffer, typically a refiner's input (`Refiner.input`),
   * instead of one the rasterizer allocates.
   */
  image?: GPUBuffer;
  outputFormat?: 'float16' | 'float32';
  tileTest?: 'ellipse' | 'box';
  /** Most (tile, Gaussian) pairs a frame can hold. */
  pairCapacity?: number;
}

export interface AvatarFrame {
  /** The head vertices, x y z per vertex in the avatar's frame; omit to keep the last ones. */
  vertices?: Float32Array;
  /** GAGAvatar's 3x4 row-major camera for the frame, head pose included. */
  camera: ArrayLike<number>;
  /**
   * Turns the head on the neck by the camera's head pose and keeps the
   * shoulders still, instead of turning the whole bust with the camera.
   * Needs the bundle's neck.
   */
  bodyStill?: boolean;
}

export interface AvatarRenderer {
  readonly gaussians: AvatarGaussians;
  readonly rasterizer: GaussianRasterizer;
  /** The feature image: `channels` planes of size x size. */
  readonly image: GPUBuffer;
  /** Sets the next frame's pose. The writes go through the queue, so they apply from the next submit. */
  setFrame(frame: AvatarFrame): void;
  /** Records the frame: the neck turn, then the rasterizer. */
  encode(encoder: GPUCommandEncoder, options?: { readback?: boolean }): void;
  destroy(): void;
}

/** Splats a GAGAvatar avatar's Gaussians at a frame's head vertices into a feature image on the GPU. */
export async function createAvatarRenderer(device: GPUDevice, options: AvatarRendererOptions): Promise<AvatarRenderer> {
  const { bundle, size = 512, image, outputFormat = 'float16', tileTest = 'ellipse', pairCapacity = 3_000_000 } = options;
  const g = avatarGaussians(bundle);
  const rasterizer = await createGaussianRasterizer(device, {
    capacity: g.count, width: size, height: size, channels: g.channels, outputFormat, tileTest, pairCapacity, image,
  });
  rasterizer.writeOpacities(g.opacities);
  rasterizer.writeFeatures(g.features);

  const mk = (data: Float32Array, label: string) => {
    const b = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  // The head's vertices land here each frame, and the neck turn reads them
  // from here, because it overwrites the rasterizer's own positions.
  const posed = mk(g.positions, 'posed_positions');
  const restCovariances = mk(g.covariances, 'rest_covariances');
  const turnPass = createHeadTurnPass(device, {
    count: g.count, neck: g.neck,
    restHeights: g.positions.filter((_, k) => k % 3 === 1),
    source: { positions: posed, covariances: restCovariances },
    target: { positions: rasterizer.buffers.positions, covariances: rasterizer.buffers.covariances },
  });

  const view = new Float32Array(16), stillCamera = new Float32Array(12), turn = new Float32Array(3);
  return {
    gaussians: g,
    rasterizer,
    image: rasterizer.buffers.image,
    setFrame({ vertices, camera, bodyStill = false }) {
      if (vertices) {
        if (vertices.length !== 3 * g.headCount) throw new Error(`${vertices.length / 3} head vertices for an avatar with ${g.headCount}`);
        device.queue.writeBuffer(posed, 0, vertices);
      }
      if (bodyStill) {
        takeHeadTurnOffCamera(camera, stillCamera, turn);
        turnPass.setTurn(turn);
        rasterizer.setCamera(rasterizerCamera(stillCamera, size, g.tanFov, view));
      } else {
        turnPass.setTurn(null);
        rasterizer.setCamera(rasterizerCamera(camera, size, g.tanFov, view));
      }
    },
    encode(encoder, encodeOptions) {
      turnPass.encode(encoder);
      rasterizer.encode(encoder, encodeOptions);
    },
    destroy() {
      turnPass.destroy();
      posed.destroy();
      restCovariances.destroy();
      rasterizer.destroy();
    },
  };
}
