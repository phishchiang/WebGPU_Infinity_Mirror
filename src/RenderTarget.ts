export class RenderTarget {
  public texture: GPUTexture;
  public view: GPUTextureView;

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    format: GPUTextureFormat = 'rgba8unorm'
  ) {
    this.texture = device.createTexture({
      size: [width, height, 1],
      format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.view = this.texture.createView();
  }

  resize(device: GPUDevice, width: number, height: number, format: GPUTextureFormat = 'rgba8unorm') {
    this.texture.destroy();
    this.texture = device.createTexture({
      size: [width, height, 1],
      format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });
    this.view = this.texture.createView();
  }
}