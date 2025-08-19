import blurWGSL from '../shaders/gaussianBlur.wgsl?raw';
import { PostProcessEffect } from './PostProcessEffect';

export class BlurEffect implements PostProcessEffect {
private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private directionBuffer: GPUBuffer;
  private texelSizeBuffer: GPUBuffer;
  private radiusBuffer: GPUBuffer;

  constructor(device: GPUDevice, format: GPUTextureFormat, sampler: GPUSampler, direction: [number, number], texelSize: [number, number], radius: number) {
    // Ensure direction and texelSize are normalized
    this.device = device;
    this.sampler = sampler;

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: blurWGSL }),
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: this.device.createShaderModule({ code: blurWGSL }),
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      // No depthStencil for post-process
    });

    this.radiusBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.radiusBuffer, 0, new Float32Array([radius]));

    this.directionBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.directionBuffer, 0, new Float32Array(direction));

    this.texelSizeBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.texelSizeBuffer, 0, new Float32Array(texelSize));
  }

  apply(
    commandEncoder: GPUCommandEncoder,
    input: { [key: string]: GPUTextureView },
    outputView: GPUTextureView,
    size: [number, number]
  ): void {
    const inputView = input.A; // Use the 'A' key for the main input
    // Update texel size if needed
    this.device.queue.writeBuffer(this.texelSizeBuffer, 0, new Float32Array([1 / size[0], 1 / size[1]]));

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputView },
        { binding: 2, resource: { buffer: this.directionBuffer } },
        { binding: 3, resource: { buffer: this.texelSizeBuffer } },
        { binding: 4, resource: { buffer: this.radiusBuffer } },
      ],
    });

    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: outputView,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const pass = commandEncoder.beginRenderPass(passDesc);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  setRadius(radius: number): void {
    this.device.queue.writeBuffer(this.radiusBuffer, 0, new Float32Array([radius]));
  }
}