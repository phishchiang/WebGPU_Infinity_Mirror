import fxaaWGSL from '../shaders/fxaa.wgsl?raw';
import { PostProcessEffect } from './PostProcessEffect';

export class FXAAEffect implements PostProcessEffect {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private resolutionBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;

  constructor(device: GPUDevice, format: GPUTextureFormat, sampler: GPUSampler, size: [number, number]) {
    this.device = device;
    this.sampler = sampler;

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: fxaaWGSL }),
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: this.device.createShaderModule({ code: fxaaWGSL }),
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      // No depthStencil for post-process
    });

    this.resolutionBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.resolutionBuffer, 0, new Float32Array(size));
  }

  apply(
    commandEncoder: GPUCommandEncoder,
    input: { [key: string]: GPUTextureView },
    outputView: GPUTextureView,
    size: [number, number]
  ): void {
    const inputView = input.A; // Use the 'A' key for the main input
    
    // Update resolution buffer if size changes
    this.device.queue.writeBuffer(this.resolutionBuffer, 0, new Float32Array(size));

    // Recreate bind group if inputView changes (optional)
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputView },
        { binding: 2, resource: { buffer: this.resolutionBuffer } },
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
}