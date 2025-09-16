import { PostProcessEffect } from './PostProcessEffect';
import passThroughWGSL from '../shaders/depthComp.wgsl?raw';

export class DepthCompEffect implements PostProcessEffect {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private depthView: GPUTextureView | null = null;
  private depthCmpSampler: GPUSampler;

  constructor(device: GPUDevice, format: GPUTextureFormat, sampler: GPUSampler) {
    this.device = device;
    this.sampler = sampler;
    this.depthCmpSampler = device.createSampler({ compare: 'less' });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: passThroughWGSL }),
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: this.device.createShaderModule({ code: passThroughWGSL }),
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      // No depthStencil for post-process
    });
  }

  public setDepthTexture(view: GPUTextureView) {
    this.depthView = view;
  }

  apply(
    commandEncoder: GPUCommandEncoder,
    input: { [key: string]: GPUTextureView },
    outputView: GPUTextureView,
    _size: [number, number]
  ): void {
    const inputView = input.A; // Use the 'A' key for the main input
    
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: outputView,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    if (!this.depthView) {
      throw new Error('Depth texture view is not set.');
    }

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputView },
        { binding: 2, resource: this.depthView },
        { binding: 3, resource: this.depthCmpSampler },
      ],
    });

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.draw(6, 1, 0, 0); // Fullscreen quad
    passEncoder.end();
  }
}