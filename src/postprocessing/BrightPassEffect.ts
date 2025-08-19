import brightPassWGSL from '../shaders/brightPass.wgsl?raw';
import { PostProcessEffect } from './PostProcessEffect';

export class BrightPassEffect implements PostProcessEffect {
private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private thresholdBuffer: GPUBuffer;
  private kneeBuffer: GPUBuffer; 

  constructor(device: GPUDevice, format: GPUTextureFormat, sampler: GPUSampler, threshold: number, knee: number ) {
    this.device = device;
    this.sampler = sampler;

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, 
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: brightPassWGSL }),
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: this.device.createShaderModule({ code: brightPassWGSL }),
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      // No depthStencil for post-process
    });

    this.thresholdBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.thresholdBuffer, 0, new Float32Array([threshold]));

    this.kneeBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.kneeBuffer, 0, new Float32Array([knee]));

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: device.createShaderModule({ code: brightPassWGSL }),
        entryPoint: 'vs_main',
      },
      fragment: {
        module: device.createShaderModule({ code: brightPassWGSL }),
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  apply(
    commandEncoder: GPUCommandEncoder,
    input: { [key: string]: GPUTextureView },
    outputView: GPUTextureView,
    _size: [number, number]
  ): void {
    const inputView = input.A; // Use the 'A' key for the main input
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputView },
        { binding: 2, resource: { buffer: this.thresholdBuffer } },
        { binding: 3, resource: { buffer: this.kneeBuffer } },
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

  setThreshold(threshold: number): void {
    this.device.queue.writeBuffer(this.thresholdBuffer, 0, new Float32Array([threshold]));
  }

  setKnee(knee: number): void {
    // Update the knee value in the shader
    this.device.queue.writeBuffer(this.kneeBuffer, 0, new Float32Array([knee]));
  }
}