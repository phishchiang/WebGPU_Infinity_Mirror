import glowAddWGSL from '../shaders/glowAdd.wgsl?raw';
import { PostProcessEffect } from './PostProcessEffect';

export class GlowAddEffect implements PostProcessEffect {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private bindGroupLayout: GPUBindGroupLayout;
  private intensityBuffer: GPUBuffer;

  constructor(device: GPUDevice, format: GPUTextureFormat, sampler: GPUSampler, intensity: number ) {
    this.device = device;
    this.sampler = sampler;

    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, // texA
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }, // texB
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: glowAddWGSL }),
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: this.device.createShaderModule({ code: glowAddWGSL }),
        entryPoint: 'fs_main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      // No depthStencil for post-process
    });

    this.intensityBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.intensityBuffer, 0, new Float32Array([intensity]));
  }

  apply(
    commandEncoder: GPUCommandEncoder,
    input: { [key: string]: GPUTextureView },
    outputView: GPUTextureView,
    _size: [number, number]
  ): void {
    const inputA = input.A; // Use the 'A' key for the first input
    const inputB = input.B; // Use the 'B' key for the second input

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: inputA },
        { binding: 2, resource: inputB },
        { binding: 3, resource: { buffer: this.intensityBuffer } },
      ],
    });

    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: outputView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: [0, 0, 0, 1],
        },
      ],
    };

    const pass = commandEncoder.beginRenderPass(passDesc);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();
  }

  setIntensity(intensity: number): void {
    this.device.queue.writeBuffer(this.intensityBuffer, 0, new Float32Array([intensity]));
  }
}