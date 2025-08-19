import { RenderTarget } from '../RenderTarget';
import { BrightPassEffect } from './BrightPassEffect';
import { BlurEffect } from './GaussianBlurEffect';
import { GlowAddEffect } from './GlowAddEffect';
import { PassThroughEffect } from './PassThroughEffect';
// import { UpsampleTentEffect } from './UpsampleTentEffect';

export class UnrealGlowEffect {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private sampler: GPUSampler;
  private levels: number;
  private width: number;
  private height: number;
  private renderTargets: { ping: RenderTarget, pong: RenderTarget, size: [number, number] }[] = [];
  private brightPass: BrightPassEffect;
  private blurH: BlurEffect;
  private blurV: BlurEffect;
  private add: GlowAddEffect;
  private passThrough: PassThroughEffect;

  constructor(
    device: GPUDevice,
    format: GPUTextureFormat,
    sampler: GPUSampler,
    width: number,
    height: number,
    levels: number,
    brightPass: BrightPassEffect,
    blurH: BlurEffect,
    blurV: BlurEffect,
    add: GlowAddEffect,
    passThrough: PassThroughEffect
  ) {
    this.device = device;
    this.format = format;
    this.sampler = sampler;
    this.levels = levels;
    this.width = width;
    this.height = height;
    this.brightPass = brightPass;
    this.blurH = blurH;
    this.blurV = blurV;
    this.add = add;
    this.passThrough = passThrough;
    this.initRenderTargets();
  }

  private initRenderTargets() {
    let w = this.width, h = this.height;
    for (let i = 0; i < this.levels; ++i) {
      w = Math.max(1, Math.floor(w / 2));
      h = Math.max(1, Math.floor(h / 2));
      this.renderTargets.push({
        ping: new RenderTarget(this.device, w, h, this.format),
        pong: new RenderTarget(this.device, w, h, this.format),
        size: [w, h],
      });
    }
  }

  public apply(
    commandEncoder: GPUCommandEncoder,
    inputView: GPUTextureView,
    outputView: GPUTextureView
  ) {
    // 1. Bright pass at full res
    this.brightPass.apply(
      commandEncoder,
      { A: inputView },
      this.renderTargets[0].ping.view,
      this.renderTargets[0].size
    );

    // 2. Downsample chain
    for (let i = 1; i < this.levels; ++i) {
      this.passThrough.apply(
        commandEncoder,
        { A: this.renderTargets[i - 1].ping.view },
        this.renderTargets[i].ping.view,
        this.renderTargets[i].size
      );
    }

    // 3. Blur each level (ping-pong)
    for (let i = 0; i < this.levels; ++i) {
      this.blurH.apply(
        commandEncoder,
        { A: this.renderTargets[i].ping.view },
        this.renderTargets[i].pong.view,
        this.renderTargets[i].size
      );
      this.blurV.apply(
        commandEncoder,
        { A: this.renderTargets[i].pong.view },
        this.renderTargets[i].ping.view,
        this.renderTargets[i].size
      );
    }

    // 4. Upsample and combine with extra blur after add
    let prevView = this.renderTargets[this.levels - 1].ping.view;
    for (let i = this.levels - 2; i >= 0; --i) {
      // Add upsampled result to current level's blurred image, output to pong
      this.add.apply(
        commandEncoder,
        { A: this.renderTargets[i].ping.view, B: prevView },
        this.renderTargets[i].pong.view,
        this.renderTargets[i].size
      );
      // Extra blur (horizontal)
      this.blurH.apply(
        commandEncoder,
        { A: this.renderTargets[i].pong.view },
        this.renderTargets[i].ping.view,
        this.renderTargets[i].size
      );
      // Extra blur (vertical)
      this.blurV.apply(
        commandEncoder,
        { A: this.renderTargets[i].ping.view },
        this.renderTargets[i].pong.view,
        this.renderTargets[i].size
      );
      // For next iteration, pong holds the combined result
      prevView = this.renderTargets[i].pong.view;
    }

    // 5. Composite with original scene
    this.add.apply(
      commandEncoder,
      { A: inputView, B: prevView },
      outputView,
      [this.width, this.height]
    );
  }
}