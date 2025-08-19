export interface PostProcessEffect {
  /**
   * Apply the effect.
   * @param commandEncoder The GPUCommandEncoder for recording commands.
   * @param inputView The input GPUTextureView (from previous pass or scene render).
   * @param outputView The output GPUTextureView (to write the result).
   * @param size The [width, height] of the textures.
   */
  apply(
    commandEncoder: GPUCommandEncoder,
    input: { [key: string]: GPUTextureView }, // some FX may have multiple inputs
    outputView: GPUTextureView,
    size: [number, number]
  ): void;
}