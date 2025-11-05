@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var myTexture: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_depth_2d;


struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var pos = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
  );
  var uv = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0)
  );
  var output: VertexOutput;
  // because hardcoded array; so we have to use vertexIndex for those arrays
  // otherwise for input.position, GPU handles this automatically
  output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  output.uv = uv[vertexIndex];
  return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  // return textureSample(myTexture, mySampler, uv);

  let dims = textureDimensions(depthTex);
  let uvClamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let coord = vec2<i32>(clamp(vec2<f32>(dims) * uvClamped, vec2<f32>(0.0), vec2<f32>(vec2<f32>(dims) - 1.0)));
  let rawDepth: f32 = textureLoad(depthTex, coord, 0);
  // return vec4<f32>(depth, depth, depth, 1.0);


  var black = 0.95;
  var white = 1.0;
  var gamma = 1.0;
  var invert = 1.0;

  // Remap to [0..1] using black/white, optional gamma and invert
  let eps = 1e-6;
  let denom = max(white - black, eps);
  var t = clamp((rawDepth - black) / denom, 0.0, 1.0);
  let g = max(gamma, eps);
  t = pow(t, 1.0 / g);
  if (invert > 0.5) {
    t = 1.0 - t;
  }

  // Compp
  // use math exponential to increase contrast
  // var contrast_T = exp(3.0 * (t - 1.0));
  var comp = textureSample(myTexture, mySampler, uv) * t;

  // return vec4<f32>(t, t, t, 1.0);
  return comp;
}