@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var<uniform> intensity: f32;

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
  output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  output.uv = uv[vertexIndex];
  return output;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let colorA = textureSample(texA, mySampler, uv);
  let colorB = textureSample(texB, mySampler, uv);

  // Optional: Clamp bloom to avoid over-bright
  let bloom = min(colorB * intensity, vec4<f32>(1.0));

  // Optional: Non-linear blend (screen blend)
  var result = 1.0 - (1.0 - colorA) * (1.0 - bloom);

  // // Or: Additive with soft clamp
  // var result = colorA + bloom;
  // result = min(result, vec4<f32>(1.0));

  // // Or: Additive with tonemapping (ACES or Reinhard)
  // var result = colorA + bloom;
  // result = result / (result + vec4<f32>(1.0));

  // Return
  return result;
  // return colorA + colorB * intensity;
}