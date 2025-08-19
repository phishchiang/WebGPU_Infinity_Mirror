@group(0) @binding(0) var<uniform> viewMatrix : mat4x4<f32>;
@group(0) @binding(1) var<uniform> projectionMatrix : mat4x4<f32>;
@group(0) @binding(2) var<uniform> canvasSize : vec2<f32>;
@group(0) @binding(3) var<uniform> uTime : f32;
@group(0) @binding(4) var<uniform> modelMatrix : mat4x4<f32>;
@group(0) @binding(5) var<uniform> uTestValue : f32;
@group(0) @binding(6) var<uniform> uTestValue_02 : f32;
@group(0) @binding(7) var mySampler: sampler;
@group(0) @binding(8) var myTexture: texture_2d<f32>;

struct VertexInput {
  @location(0) position : vec3f,
  @location(1) normal : vec3f,
  @location(3) uv : vec2f,
}

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec3f,
  @location(2) frag_uv : vec2f,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
  let translateYMatrix = mat4x4<f32>(
    1.0, 0.0, 0.0, 0.0,  // Scale X by 1.0
    0.0, 1.0, 0.0, 0.0,  // Scale Y by 1.0
    0.0, 0.0, 1.0, 0.0,  // Scale Z by 1.0
    0.0, uTestValue_02, 0.0, 1.0   // Translation along Y-axis
  );

  var transformedModelMatrix = modelMatrix * translateYMatrix;

  return VertexOutput(
    projectionMatrix * viewMatrix * transformedModelMatrix * vec4f(input.position, 1.0), 
    input.normal,
    input.uv,
  );
}

struct FragmentInput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec3f,
  @location(2) frag_uv : vec2f,
}

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4f {
  var finalColor: vec4f = textureSample(myTexture, mySampler, input.frag_uv);
  finalColor *= uTestValue;
  return finalColor;
}