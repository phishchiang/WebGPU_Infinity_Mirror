@group(0) @binding(0) var<uniform> viewMatrix : mat4x4<f32>;
@group(0) @binding(1) var<uniform> projectionMatrix : mat4x4<f32>;
@group(0) @binding(2) var<uniform> canvasSize : vec2<f32>;
@group(0) @binding(3) var<uniform> uTime : f32;
@group(0) @binding(4) var<uniform> modelMatrix : mat4x4<f32>;
@group(0) @binding(5) var<uniform> uTestValue : f32;
@group(0) @binding(6) var<uniform> uTestValue_02 : f32;
@group(0) @binding(7) var mySampler: sampler;
@group(0) @binding(8) var myTexture: texture_2d<f32>;
@group(0) @binding(9) var myMaskTexture: texture_2d<f32>;

struct VertexInput {
  @location(0) position : vec3f,
  @location(1) normal : vec3f,
  @location(3) uv : vec2f,
  @location(4) uv2 : vec2f, 
}

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec4f,
  @location(2) frag_uv : vec2f,
  @location(3) frag_uv2 : vec2f,
}


@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {

  let vWorldSpaceNormal = modelMatrix * vec4(input.normal, 0.0);

  return VertexOutput(
    projectionMatrix * viewMatrix * modelMatrix * vec4f(input.position, 1.0), 
    vWorldSpaceNormal,
    input.uv,
    input.uv2,
  );
}

struct FragmentInput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec4f,
  @location(2) frag_uv : vec2f,
  @location(3) frag_uv2 : vec2f, 
}

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4f {
  // float color = dot(vNormal, vec3(1.0));

  var basicNormal = normalize(vec3f(input.frag_normal.xyz));
  var basicNormalLT = dot(basicNormal.xyz, normalize(vec3f(1.0, 1.0, 1.0))); 
  basicNormalLT = clamp(basicNormalLT, 0.35, 1.0);
  var basicColor = vec3f(0.322, 0.659, 0.78) * basicNormalLT;

  return vec4f(basicColor, 1.0);
}