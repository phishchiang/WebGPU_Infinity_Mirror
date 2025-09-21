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
}

struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) frag_normal : vec3f,
  @location(2) frag_uv : vec2f,
}

// Hash function for pseudo-random gradients
fn hash3(p: vec3f) -> f32 {
  let p3 = fract(p * 0.1031);
  let p4 = dot(p3, p3.yzx + 19.19);
  return fract((p4 + 19.19) * p4);
}

// Simple value noise in 3D
fn valueNoise3D(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);

  // Trilinear interpolation of 8 corners
  let c000 = hash3(i + vec3f(0.0, 0.0, 0.0));
  let c100 = hash3(i + vec3f(1.0, 0.0, 0.0));
  let c010 = hash3(i + vec3f(0.0, 1.0, 0.0));
  let c110 = hash3(i + vec3f(1.0, 1.0, 0.0));
  let c001 = hash3(i + vec3f(0.0, 0.0, 1.0));
  let c101 = hash3(i + vec3f(1.0, 0.0, 1.0));
  let c011 = hash3(i + vec3f(0.0, 1.0, 1.0));
  let c111 = hash3(i + vec3f(1.0, 1.0, 1.0));

  let u = f * f * (3.0 - 2.0 * f); // Smoothstep

  let x00 = mix(c000, c100, u.x);
  let x10 = mix(c010, c110, u.x);
  let x01 = mix(c001, c101, u.x);
  let x11 = mix(c011, c111, u.x);

  let y0 = mix(x00, x10, u.y);
  let y1 = mix(x01, x11, u.y);

  return mix(y0, y1, u.z);
}

// Add a Direction Array for Each Octave
const octaveDirs = array<vec3f, 5>(
  vec3f(1.0, 0.5, 0.0),
  vec3f(-0.7, 1.0, 0.2),
  vec3f(0.3, -0.6, 1.0),
  vec3f(-1.0, 0.2, -0.5),
  vec3f(0.5, -1.0, 0.7)
);

// fBM (fractional Brownian motion)
fn fbm(p: vec3f) -> f32 {
  var value = 0.0;
  var amplitude = 0.5;
  var frequency = 10.0;
  for (var i = 0; i < 5; i = i + 1) {
    // Animate each octave in a different direction and speed
    let timeOffset = uTime * (0.2 + 0.15 * f32(i));
    let animatedP = p * frequency + octaveDirs[i] * timeOffset;
    value = value + amplitude * valueNoise3D(animatedP);
    frequency = frequency * 2.0;
    amplitude = amplitude * 0.5;
  }
  return value;
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
  let translateYMatrix = mat4x4<f32>(
    1.0, 0.0, 0.0, 0.0,  // Scale X by 1.0
    0.0, 1.0, 0.0, 0.0,  // Scale Y by 1.0
    0.0, 0.0, 1.0, 0.0,  // Scale Z by 1.0
    0.0, 0.0, 0.0, 1.0   // Translation along Y-axis
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
  var flipped_uv = input.frag_uv;
  flipped_uv.x = 1.0 - flipped_uv.x; // flip horizontally
  var finalColor: vec4f = textureSample(myTexture, mySampler, flipped_uv);

  // Animate along Z for turbulence
  let noisePos = vec3f(input.frag_uv.x, input.frag_uv.y, 1.0) + vec3f(0.0, uTime * 0.05, uTime * 0.016);
  var noiseValue = fbm(noisePos);

  // apply power to noise value
  noiseValue = pow(noiseValue, 4.0); // Uncomment to apply power to the noise value

  var maskColor: vec4f = textureSample(myMaskTexture, mySampler, flipped_uv);
  maskColor = maskColor * noiseValue;
  finalColor -= maskColor.r;
  return finalColor;
}