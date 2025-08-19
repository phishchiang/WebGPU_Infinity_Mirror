@group(0) @binding(0) var mySampler: sampler;
@group(0) @binding(1) var myTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> resolution: vec2<f32>;

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
  let texel = 1.0 / resolution;
  let rgbNW = textureSample(myTexture, mySampler, uv + texel * vec2(-1.0, -1.0)).rgb;
  let rgbNE = textureSample(myTexture, mySampler, uv + texel * vec2( 1.0, -1.0)).rgb;
  let rgbSW = textureSample(myTexture, mySampler, uv + texel * vec2(-1.0,  1.0)).rgb;
  let rgbSE = textureSample(myTexture, mySampler, uv + texel * vec2( 1.0,  1.0)).rgb;
  let rgbM  = textureSample(myTexture, mySampler, uv).rgb;

  let luma = vec3<f32>(0.299, 0.587, 0.114);
  let lumaNW = dot(rgbNW, luma);
  let lumaNE = dot(rgbNE, luma);
  let lumaSW = dot(rgbSW, luma);
  let lumaSE = dot(rgbSE, luma);
  let lumaM  = dot(rgbM,  luma);

  let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

  var dir = vec2<f32>(
    -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
    ((lumaNW + lumaSW) - (lumaNE + lumaSE))
  );

  let dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.5, 1.0 / 32.0);
  let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2<f32>(-8.0, -8.0), vec2<f32>(8.0, 8.0)) * texel;

  let rgbA = 0.5 * (
    textureSample(myTexture, mySampler, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
    textureSample(myTexture, mySampler, uv + dir * (2.0 / 3.0 - 0.5)).rgb
  );
  let rgbB = rgbA * 0.5 + 0.25 * (
    textureSample(myTexture, mySampler, uv + dir * -0.5).rgb +
    textureSample(myTexture, mySampler, uv + dir * 0.5).rgb
  );

  let lumaB = dot(rgbB, luma);
  var color: vec4<f32>;
  if (lumaB < lumaMin || lumaB > lumaMax) {
      color = vec4<f32>(rgbA, 1.0);
  } else {
      color = vec4<f32>(rgbB, 1.0);
  }
  return color;
}