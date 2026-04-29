window.SHADERS = Object.assign(window.SHADERS || {}, {
    volumeSlice: /* wgsl */ `

struct Uniforms {
    windowWidth: f32,
    windowLevel: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volumeTexture: texture_3d<f32>;
@group(0) @binding(2) var volumeSampler: sampler;

struct VSIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vert_main(input: VSIn) -> VSOut {
    var o: VSOut;
    o.position = vec4<f32>(input.pos, 0.0, 1.0);
    o.uv = input.uv;
    return o;
}

@fragment
fn frag_main(input: VSOut) -> @location(0) vec4<f32> {
    
    let size = vec3<f32>(textureDimensions(volumeTexture));
    // Guard
    if (input.uv.x < 0.0 || input.uv.x > 1.0 || input.uv.y < 0.0 || input.uv.y > 1.0) {
        return vec4<f32>(0.0);
    }
    let start = vec3<f32>(input.uv.x * size.x,
                          input.uv.y * size.y,
                          0.0);
    var maxVal: f32 = 0.0;

    for (var z: u32 = 0u; z < u32(size.z); z = z + 1u) {
        let samplePos = vec3<f32>(start.x, start.y, f32(z)) / size;
        let s = textureSampleLevel(volumeTexture, volumeSampler, samplePos, 0.0).r;
        maxVal = max(maxVal, s);
    }

    let ww = uniforms.windowWidth;
    let wl = uniforms.windowLevel;
    let normalized = (maxVal - (wl - ww * 0.5)) / ww;
    let outVal = clamp(normalized, 0.0, 1.0);
    return vec4<f32>(outVal, outVal, outVal, 1.0);
}
`});