window.SHADERS = Object.assign(window.SHADERS || {}, {
    volVis: /* wgsl */ `
struct Uniforms {
    inverseViewMatrix: mat4x4<f32>,
    windowWidth: f32,
    windowLevel: f32,
    highlightFactor: f32, // 0 = no highlight, 1 = full highlight
    showOnlyHighlight: f32,   // 0 = normal, 1 = only highlight
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volumeTexture: texture_3d<f32>;
@group(0) @binding(2) var volumeSampler: sampler;
@group(0) @binding(3) var regionMask: texture_3d<f32>;

struct VSIn {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vert_main(input: VSIn) -> VSOut {
    var output: VSOut;
    output.position = vec4<f32>(input.pos, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

fn intersectBox(rayOrigin: vec3<f32>, rayDir: vec3<f32>) -> vec2<f32> {
    let boxMin = vec3<f32>(-0.5);
    let boxMax = vec3<f32>(0.5);
    
    let invDir = 1.0 / rayDir;
    let t0 = (boxMin - rayOrigin) * invDir;
    let t1 = (boxMax - rayOrigin) * invDir;
    
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    
    let tNear = max(max(tmin.x, tmin.y), tmin.z);
    let tFar = min(min(tmax.x, tmax.y), tmax.z);
    
    return vec2<f32>(tNear, tFar);
}

@fragment
fn frag_main(input: VSOut) -> @location(0) vec4<f32> 
{
    let rayOriginView = vec3<f32>(0.0, 0.0, 2.0);
    let rayDirView = normalize(vec3<f32>(input.uv * 2.0 - 1.0, -1.0));
    let rayOrigin = (uniforms.inverseViewMatrix * vec4<f32>(rayOriginView, 1.0)).xyz;
    let rayDir = normalize((uniforms.inverseViewMatrix * vec4<f32>(rayDirView, 0.0)).xyz);
    
    let tHit = intersectBox(rayOrigin, rayDir);
    
    if (tHit.x > tHit.y || tHit.y < 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    
    let tStart = max(tHit.x, 0.0);
    let tEnd = tHit.y;
    
    let numSteps = 256; //TODO: adjust for performance/quality
    let stepSize = (tEnd - tStart) / f32(numSteps);
    
    var accumColor = vec3<f32>(0.0);
    var accumAlpha = 0.0;
    let threshold = 0.02;
    
    for (var i = 0; i < numSteps; i++) {
        if (accumAlpha > 0.95) {
            break;
        }
        
        let t = tStart + f32(i) * stepSize;
        let pos = rayOrigin + t * rayDir;
        let texCoord = pos + 0.5;
        
        if (all(texCoord >= vec3<f32>(0.0)) && all(texCoord <= vec3<f32>(1.0))) {
            let intensity = textureSampleLevel(volumeTexture, volumeSampler, texCoord, 0.0).r;
            let maskVal = textureSampleLevel(regionMask, volumeSampler, texCoord, 0.0).r;
            
            // If in "only selected region" mode, skip voxels not in the mask. if not in that mode, skip voxels below threshold outside the mask.
            if ((uniforms.showOnlyHighlight > 0.5 && maskVal < 0.5) || (intensity <= threshold && uniforms.showOnlyHighlight < 0.5)) {
                continue;
            }

            let ww = uniforms.windowWidth;
            let wl = uniforms.windowLevel;
            let normalized = (intensity - (wl - ww * 0.5)) / ww;
                
            let intensityVal = clamp(normalized, 0.0, 1.0);
            let volumeColor = vec3<f32>(intensityVal);

            if (intensity > threshold && maskVal < 0.5) {
                let alpha = intensityVal * 0.08; // opacity scaling
                accumColor = accumColor + volumeColor * alpha * (1.0 - accumAlpha);
                accumAlpha = accumAlpha + alpha * (1.0 - accumAlpha);
            }

            // Highlight voxels inside the selected region
            if (maskVal > 0.5) {
                
                let highlightStrength = clamp(uniforms.highlightFactor, 0.0, 1.0);
                let highlightColor = vec3<f32>(1.0, 0.5, 0.0); // orange-ish
                var sampleColor = mix(volumeColor, highlightColor, highlightStrength);

                let alpha = intensityVal * 0.08; // opacity scaling
                accumColor = accumColor + sampleColor * alpha * (1.0 - accumAlpha);
                accumAlpha = accumAlpha + alpha * (1.0 - accumAlpha);
            }
        }
    }
    
    return vec4<f32>(accumColor, 1.0);
}
`});