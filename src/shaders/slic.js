window.SHADERS = Object.assign(window.SHADERS || {}, {
    slicInit: /* wgsl */ `
// Initialize cluster centers on a regular grid

struct SlicParams {
    volumeWidth: u32,
    volumeHeight: u32,
    volumeDepth: u32,
    gridSpacing: u32, // spacing between cluster centers
    compactness: f32,
    numClustersX: u32,
    numClustersY: u32,
    numClustersZ: u32,
    iteration: u32,
    _pad: u32,
}

struct ClusterCenter {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,// average intensity
    count: u32,// number of voxels assigned
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: SlicParams;
@group(0) @binding(1) var volumeTexture: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> centers: array<ClusterCenter>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    let cz = gid.z;
    
    if (cx >= params.numClustersX || cy >= params.numClustersY || cz >= params.numClustersZ) {
        return;
    }
    
    let clusterIdx = cx + cy * params.numClustersX + cz * params.numClustersX * params.numClustersY;
    
    // Place centers at regular grid positions (offset by half spacing)
    let halfSpacing = f32(params.gridSpacing) / 2.0;
    var posX = f32(cx * params.gridSpacing) + halfSpacing;
    var posY = f32(cy * params.gridSpacing) + halfSpacing;
    var posZ = f32(cz * params.gridSpacing) + halfSpacing;
    
    // Clamp to volume bounds
    posX = min(posX, f32(params.volumeWidth - 1u));
    posY = min(posY, f32(params.volumeHeight - 1u));
    posZ = min(posZ, f32(params.volumeDepth - 1u));
    
    // Move center to lowest gradient position in 3x3x3 neighborhood
    var minGrad = 1e10;
    var bestPos = vec3<f32>(posX, posY, posZ);
    
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nx = i32(posX) + dx;
                let ny = i32(posY) + dy;
                let nz = i32(posZ) + dz;
                
                if (nx >= 0 && nx < i32(params.volumeWidth) - 1 &&
                    ny >= 0 && ny < i32(params.volumeHeight) - 1 &&
                    nz >= 0 && nz < i32(params.volumeDepth) - 1) {
                    
                    let texCoord = vec3<i32>(nx, ny, nz);
                    
                    // Compute gradient magnitude
                    let gx = textureLoad(volumeTexture, texCoord + vec3<i32>(1, 0, 0), 0).r -
                             textureLoad(volumeTexture, texCoord - vec3<i32>(1, 0, 0), 0).r;
                    let gy = textureLoad(volumeTexture, texCoord + vec3<i32>(0, 1, 0), 0).r -
                             textureLoad(volumeTexture, texCoord - vec3<i32>(0, 1, 0), 0).r;
                    let gz = textureLoad(volumeTexture, texCoord + vec3<i32>(0, 0, 1), 0).r -
                             textureLoad(volumeTexture, texCoord - vec3<i32>(0, 0, 1), 0).r;
                    
                    let grad = gx * gx + gy * gy + gz * gz;
                    
                    if (grad < minGrad) {
                        minGrad = grad;
                        bestPos = vec3<f32>(f32(nx), f32(ny), f32(nz));
                    }
                }
            }
        }
    }
    
    // Sample intensity at best position
    let intensity = textureLoad(volumeTexture, vec3<i32>(bestPos), 0).r;
    
    centers[clusterIdx].x = bestPos.x;
    centers[clusterIdx].y = bestPos.y;
    centers[clusterIdx].z = bestPos.z;
    centers[clusterIdx].intensity = intensity;
    centers[clusterIdx].count = 0u;
}
`,

slicAssign: /* wgsl */ `
// Assign each voxel to nearest cluster center
struct SlicParams {
    volumeWidth: u32,
    volumeHeight: u32,
    volumeDepth: u32,
    gridSpacing: u32,
    compactness: f32,
    numClustersX: u32,
    numClustersY: u32,
    numClustersZ: u32,
    iteration: u32,
    _pad: u32,
}

struct ClusterCenter {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: SlicParams;
@group(0) @binding(1) var volumeTexture: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> centers: array<ClusterCenter>;
@group(0) @binding(3) var<storage, read_write> labels: array<u32>;
@group(0) @binding(4) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= params.volumeWidth || y >= params.volumeHeight || z >= params.volumeDepth) {
        return;
    }
    
    let voxelIdx = x + y * params.volumeWidth + z * params.volumeWidth * params.volumeHeight;
    let voxelIntensity = textureLoad(volumeTexture, vec3<i32>(gid), 0).r;
    let voxelPos = vec3<f32>(f32(x), f32(y), f32(z));
    
    // Search window: 2S x 2S x 2S around the voxel
    let S = f32(params.gridSpacing);
    let invS = 1.0 / S;
    
    // Find which cluster cell this voxel belongs to
    let cellX = i32(f32(x) * invS);
    let cellY = i32(f32(y) * invS);
    let cellZ = i32(f32(z) * invS);
    
    var minDist = 1e10;
    var bestLabel = 0u;
    
    // Check 3x3x3 neighborhood of cluster cells
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let cx = cellX + dx;
                let cy = cellY + dy;
                let cz = cellZ + dz;
                
                if (cx >= 0 && cx < i32(params.numClustersX) &&
                    cy >= 0 && cy < i32(params.numClustersY) &&
                    cz >= 0 && cz < i32(params.numClustersZ)) {
                    
                    let clusterIdx = u32(cx) + u32(cy) * params.numClustersX + 
                                    u32(cz) * params.numClustersX * params.numClustersY;
                    
                    let center = centers[clusterIdx];
                    let centerPos = vec3<f32>(center.x, center.y, center.z);
                    
                    // Compute distance (intensity + spatial)
                    // D = sqrt(dc^2 + (ds/S)^2 * m^2)
                    let dc = abs(voxelIntensity - center.intensity); //
                    let spatialDiff = voxelPos - centerPos;
                    let ds = length(spatialDiff);
                    
                    // Combined distance with compactness
                    let m = params.compactness;

                    //let dsScaled = (ds * invS) * m;
                    //let dist = sqrt(dc * dc + dsScaled * dsScaled);
                    let dist = sqrt(dc * dc + (ds * invS) * (ds * invS) * m * m);
                    
                    if (dist < minDist) {
                        minDist = dist;
                        bestLabel = clusterIdx;
                    }
                }
            }
        }
    }
    
    labels[voxelIdx] = bestLabel;
}
`,

slicUpdatePrepare: /* wgsl */ `
// Reset cluster accumulators before update

struct SlicParams {
    volumeWidth: u32,
    volumeHeight: u32,
    volumeDepth: u32,
    gridSpacing: u32,
    compactness: f32,
    numClustersX: u32,
    numClustersY: u32,
    numClustersZ: u32,
    iteration: u32,
    _pad: u32,
}

struct ClusterAccum {
    sumX: atomic<u32>,
    sumY: atomic<u32>,
    sumZ: atomic<u32>,
    sumIntensity: atomic<u32>,
    count: atomic<u32>,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: SlicParams;
@group(0) @binding(1) var<storage, read_write> accumulators: array<ClusterAccum>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numClusters = params.numClustersX * params.numClustersY * params.numClustersZ;
    
    if (gid.x >= numClusters) {
        return;
    }
    
    atomicStore(&accumulators[gid.x].sumX, 0u);
    atomicStore(&accumulators[gid.x].sumY, 0u);
    atomicStore(&accumulators[gid.x].sumZ, 0u);
    atomicStore(&accumulators[gid.x].sumIntensity, 0u);
    atomicStore(&accumulators[gid.x].count, 0u);
}
`,

slicAccumulate: /* wgsl */ `   
// Accumulate voxel contributions to cluster centers

struct SlicParams {
    volumeWidth: u32,
    volumeHeight: u32,
    volumeDepth: u32,
    gridSpacing: u32,
    compactness: f32,
    numClustersX: u32,
    numClustersY: u32,
    numClustersZ: u32,
    iteration: u32,
    _pad: u32,
}

struct ClusterAccum {
    sumX: atomic<u32>,
    sumY: atomic<u32>,
    sumZ: atomic<u32>,
    sumIntensity: atomic<u32>,
    count: atomic<u32>,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: SlicParams;
@group(0) @binding(1) var volumeTexture: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> accumulators: array<ClusterAccum>;

// Fixed-point scale for accumulating floats as integers
const FIXED_SCALE: u32 = 4096u; //256u;// 65536u; //256u TODD

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= params.volumeWidth || y >= params.volumeHeight || z >= params.volumeDepth) {
        return;
    }
    
    let voxelIdx = x + y * params.volumeWidth + z * params.volumeWidth * params.volumeHeight;
    let label = labels[voxelIdx];
    let intensity = textureLoad(volumeTexture, vec3<i32>(gid), 0).r;
    
    // Accumulate using atomics (convert to fixed-point)
    atomicAdd(&accumulators[label].sumX, x * FIXED_SCALE);
    atomicAdd(&accumulators[label].sumY, y * FIXED_SCALE);
    atomicAdd(&accumulators[label].sumZ, z * FIXED_SCALE);
    atomicAdd(&accumulators[label].sumIntensity, u32(intensity * f32(FIXED_SCALE)));
    atomicAdd(&accumulators[label].count, 1u);
}
`,

    slicUpdateCenters: /* wgsl */ `
// Compute new cluster centers from accumulators

struct SlicParams {
    volumeWidth: u32,
    volumeHeight: u32,
    volumeDepth: u32,
    gridSpacing: u32,
    compactness: f32,
    numClustersX: u32,
    numClustersY: u32,
    numClustersZ: u32,
    iteration: u32,
    _pad: u32,
}

struct ClusterCenter {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct ClusterAccum {
    sumX: u32,
    sumY: u32,
    sumZ: u32,
    sumIntensity: u32,
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: SlicParams;
@group(0) @binding(1) var<storage, read> accumulators: array<ClusterAccum>;
@group(0) @binding(2) var<storage, read_write> centers: array<ClusterCenter>;

const FIXED_SCALE: f32 = 4096.0; //256;//65536.0; TODO

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numClusters = params.numClustersX * params.numClustersY * params.numClustersZ;
    
    if (gid.x >= numClusters) {
        return;
    }
    
    let accum = accumulators[gid.x];
    
    if (accum.count > 0u) {
        let invCount = 1.0 / (f32(accum.count) * FIXED_SCALE);
        centers[gid.x].x = f32(accum.sumX) * invCount;
        centers[gid.x].y = f32(accum.sumY) * invCount;
        centers[gid.x].z = f32(accum.sumZ) * invCount;
        centers[gid.x].intensity = f32(accum.sumIntensity) * invCount;
        centers[gid.x].count = accum.count;
    }
}
`
});