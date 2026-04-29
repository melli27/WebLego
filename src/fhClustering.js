// fhClustering.js
// FH clustering over SLIC supervoxels:
// per-supervoxel intensity histogram
// χ² edge weights
// 26-neighborhood adjacency
// exhaustive k scan on a log grid

/**
 * @typedef {{ width: number, height: number, depth: number }} VolumeDims
 * @typedef {{
 *   id: number,
 *   voxelCount: number,
 *   meanIntensity: number,
 *   histogram: Float32Array
 * }} SupervoxelFeature
 * @typedef {{ a: number, b: number, weight: number }} Edge
 * @typedef {{ id: number, supervoxels: number[], kMin: number, kMax: number }} Region
 */

const HIST_BINS = 64;

class FHClustering {
    /**
     * @param {Uint32Array} labels // voxel -> supervoxel label from SLIC
     * @param {Float32Array} volumeData // voxel intensities in [0,1]
     * @param {VolumeDims} dims
     */
    constructor(labels, volumeData, dims) {
        this.labels = labels;
        this.volumeData = volumeData;
        this.dims = dims;

        /** @type {SupervoxelFeature[]} */
        this.features = this.computeSupervoxelFeatures();
        this.numSupervoxels = this.features.length; // == maxLabel+1
        /** @type {Edge[]} */
        this.edges = this.buildSupervoxelAdjacency();
    }


    //
    /* Compute per-supervoxel features: mean intensity + histogram */
    //
    /** @returns {SupervoxelFeature[]} */
    computeSupervoxelFeatures() {
        const { width, height, depth } = this.dims;
        const totalVoxels = width * height * depth;

        // Find max label (assume labels in [0, maxLabel])
        let maxLabel = 0;
        for (let i = 0; i < totalVoxels; i++) {
            if (this.labels[i] > maxLabel) maxLabel = this.labels[i];
        }
        const numSV = maxLabel + 1;

        const sumIntensity = new Float64Array(numSV);
        const count = new Uint32Array(numSV);
        /** @type {Float32Array[]} histograms[label] = Float32Array(HIST_BINS) */
        const histograms = new Array(numSV);
        for (let l = 0; l < numSV; l++) {
            histograms[l] = new Float32Array(HIST_BINS);
        }

        // Accumulate sums + hist counts per label
        for (let i = 0; i < totalVoxels; i++) {
            const l = this.labels[i];
            const v = this.volumeData[i]; // assumed [0,1]

            sumIntensity[l] += v;
            count[l]++;

            // Histogram bin index
            let bin = Math.floor(v * HIST_BINS);
            if (bin < 0) bin = 0;
            if (bin >= HIST_BINS) bin = HIST_BINS - 1;
            histograms[l][bin] += 1;
        }

        /** @type {SupervoxelFeature[]} */
        const features = new Array(numSV);
        for (let id = 0; id < numSV; id++) {
            const voxCount = count[id];
            const hist = histograms[id];

            let mean = 0;
            if (voxCount > 0) {
                mean = sumIntensity[id] / voxCount;

                // Normalize histogram to a probability distribution
                const inv = 1.0 / voxCount;
                for (let b = 0; b < HIST_BINS; b++) {
                    hist[b] *= inv;
                }
            } else {
                // If a label somehow has no voxels, leave hist all zero
                for (let b = 0; b < HIST_BINS; b++) {
                    hist[b] = 0;
                }
            }

            features[id] = {
                id,
                voxelCount: voxCount,
                meanIntensity: mean,
                histogram: hist,
            };
        }

        return features;
    }

    //
    /* Build supervoxel adjacency graph using 26-neighborhood and χ² distance */
    //
    /**
     * χ² distance between two histograms (FeatureLego / Grundmann style)
     * @param {Float32Array} h1
     * @param {Float32Array} h2
     */
    static chiSquareDistance(h1, h2) {
        let d = 0;
        const n = h1.length;
        for (let i = 0; i < n; i++) {
            const a = h1[i];
            const b = h2[i];
            const s = a + b;
            if (s > 0) {
                const diff = a - b;
                d += (diff * diff) / s;
            }
        }
        return 0.5 * d;
    }

    /** @returns {Edge[]} */
    buildSupervoxelAdjacency() {
        const { width, height, depth } = this.dims;

        // Quick label -> histogram lookup
        /** @type {Float32Array[]} */
        const histograms = new Array(this.numSupervoxels);
        for (const f of this.features) {
            histograms[f.id] = f.histogram;
        }

        /** @type {Set<string>} */
        const edgeSet = new Set();
        /** @type {Edge[]} */
        const edges = [];

        const idx = (/** @type {number} */ x, /** @type {number} */ y, /** @type {number} */ z) => x + y * width + z * width * height;

        // 26-neighborhood in 3D: all offsets in [-1,0,1]^3 except (0,0,0)
        /** @type {number[][]} */
        const neighbors = [];
        for (let dz = -1; dz <= 1; dz++) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if (dx === 0 && dy === 0 && dz === 0) continue;
                    neighbors.push([dx, dy, dz]);
                }
            }
        }

        for (let z = 0; z < depth; z++) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const i = idx(x, y, z);
                    const l = this.labels[i];

                    for (const [dx, dy, dz] of neighbors) {
                        const nx = x + dx;
                        const ny = y + dy;
                        const nz = z + dz;
                        if (
                            nx < 0 || nx >= width ||
                            ny < 0 || ny >= height ||
                            nz < 0 || nz >= depth
                        ) {
                            continue;
                        }

                        const j = idx(nx, ny, nz);
                        const l2 = this.labels[j];
                        if (l2 === l) continue;

                        const a = Math.min(l, l2);
                        const b = Math.max(l, l2);
                        const key = `${a},${b}`;
                        if (edgeSet.has(key)) continue;
                        edgeSet.add(key);

                        const ha = histograms[a];
                        const hb = histograms[b];
                        if (!ha || !hb) continue; // safety

                        const w = FHClustering.chiSquareDistance(ha, hb);
                        edges.push({ a, b, weight: w });
                    }
                }
            }
        }

        return edges;
    }

    /*FH segmentation & Clustering
     */

    /**
     * Exhaustive FH over an explicit set of k values.
     * Merges identical regions across consecutive k's into [kMin,kMax] intervals.
     * giving each region a kmin and kmax (the smallest and largest k where that exact supervoxel set appear)
     * @param {number[]} kValues
     * @returns {Map<string, Region>} key -> region with [kMin,kMax]
     */
    runExhaustive(kValues) {
        /** @type {Map<string, Region>} */
        const regions = new Map();

        for (const k of kValues) {
            const comp = this.segmentFHForK(k);
            const regionMap = this.componentsToRegions(comp);
            //console.log("k =", k, "num regions at this k:", regionMap.size);

            for (const [key, svs] of regionMap.entries()) {
                if (!regions.has(key)) {
                    regions.set(key, {
                        id: svs[0],
                        supervoxels: svs.slice(),
                        kMin: k,
                        kMax: k,
                    });
                } else {
                    const r = regions.get(key);
                    // extend interval if we see region again
                    if (k < r.kMin) r.kMin = k;
                    if (k > r.kMax) r.kMax = k;
                }
            }
        }

        return regions;
    }

    /**
     * Derives a reasonable [kMin,kMax] from edge weights and runs FH on log-spaced k values.
     * @param {number} numSamples how many k values to sample (log spaced)
     * @returns {Map<string, Region>}
     */
    runExhaustiveAuto(numSamples = 32) {
        if (this.edges.length === 0) {
            return new Map();
        }

        // Rough k range from edge weights
        let minW = Infinity;
        let maxW = -Infinity;
        for (const e of this.edges) {
            if (e.weight < minW) minW = e.weight;
            if (e.weight > maxW) maxW = e.weight;
        }
        if (!isFinite(minW)) minW = 1e-4;
        if (!isFinite(maxW) || maxW <= 0) maxW = 1.0;

        let kMin = Math.max(minW * 0.25, 1e-4);
        let kMax = maxW * this.numSupervoxels;

        if (kMin <= 0) kMin = 1e-4;
        if (kMax <= kMin) kMax = kMin * 10;

        const kValues = [];
        const logMin = Math.log10(kMin);
        const logMax = Math.log10(kMax);
        for (let i = 0; i < numSamples; i++) {
            const t = numSamples === 1 ? 0 : i / (numSamples - 1);
            const k = Math.pow(10, logMin + t * (logMax - logMin));
            kValues.push(k);
        }

        return this.runExhaustive(kValues);
    }

    /**
     * Run FH segmentation for one k
     * @param {number} k
     * @returns {Int32Array} comp[svId] -> component ID
     */
    segmentFHForK(k) {
        const numSV = this.numSupervoxels;
        const sortedEdges = this.edges.slice().sort((e1, e2) => e1.weight - e2.weight);

        const uf = FHClustering.createUnionFindFH(numSV);

        for (const e of sortedEdges) {
            uf.union(e.a, e.b, e.weight, k);
        }

        const comp = new Int32Array(numSV);
        for (let i = 0; i < numSV; i++) {
            comp[i] = uf.find(i);
        }
        return comp;
    }

    /**
    * Internal union-find structure for FH
    * @param {number} n
    */
    static createUnionFindFH(n) {
        const parent = new Int32Array(n);
        const size = new Int32Array(n);
        const intDiff = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
            intDiff[i] = 0;
        }

        const uf = {
            parent,
            size,
            intDiff,
            /**
             * @param {number} x
             * @returns {number}
             */
            find(x) {
                let root = x;
                while (this.parent[root] !== root) root = this.parent[root];
                let cur = x;
                while (cur !== root) {
                    const next = this.parent[cur];
                    this.parent[cur] = root;
                    cur = next;
                }
                return root;
            },
            /**
             * @param {number} a
             * @param {number} b
             * @param {number} edgeW
             * @param {number} k
             * @returns {boolean} true if merged
             */
            union(a, b, edgeW, k) {
                let ra = this.find(a);
                let rb = this.find(b);
                if (ra === rb) return false;

                const sizeA = this.size[ra];
                const sizeB = this.size[rb];
                const intA = this.intDiff[ra];
                const intB = this.intDiff[rb];

                const threshA = intA + k / sizeA;
                const threshB = intB + k / sizeB;

                if (edgeW > Math.min(threshA, threshB)) {
                    return false;
                }

                // merge smaller into larger
                if (sizeA < sizeB) {
                    const tmp = ra; ra = rb; rb = tmp;
                }

                this.parent[rb] = ra;
                this.size[ra] = sizeA + sizeB;
                this.intDiff[ra] = Math.max(edgeW, intA, intB);
                return true;
            }
        };
        return uf;
    }

    /**
     * Convert component labels to canonical region definitions
     * @param {Int32Array} comp
     * @returns {Map<string, number[]>} key -> list of supervoxel IDs
     */
    componentsToRegions(comp) {
        /** @type {Map<number, number[]>} */
        const byRoot = new Map();
        for (let sv = 0; sv < comp.length; sv++) {
            const c = comp[sv];
            if (!byRoot.has(c)) byRoot.set(c, []);
            byRoot.get(c).push(sv);
        }

        /** @type {Map<string, number[]>} */
        const regionMap = new Map();
        for (const svs of byRoot.values()) {
            svs.sort((a, b) => a - b);
            const key = svs.join(",");
            regionMap.set(key, svs);
        }
        return regionMap;
    }




    // Voxel masks for Visualization
    // 

    /**
     * Build a voxel mask (0 or 255) for a given region.
     * @param {number[]} regionSupervoxels - list of supervoxel IDs in the region
     * @returns {Uint8Array} length = width*height*depth, 0 or 255
     */
    regionToVoxelMask(regionSupervoxels) {
        const { width, height, depth } = this.dims;
        const totalVoxels = width * height * depth;

        const inRegion = new Uint8Array(this.numSupervoxels);
        for (const sv of regionSupervoxels) {
            if (sv >= 0 && sv < inRegion.length) inRegion[sv] = 1;
        }

        const mask = new Uint8Array(totalVoxels);
        for (let i = 0; i < totalVoxels; i++) {
            const sv = this.labels[i];
            mask[i] = inRegion[sv] ? 255 : 0; // 255 so r8unorm gives 1.0
        }
        return mask;
    }

    /**
     * Create a 3D r8unorm texture from a mask (0..255).
     * @param {GPUDevice} device
     * @param {Uint8Array} mask
     * @returns {GPUTexture}
     */
    createRegionMaskTexture(device, mask) {
        const { width, height, depth } = this.dims;

        const texture = device.createTexture({
            size: { width, height, depthOrArrayLayers: depth },
            dimension: "3d",
            format: "r8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        device.queue.writeTexture(
            { texture },
            mask.buffer,
            {
                offset: 0,
                bytesPerRow: width,      // 1 byte per voxel
                rowsPerImage: height,
            },
            { width, height, depthOrArrayLayers: depth }
        );

        return texture;
    }
}
