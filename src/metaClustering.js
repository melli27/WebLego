// metaClustering.js

// Build a meta-cluster hierarchy over FH regions using Jaccard similarity
// over supervoxel sets + MST (reverse-delete).

/**
 * @typedef {Object} FHRegion
 * @property {number} id
 * @property {number[]} supervoxels
 * @property {number} kMin
 * @property {number} kMax
 */

/**
 * A "meta-cluster" is a group of FH regions that heavily overlap.
 * It is defined purely as a set of region indices.
 *
 * @typedef {Object} MetaCluster
 * @property {number} id
 * @property {number[]} regionIndices
 * @property {number} voxelCount
 */

/**
 * A node in the meta-cluster tree.
 *
 * @typedef {Object} MetaClusterNode
 * @property {number} id
 * @property {boolean} isLeaf
 * @property {number=} metaIndex // for leaves: index into metaClusters[]
 * @property {MetaClusterNode[]} children
 * @property {MetaClusterNode|null} parent
 * @property {number} voxelCount
 */

/**
 * @typedef {Object} MetaClusterTree
 * @property {MetaClusterNode} root
 * @property {MetaClusterNode[]} nodes
 * @property {MetaClusterNode[]} leafNodes
 */

/**
 * Build a meta-cluster hierarchy over FH regions.
 * Works purely in supervoxel space for speed; voxel masks are only built
 * when you actually want to visualize a node.
 */
class MetaClusterBuilder {
    /**
     * @param {FHClustering} fh
     * @param {Map<string, FHRegion>} fhRegions
     */
    constructor(fh, fhRegions) {
        /** @type {FHClustering} */
        this.fh = fh;
        /** @type {Map<string, FHRegion>} */
        this.fhRegions = fhRegions;

        /** @type {{ key:string, region:FHRegion }[]} */
        this.regions = Array.from(fhRegions.entries()).map(
            /** @returns {{ key:string, region:FHRegion }} */
            ([key, region]) => ({ key, region })
        );

        /** @type {number} */
        this.numRegions = this.regions.length;

        /** @type {Array<Set<number>>} supervoxel sets per region */
        this.regionSVSets = new Array(this.numRegions);
        /** @type {number[]} voxel counts per region */
        this.regionVoxelCounts = new Array(this.numRegions);

        /** @type {number[]} per-supervoxel voxel counts (features[id].voxelCount) */
        this.supervoxelVoxelCounts = this.fh.features.map(f => f.voxelCount);

        /** @type {{ a:number, b:number, jacc:number, weight:number }[]} */
        this.edges = [];

        /** @type {MetaCluster[]} */
        this.metaClusters = [];
    }

    /**
     * Precompute:
     * - regionSVSets[i] = Set of supervoxel IDs in region i
     * - regionVoxelCounts[i] = sum of voxelCount over those supervoxels
     *   using fh.features[sv].voxelCount (SupervoxelFeature from fhClustering.js)
     */
    precomputeRegionStats() {
        /** @type {Array<{voxelCount:number}>} */
        const features = /** @type {any} */ (this.fh.features);

        for (let i = 0; i < this.numRegions; i++) {
            const svs = this.regions[i].region.supervoxels;
            /** @type {Set<number>} */
            const svSet = new Set(svs);
            let voxCount = 0;

            for (const sv of svSet) {
                const f = features[sv];
                if (!f) continue;
                voxCount += f.voxelCount;
            }

            this.regionSVSets[i] = svSet;
            this.regionVoxelCounts[i] = voxCount;
        }
    }

    /**
     * Region similarity graph (Jaccard over supervoxel sets)
     * Voxel-weighted Jaccard similarity between two regions
     *
     * @param {Set<number>} setA
     * @param {Set<number>} setB
     * @param {number[]} supervoxelVoxelCounts   // per-supervoxel voxelCount
     * @param {number} voxCountA                // total voxels in region A
     * @param {number} voxCountB                // total voxels in region B
     * @returns {number}
     */
    jaccard(setA, setB, supervoxelVoxelCounts, voxCountA, voxCountB) {
        // Sum voxelCount over the intersection of the two supervoxel sets
        let inter = 0;
        const smaller = setA.size < setB.size ? setA : setB;
        const bigger = smaller === setA ? setB : setA;

        for (const sv of smaller) {
            if (bigger.has(sv)) {
                const c = supervoxelVoxelCounts[sv] || 0;
                inter += c;
            }
        }

        const union = voxCountA + voxCountB - inter;
        return union === 0 ? 0 : inter / union;
    }

    /**
     * Build region similarity graph edges with Jaccard over supervoxel sets.
     *
     */
    buildSimilarityGraph() {
        this.edges.length = 0;
        const R = this.numRegions;

        for (let i = 0; i < R; i++) {
            const setA = this.regionSVSets[i];
            const voxCountA = this.regionVoxelCounts[i];

            for (let j = i + 1; j < R; j++) {
                const setB = this.regionSVSets[j];
                const voxCountB = this.regionVoxelCounts[j];

                const jacc = this.jaccard(setA, setB, this.supervoxelVoxelCounts, voxCountA, voxCountB);

                // Skip non-overlapping regions
                if (jacc <= 0) continue;

                const weight = 1.0 - jacc; // Jaccard distance
                this.edges.push({ a: i, b: j, jacc, weight });
            }
        }

        console.log(`[MetaClustering] Similarity graph built with ${this.edges.length} edges`);
    }

    /**
     * Build MST edges using the reverse-delete algorithm.
     * @returns {number[]} indices into this.edges[] that are in the MST
     */
    buildMSTReverseDelete() {
        const R = this.numRegions;
        const E = this.edges.length;
        if (R === 0 || E === 0) return [];

        /** @type {Array<Array<{to:number, edgeIndex:number}>>} */
        const adj = new Array(R);
        for (let i = 0; i < R; i++) adj[i] = [];
        for (let ei = 0; ei < E; ei++) {
            const e = this.edges[ei];
            adj[e.a].push({ to: e.b, edgeIndex: ei });
            adj[e.b].push({ to: e.a, edgeIndex: ei });
        }

        /** @type {boolean[]} */
        const edgeUsed = new Array(E).fill(true);
        const order = Array.from({ length: E }, (_, i) => i);
        order.sort((i, j) => this.edges[j].weight - this.edges[i].weight);

        /**
         * DFS from start over edges that are currently used.
         * @param {number} start
         * @param {number} target
         * @returns {boolean}
         */
        const bfsConnectivity = (start, target) => {
            if (start === target) return true;
            /** @type {Uint8Array} */
            const visited = new Uint8Array(R);
            /** @type {number[]} */
            const queue = [start];
            visited[start] = 1;

            while (queue.length > 0) {
                const v = queue.shift();
                if (v === target) return true;
                const neighbors = adj[v];
                for (let k = 0; k < neighbors.length; k++) {
                    const { to, edgeIndex } = neighbors[k];
                    if (!edgeUsed[edgeIndex]) continue;
                    if (!visited[to]) {
                        visited[to] = 1;
                        queue.push(to);
                    }
                }
            }
            return false;
        };

        for (const ei of order) {
            const e = this.edges[ei];
            edgeUsed[ei] = false;
            const stillConnected = bfsConnectivity(e.a, e.b);
            if (!stillConnected) {
                edgeUsed[ei] = true;
            }
        }

        /** @type {number[]} */
        const mstEdges = [];
        for (let ei = 0; ei < E; ei++) {
            if (edgeUsed[ei]) mstEdges.push(ei);
        }
        return mstEdges;
    }

    /**
     * Build meta-clusters as connected components of the MST
     * after cutting edges with weight > maxAllowedWeight (J < simThreshold).
     * only keep egdes with d_J < distanceThreshold
     * @param {number[]} mstEdgesIndices
     * @param {number} distanceThreshold // max allowed weight
     * @returns {MetaCluster[]}
     */
    buildMetaClustersFromMST(mstEdgesIndices, distanceThreshold) {
        const R = this.numRegions;
        if (R === 0) return [];

        // 1. Build Adjacency List from filtered MST edges
        // adjacency over region indices, only for MST edges <= maxAllowedWeight
        /** @type {Array<Set<number>>} */
        const adj = Array.from({ length: R }, () => new Set());

        // If mstEdgesIndices is empty, adj simply remains unconnected (correct behavior)
        if (mstEdgesIndices && mstEdgesIndices.length > 0) {
            for (const ei of mstEdgesIndices) {
                const e = this.edges[ei];
                // Keep edge if it meets the similarity requirement (distance < threshold)
                if (e.weight <= distanceThreshold) {
                    adj[e.a].add(e.b);
                    adj[e.b].add(e.a);
                }
            }
        }

        // 2. Find Connected Components (Meta-Clusters)
        const visited = new Uint8Array(R); // 0 = false, 1 = true
        /** @type {MetaCluster[]} */
        const metaClusters = [];
        let mcId = 0;

        for (let i = 0; i < R; i++) {
            if (visited[i]) continue;

            /** @type {number[]} */
            const indices = [];
            const stack = [i];
            visited[i] = 1;

            while (stack.length > 0) {
                const v = stack.pop();
                indices.push(v);

                for (const nb of adj[v]) {
                    if (!visited[nb]) {
                        visited[nb] = 1;
                        stack.push(nb);
                    }
                }
            }

            let voxCount = 0;
            for (const ri of indices) {
                voxCount += this.regionVoxelCounts[ri];
            }

            metaClusters.push({
                id: mcId++,
                regionIndices: indices,
                voxelCount: voxCount
            });
        }

        console.log(`Built ${metaClusters.length} meta-clusters from MST with distance threshold ${distanceThreshold}`);
        return metaClusters;
    }

    /**
     * Construct meta-cluster tree from metaClusters.
     * Parent-child is based on containment of voxel supports (union of supervoxels).
     * Siblings are sorted by decreasing voxelCount.
     *
     * @param {MetaCluster[]} metaClusters
     * @param {number} [maxBranching] maximum number of children per node
     * @returns {{ root: MetaClusterNode, nodes: MetaClusterNode[] }}
     */
    buildMetaClusterTree(metaClusters, maxBranching = 1) {
        const M = metaClusters.length;
        if (M === 0) {
            /** @type {MetaClusterNode} */
            const root = {
                id: 0,
                isLeaf: true,
                metaIndex: -1,
                children: [],
                parent: null,
                voxelCount: 0
            };
            return { root, nodes: [root] };
        }

        // For each meta-cluster, build the union of its supervoxels and compute its voxel count based on unique supervoxels.
        /** @type {Array<Set<number>>} */
        const metaSVSets = new Array(M);
        /** @type {number[]} */
        const metaVoxelCounts = new Array(M);

        for (let i = 0; i < M; i++) {
            const mc = metaClusters[i];
            /** @type {Set<number>} */
            const svSet = new Set();
            let voxCount = 0;

            for (const ri of mc.regionIndices) {
                const rSet = this.regionSVSets[ri];
                if (!rSet) continue;

                for (const sv of rSet) {
                    if (!svSet.has(sv)) {
                        svSet.add(sv);
                        voxCount += this.supervoxelVoxelCounts[sv] || 0;
                    }
                }
            }

            metaSVSets[i] = svSet;
            metaVoxelCounts[i] = voxCount;
        }

        /** @type {MetaClusterNode[]} */
        const nodes = new Array(M);
        for (let i = 0; i < M; i++) {
            nodes[i] = {
                id: i,
                isLeaf: true,
                metaIndex: i,
                children: [],
                parent: null,
                voxelCount: metaVoxelCounts[i]
            };
        }

        /** @type {MetaClusterNode} */
        const root = {
            id: M,
            isLeaf: false,
            metaIndex: -1,
            children: [],
            parent: null,
            voxelCount: metaVoxelCounts.reduce((s, v) => s + v, 0)
        };

        /** @type {MetaClusterNode[]} */
        const allNodes = [...nodes, root];

        // For each meta-cluster, find the smallest voxel-superset as parent.
        for (let idx = 0; idx < M; idx++) {
            const childNode = nodes[idx];
            const childSet = metaSVSets[idx];

            /** @type {MetaClusterNode|null} */
            let parentCandidate = null;
            let parentSize = Infinity;

            for (let j = 0; j < M; j++) {
                if (j === idx) continue;
                const supSet = metaSVSets[j];

                // Check containment: child ⊆ sup over supervoxel sets
                let isSubset = true;
                for (const sv of childSet) {
                    if (!supSet.has(sv)) {
                        isSubset = false;
                        break;
                    }
                }
                if (!isSubset) continue;

                const supSize = supSet.size;
                if (supSize < parentSize) {
                    parentSize = supSize;
                    parentCandidate = nodes[j];
                }
            }

            const parentNode = parentCandidate || root;
            childNode.parent = parentNode;
            parentNode.children.push(childNode);
        }

        // For each node, insert duplicates under every superset that is not already an ancestor.
        /** @type {MetaClusterNode[]} */
        const duplicateNodes = [];
        let nextId = M + 1; // IDs after root

        for (let idx = 0; idx < M; idx++) {
            const originalNode = nodes[idx];
            const childSet = metaSVSets[idx];

            /** @type {Array<{node: MetaClusterNode, size: number}>} */
            const supersets = [];

            for (let j = 0; j < M; j++) {
                if (j === idx) continue;
                const supSet = metaSVSets[j];

                // child ⊆ sup
                let isSubset = true;
                for (const sv of childSet) {
                    if (!supSet.has(sv)) {
                        isSubset = false;
                        break;
                    }
                }
                if (!isSubset) continue;

                supersets.push({
                    node: nodes[j],
                    size: metaVoxelCounts[j]
                });
            }

            if (supersets.length === 0) continue;

            // Sort supersets from smallest to largest
            supersets.sort((a, b) => a.size - b.size);

      /*       // only keep the first few (closest supersets)
            const MAX_SUPERSETS = 3;
            if (supersets.length > MAX_SUPERSETS) {
                supersets.length = MAX_SUPERSETS;
            } */

            for (const { node: supersetNode } of supersets) {
                // Skip the direct parent (already added)
                if (supersetNode === originalNode.parent) continue;

                // Avoid placing a node under one of its descendants
                /**
                 * @param {MetaClusterNode} candidate
                 * @param {MetaClusterNode} target
                 * @returns {boolean}
                 */
                const isAncestor = (candidate, target) => {
                    let current = target.parent;
                    while (current) {
                        if (current === candidate) return true;
                        current = current.parent;
                    }
                    return false;
                };

                if (!isAncestor(supersetNode, originalNode)) {
                    /** @type {MetaClusterNode} */
                    const duplicateNode = {
                        id: nextId++,
                        isLeaf: true,
                        metaIndex: originalNode.metaIndex,
                        children: [],
                        parent: supersetNode,
                        voxelCount: originalNode.voxelCount
                    };

                    supersetNode.children.push(duplicateNode);
                    duplicateNodes.push(duplicateNode);
                    allNodes.push(duplicateNode);
                }
            }
        }

        console.log(`Created ${duplicateNodes.length} duplicate nodes for non-ancestor supersets`);

        // Sort siblings by voxelCount and prune to maxBranching at each level
        /**
         * @param {MetaClusterNode} node
         */
        const sortAndPruneChildren = (node) => {
            if (!node.children || node.children.length === 0) return;

            // Sort descending by voxel count
            node.children.sort((a, b) => b.voxelCount - a.voxelCount);

            // Prune if exceeds max branching factor
            if (node.children.length > maxBranching) {
                node.children.length = maxBranching;
            }

            for (const c of node.children) {
                sortAndPruneChildren(c);
            }
        };

        sortAndPruneChildren(root);

        return { root, nodes: allNodes };
    }

    /**
     * Build the complete meta-cluster hierarchy.
     *
     * @param {number} [distanceThreshold] Jaccard distance threshold d_J (e.g. 0.3)
     * @param {number} [maxBranching] Maximum number of children per node
     * @returns {MetaClusterTree}
     */
    build(distanceThreshold = 0.3, maxBranching = 3) {
        if (this.numRegions === 0) {
            /** @type {MetaClusterNode} */
            const root = {
                id: 0,
                isLeaf: true,
                metaIndex: -1,
                children: [],
                parent: null,
                voxelCount: 0
            }
            return { root, nodes: [root], leafNodes: [root] };
        }

        this.precomputeRegionStats();

        this.buildSimilarityGraph();

        const mstEdgesIndices = this.buildMSTReverseDelete();
        console.log(`MetaClusterBuilder.build: MST has ${mstEdgesIndices.length} edges`);

        const metaClusters = this.buildMetaClustersFromMST(mstEdgesIndices, distanceThreshold);
        this.metaClusters = metaClusters;

        const { root, nodes } = this.buildMetaClusterTree(metaClusters, maxBranching);

        const leafNodes = nodes.filter(n => n.isLeaf);

        console.log(`MetaClusterBuilder.build: Meta-cluster tree has ${nodes.length} nodes (including duplicates and root).`);
        console.log(this.metaClusters.length, "meta-clusters created from", this.fhRegions.size, "regions.");

        return { root, nodes, leafNodes };
    }

    /**
     * Build a voxel mask (Uint8Array) for the given meta-cluster tree node.
     * Uses fh.regionToVoxelMask(supervoxelIndices).
     *
     * @param {MetaClusterNode} node
     * @returns {Uint8Array}
     */
    buildMetaClusterMask(node) {
        /** @type {Set<number>} */
        const svSet = new Set();

        /**
         * @param {MetaClusterNode} n
         */
        const collectSupervoxels = (n) => {
            if (n.isLeaf && n.metaIndex !== undefined && n.metaIndex >= 0) {
                const mc = this.metaClusters[n.metaIndex];
                if (!mc) return;
                for (const ri of mc.regionIndices) {
                    const regionSet = this.regionSVSets[ri];
                    if (!regionSet) continue;
                    for (const sv of regionSet) {
                        svSet.add(sv);
                    }
                }
            }
            if (n.children) {
                for (const c of n.children) {
                    collectSupervoxels(c);
                }
            }
        };

        collectSupervoxels(node);

        if (svSet.size === 0) {
            console.warn("MetaClusterBuilder.buildMetaClusterMask: Node resulted in empty mask", node);
            const totalVoxels = this.fh.dims.width * this.fh.dims.height * this.fh.dims.depth;
            return new Uint8Array(totalVoxels);
        }

        const svArray = Array.from(svSet);
        return this.fh.regionToVoxelMask(svArray);
    }
}