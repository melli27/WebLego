/**
 * 3D SLIC Supervoxel Segmentation Manager
 */
class SlicManager {
    /**
     * @param {GPUDevice} device
     * @param {GPUTexture} volumeTexture
     * @param {{width: number, height: number, depth: number}} dims
     */
    constructor(device, volumeTexture, dims) {
        this.device = device;
        this.volumeTexture = volumeTexture;
        this.dims = dims;
        
        // SLIC parameters
        this.gridSpacing = 10;  // supervoxel size (adjust based on volume size)
        this.compactness = 0.1;  // balance between intensity and spatial distance
        this.numIterations = 10;  // 8-10
        
        this.initialized = false;
    }
    
    async initialize() {
        // Calculate number of clusters
        this.numClustersX = Math.ceil(this.dims.width / this.gridSpacing);
        this.numClustersY = Math.ceil(this.dims.height / this.gridSpacing);
        this.numClustersZ = Math.ceil(this.dims.depth / this.gridSpacing);
        this.totalClusters = this.numClustersX * this.numClustersY * this.numClustersZ;
        this.totalVoxels = this.dims.width * this.dims.height * this.dims.depth;
        
        console.log("SLIC created:", this.totalClusters, "clusters/supervoxels");
        
        this.createBuffers();
        this.createBindGroupLayouts();
        this.createPipelines();
        this.createBindGroups();        
        this.initialized = true;
    }
    
    createBuffers() {
        // Uniform buffer for SLIC parameters
        this.paramsBuffer = this.device.createBuffer({
            size: 48, // 10 u32/f32 values, padded to 48 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        
        // Cluster centers buffer (32 bytes per center)
        this.centersBuffer = this.device.createBuffer({
            size: this.totalClusters * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Cluster accumulators buffer (32 bytes per accumulator)
        this.accumulatorsBuffer = this.device.createBuffer({
            size: this.totalClusters * 32,
            usage: GPUBufferUsage.STORAGE
        });
        
        // Labels buffer (1 u32 per voxel)
        this.labelsBuffer = this.device.createBuffer({
            size: this.totalVoxels * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // Distances buffer (1 f32 per voxel)
        this.distancesBuffer = this.device.createBuffer({
            size: this.totalVoxels * 4,
            usage: GPUBufferUsage.STORAGE
        });
        
        // Readback buffer for labels
        this.labelsReadBuffer = this.device.createBuffer({
            size: this.totalVoxels * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }
    
    createBindGroupLayouts() {
        // Init layout
        this.initLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });
        
        // Assign layout
        this.assignLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });
        
        // Prepare layout (reset accumulators)
        this.prepareLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });
        
        // Accumulate layout
        this.accumulateLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });
        
        // Update centers layout
        this.updateLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ]
        });
    }
    
    createPipelines() {
        // Init pipeline
        this.initPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.initLayout] }),
            compute: {
                module: this.device.createShaderModule({ code: SHADERS.slicInit }),
                entryPoint: "main"
            }
        });
        
        // Assign pipeline
        this.assignPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.assignLayout] }),
            compute: {
                module: this.device.createShaderModule({ code: SHADERS.slicAssign }),
                entryPoint: "main"
            }
        });
        
        // Prepare pipeline
        this.preparePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.prepareLayout] }),
            compute: {
                module: this.device.createShaderModule({ code: SHADERS.slicUpdatePrepare }),
                entryPoint: "main"
            }
        });
        
        // Accumulate pipeline
        this.accumulatePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.accumulateLayout] }),
            compute: {
                module: this.device.createShaderModule({ code: SHADERS.slicAccumulate }),
                entryPoint: "main"
            }
        });
        
        // Update pipeline
        this.updatePipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.updateLayout] }),
            compute: {
                module: this.device.createShaderModule({ code: SHADERS.slicUpdateCenters }),
                entryPoint: "main"
            }
        });
    }
    
    createBindGroups() {
        const textureView = this.volumeTexture.createView();
        
        this.initBindGroup = this.device.createBindGroup({
            layout: this.initLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: textureView },
                { binding: 2, resource: { buffer: this.centersBuffer } },
            ]
        });
        
        this.assignBindGroup = this.device.createBindGroup({
            layout: this.assignLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: textureView },
                { binding: 2, resource: { buffer: this.centersBuffer } },
                { binding: 3, resource: { buffer: this.labelsBuffer } },
                { binding: 4, resource: { buffer: this.distancesBuffer } },
            ]
        });
        
        this.prepareBindGroup = this.device.createBindGroup({
            layout: this.prepareLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: this.accumulatorsBuffer } },
            ]
        });
        
        this.accumulateBindGroup = this.device.createBindGroup({
            layout: this.accumulateLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: textureView },
                { binding: 2, resource: { buffer: this.labelsBuffer } },
                { binding: 3, resource: { buffer: this.accumulatorsBuffer } },
            ]
        });
        
        this.updateBindGroup = this.device.createBindGroup({
            layout: this.updateLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: this.accumulatorsBuffer } },
                { binding: 2, resource: { buffer: this.centersBuffer } },
            ]
        });
    }
    
    /**
     * Update the parameters uniform buffer
     */
    updateParams(iteration = 0) {
        const params = new ArrayBuffer(48);
        const view = new DataView(params);
        
        view.setUint32(0, this.dims.width, true);
        view.setUint32(4, this.dims.height, true);
        view.setUint32(8, this.dims.depth, true);
        view.setUint32(12, this.gridSpacing, true);
        view.setFloat32(16, this.compactness, true);
        view.setUint32(20, this.numClustersX, true);
        view.setUint32(24, this.numClustersY, true);
        view.setUint32(28, this.numClustersZ, true);
        view.setUint32(32, iteration, true);
        view.setUint32(36, 0, true); // padding
        
        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
    }
    
    /**
     * Run the full SLIC algorithm
     * @returns {Promise<Uint32Array>} Label volume
     */
    async run() {
        if (!this.initialized) {
            await this.initialize();
        }
        
        const startTime = performance.now();
        
        // Initialize cluster centers
        this.updateParams(0);
        
        const initEncoder = this.device.createCommandEncoder();
        const initPass = initEncoder.beginComputePass();
        initPass.setPipeline(this.initPipeline);
        initPass.setBindGroup(0, this.initBindGroup);
        initPass.dispatchWorkgroups(
            Math.ceil(this.numClustersX / 4),
            Math.ceil(this.numClustersY / 4),
            Math.ceil(this.numClustersZ / 4)
        );
        initPass.end();
        this.device.queue.submit([initEncoder.finish()]);
        
        // Iterate: assign -> update
        for (let iter = 0; iter < this.numIterations; iter++) {
            this.updateParams(iter);
            
            const encoder = this.device.createCommandEncoder();
            
            // 1. Assign voxels to nearest centers
            const assignPass = encoder.beginComputePass();
            assignPass.setPipeline(this.assignPipeline);
            assignPass.setBindGroup(0, this.assignBindGroup);
            assignPass.dispatchWorkgroups(
                Math.ceil(this.dims.width / 8),
                Math.ceil(this.dims.height / 8),
                Math.ceil(this.dims.depth / 4)
            );
            assignPass.end();
            
            // 2. Reset accumulators
            const preparePass = encoder.beginComputePass();
            preparePass.setPipeline(this.preparePipeline);
            preparePass.setBindGroup(0, this.prepareBindGroup);
            preparePass.dispatchWorkgroups(Math.ceil(this.totalClusters / 64));
            preparePass.end();
            
            // 3. Accumulate voxel contributions
            const accumPass = encoder.beginComputePass();
            accumPass.setPipeline(this.accumulatePipeline);
            accumPass.setBindGroup(0, this.accumulateBindGroup);
            accumPass.dispatchWorkgroups(
                Math.ceil(this.dims.width / 8),
                Math.ceil(this.dims.height / 8),
                Math.ceil(this.dims.depth / 4)
            );
            accumPass.end();
            
            // 4. Update cluster centers
            const updatePass = encoder.beginComputePass();
            updatePass.setPipeline(this.updatePipeline);
            updatePass.setBindGroup(0, this.updateBindGroup);
            updatePass.dispatchWorkgroups(Math.ceil(this.totalClusters / 64));
            updatePass.end();
            
            this.device.queue.submit([encoder.finish()]);
        }
        
        // Wait for GPU to finish
        await this.device.queue.onSubmittedWorkDone();
        
        const endTime = performance.now();
        console.log(`SLIC completed in ${(endTime - startTime).toFixed(2)}ms`);
        
        return await this.readLabels();
    }
    
    /**
     * Read the label volume back to CPU
     * @returns {Promise<Uint32Array>}
     */
    async readLabels() {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(
            this.labelsBuffer, 0,
            this.labelsReadBuffer, 0,
            this.totalVoxels * 4
        );
        this.device.queue.submit([encoder.finish()]);
        
        await this.labelsReadBuffer.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(this.labelsReadBuffer.getMappedRange().slice(0));
        this.labelsReadBuffer.unmap();
        
        return data;
    }
}