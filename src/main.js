const DATASETS = {
    Aneurism: "data/aneurism.vol",
    Neghip: "data/neghip.vol",
    Fuel: "data/fuel.vol",
    Nucleon: "data/nucleon.vol",
    Silicium: "data/silicium.vol",
    Skull: "data/skull.vol",
    Shockwave: "data/shockwave.vol",
};

const SUPERVOXEL_SIZES = [512, 2197, 4096, 8000, 16384];

class Main {
    animate() {
        this.render();
        requestAnimationFrame(this.animate.bind(this));
    }

    async start() {
        await this.initializeWebGPU();
        this.initializeBuffers();
        this.initializeLayouts();
        await this.initializeVolume();
        this.initializePipelines();
        this.initializeGUI();
        await this.SLIC();
        this.fhClustering();
        this.initializeAttachments();
        this.animate();
    }

    /**
     * @param {any} gui 
     * @param {HTMLCanvasElement} canvas 
     */
    constructor(gui, canvas) {
        this.gui = gui;
        this.canvas = canvas;

        // Rotation state
        this.rotationX = 0;
        this.rotationY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Visualization parameters
        this.highlightFactor = 1.0;
        this.wholeVolume = true;

        //Dataset parameters
        this.currentDataset = DATASETS.Aneurism;

        // Slic parameters
        this.slicParams = {
            superVoxelSize: 4096,
            compactness: 0.18,
            numIterations: 10 //4-10
        };

        // Meta-clustering parameters
        this.metaParams = {
            maxBranching: 3,
            distThreshold: 0.3
        };
        this.onTreeChange = null;

        this.setupMouseControls();
    }

    setupMouseControls() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;

            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            this.rotationY += deltaX * 0.01;
            this.rotationX += deltaY * 0.01;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });

        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
        });
    }

    async initializeWebGPU() {
        if (!this.gpu) {
            this.gpu = navigator.gpu;
            if (!this.gpu) {
                const message = "WebGPU is not supported in your browser. Please use/update Chrome or Edge.";
                alert(message);
                throw new Error(message);
            }
            console.log("Hooray! WebGPU is supported in your browser!");
            this.adapter = await this.gpu.requestAdapter();

            // Check adapter limits
            const maxInvocations = this.adapter.limits.maxComputeInvocationsPerWorkgroup;
            console.log("Max compute invocations per workgroup: ", maxInvocations);

            // Request device with higher limits
            this.device = await this.adapter.requestDevice({
                requiredLimits: {
                    maxComputeInvocationsPerWorkgroup: Math.min(512, maxInvocations)
                }
            });

            this.context = this.canvas.getContext("webgpu");
            this.context.configure({
                device: this.device,
                format: this.gpu.getPreferredCanvasFormat()
            });
        }
    }


    // Create uniform buffer
    initializeBuffers() {
        // Quad positions (clip space) and UVs
        const quad = new Float32Array([
            // x, y, u, v
            -1, -1, 0, 0,
            1, -1, 1, 0,
            -1, 1, 0, 1,
            1, 1, 1, 1,
        ]);
        const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);

        this.sliceVB = this.device.createBuffer({
            size: quad.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.sliceVB, 0, quad);

        this.sliceIB = this.device.createBuffer({
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.sliceIB, 0, indices);

        // Uniform buffer - increase size for additional parameters
        this.uniformBuffer = this.device.createBuffer({
            size: 96, // 16*4 (matrix) + 5*4 (params) = 84, padded
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }


    // create 3D texture, write Texture, create sampler, setup renderpassdescriptor
    async initializeVolume() {
        // Load and normalize the volume data
        //const volume = await VolumeLoader.loadVol("data/shockwave.vol", {});
        //const volume = await VolumeLoader.loadVol("data/aneurism.vol", {});
        const volume = await VolumeLoader.loadVol(this.currentDataset, {});


        const { width, height, depth } = volume.dims;
        this.volumeDims = volume.dims; // Store for later use in SLIC
        this.volumeData = volume.data; // Store for later use in SLIC
        const src = volume.raw;
        this.totalVoxels = width * height * depth;

        console.log("Voxel bytes:", src.byteLength, "expected:", this.totalVoxels);

        if (src.byteLength !== this.totalVoxels) {
            throw new Error("Voxel byte length mismatch after loader fix.");
        }

        const bytesPerRow = width; //r8unorm
        const rowsPerImage = height;

        // Create a 3D texture on the GPU
        this.volumeTexture = this.device.createTexture({
            size: {
                width,
                height,
                depthOrArrayLayers: depth
            },
            dimension: "3d",
            format: "r8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // Upload raw data into the 3D texture
        this.device.queue.writeTexture(
            { texture: this.volumeTexture },
            src,
            {
                offset: 0,
                bytesPerRow,
                rowsPerImage,
            },
            {
                width,
                height,
                depthOrArrayLayers: depth,
            }
        );
        // Create a sampler for use in raymarcher shader
        this.volumeSampler = this.device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
        });
        console.log("3D volume texture + sampler created");

        // Setup render pass descriptor
        this.renderPassDescriptor = {
            colorAttachments: [{
                view: undefined, // set in render loop
                clearValue: [0.0, 0.0, 0.0, 1.0],
                loadOp: 'clear',
                storeOp: 'store'
            }]
        };
    }

    async reloadVolume() {
        console.log("Reloading dataset:", this.currentDataset);

        // Destroy old GPU resources
        if (this.volumeTexture) this.volumeTexture.destroy();
        if (this.regionMaskTexture) this.regionMaskTexture.destroy();

        await this.initializeVolume();
        await this.SLIC();
        this.fhClustering();
    }

    // create bindgrouplayout
    initializeLayouts() {
        // Layout for SLIC visualization + region mask
        this.slicBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } },
            ]
        });
    }

    // initialize pipeline
    initializePipelines() {
        // SLIC 3D visualization pipeline
        const slicModule = this.device.createShaderModule({ code: SHADERS.volVis });
        this.slicPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.slicBindGroupLayout] }),
            vertex: {
                module: slicModule,
                entryPoint: "vert_main",
                buffers: [{
                    arrayStride: 4 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x2" },
                        { shaderLocation: 1, offset: 8, format: "float32x2" },
                    ]
                }]
            },
            fragment: {
                module: slicModule,
                entryPoint: "frag_main",
                targets: [{ format: this.gpu.getPreferredCanvasFormat() }]
            },
            primitive: { topology: "triangle-list", cullMode: "none" }
        });
    }

    async SLIC() {
        console.log("Initializing 3D SLIC...");

        this.slicManager = new SlicManager(
            this.device,
            this.volumeTexture,
            this.volumeDims
        );
        // targetSupervoxels = voxels per supervoxel.
        const targetSupervoxels = this.slicParams.superVoxelSize; //512: Tooth, 2197: Tomato,Pancreas; 4096: Knee MRI, Chest CT, Abdominal CT
        const idealSpacing = Math.cbrt(this.totalVoxels / targetSupervoxels);
        const dynamicSpacing = Math.max(4, Math.round(idealSpacing));

        console.log("SLIC supervoxel spacing:", idealSpacing);
        // Configure SLIC parameters
        this.slicManager.gridSpacing = dynamicSpacing; // Supervoxel size/spacing
        this.slicManager.compactness = this.slicParams.compactness; // Balance intensity vs spatial
        this.slicManager.numIterations = this.slicParams.numIterations;

        // Run SLICss
        this.labels = await this.slicManager.run();
        console.log("SLIC labels array length:", this.labels.length);
        console.log("SLIC produced:", this.slicManager.totalClusters, "supervoxels/clusters");
    }

    fhClustering() {
        console.log("Initializing FH clustering...");
        // ---FH clustering ---
        this.fh = new FHClustering(this.labels, this.volumeData, this.volumeDims);

        //const kValues = [0.1, 0.3, 1, 3, 10, 30]; // χ² + 26-neighbors
        //const kValues = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        // this.fhRegions = this.fh.runExhaustive(kValues);
        this.fhRegions = this.fh.runExhaustiveAuto(16); // 16 od 24

        console.log("FH produced:", this.fhRegions.size, "regions");

        const MIN_REGION_VOXELS = 2000;

        const filtered = new Map();
        for (const [key, region] of this.fhRegions.entries()) {
            let voxCount = 0;
            for (const sv of region.supervoxels) {
                voxCount += this.fh.features[sv].voxelCount;
            }
            if (voxCount >= MIN_REGION_VOXELS) {
                filtered.set(key, region);
            }
        }
        console.log("Filtered FH regions:", filtered.size, "(from", this.fhRegions.size, ")");
        
        this.fhRegions = filtered;

        this.totalVoxels = this.volumeDims.width * this.volumeDims.height * this.volumeDims.depth;

        // --- Meta-cluster tree ---
        this.buildMetaClusters();
    }


    buildMetaClusters() {
        console.log("Building meta-cluster tree...");
        try {
            // 1. Define Parameters
            const DIST_THRESHOLD = this.metaParams.distThreshold; // how "different" two regions can be while still belonging to the same meta-cluster.

            // 2. Build the Tree
            this.metaBuilder = new MetaClusterBuilder(this.fh, this.fhRegions);

            this.metaTree = this.metaBuilder.build(DIST_THRESHOLD, this.metaParams.maxBranching);

            console.log("Meta-cluster tree built. Root:", this.metaTree.root);

            if (this.onTreeChange && this.metaTree.root) {
                this.onTreeChange(this.metaTree.root);
            }

            // 3. (Optional) Highlight the root initially so the view isn't empty
            if (this.metaTree.root) {
                this.highlightMetaCluster(this.metaTree.root);
            }

        } catch (e) {
            console.warn("Meta-cluster tree construction failed:", e);
            this.metaBuilder = null;
            this.metaTree = null;
        }
    }

    /**
     * Highlight a meta-cluster node (union of all its leaf regions).
     * @param {MetaClusterNode} node
     */
    highlightMetaCluster(node) {
        if (!this.metaBuilder || !node) {
            console.warn("No meta-cluster builder or node to highlight");
            return;
        }

        // Build a 0/255 mask from meta-cluster (union of leaves)
        const mask255 = this.metaBuilder.buildMetaClusterMask(node);

        // Create / update region mask texture
        this.regionMaskTexture = this.fh.createRegionMaskTexture(this.device, mask255);

        // Recreate the SLIC visualization bind group with the new mask
        this.initializeBindGroups();
    }


    // create bindgroup
    initializeBindGroups() {
        this.slicBindGroup = this.device.createBindGroup({
            layout: this.slicBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: this.volumeTexture.createView() },
                { binding: 2, resource: this.volumeSampler },
                { binding: 3, resource: this.regionMaskTexture.createView() },
            ]
        });
    }

    initializeAttachments() {
        // Color attachment to draw to
        /** @type {GPURenderPassColorAttachment} */
        this.colorAttachment = {
            view: null, // Will be set in render()
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            storeOp: "store"
        };
    }


    initializeGUI() {
        const datasetParams = {
            Dataset: "Aneurism"
        };

        this.gui.add(datasetParams, "Dataset", Object.keys(DATASETS))
            .name("Dataset")
            .onChange(async (/** @type {string} */ value) => {
                // @ts-ignore
                this.currentDataset = DATASETS[value];
                await this.reloadVolume();
            });

        const slicFolder = this.gui.addFolder("SLIC Parameters");
        const superVoxelChoices = [216, 512, 2197, 4096, 8000, 12167];
        this.slicParams.superVoxelSize = superVoxelChoices[1];
        slicFolder.add(this.slicParams, 'superVoxelSize', superVoxelChoices)
            .name('Supervoxel Size')
            .onChange(async () => {
                await this.SLIC();
                this.fhClustering();
                this.initializeAttachments();
                this.animate();
            });
        slicFolder.add(this.slicParams, 'compactness').min(0.01).max(1.0).step(0.01)
            .name('Compactness')
            .onChange(async () => {
                await this.SLIC();
                this.fhClustering();
                this.initializeAttachments();
                this.animate();
            });

        const folder = this.gui.addFolder("Meta Clustering");
        folder.add(this.metaParams, 'maxBranching').min(1).max(10).step(1)
            .name('Max Branching')
            .onChange(() => {
                this.buildMetaClusters();
            });
        folder.add(this.metaParams, 'distThreshold').min(0.0).max(1.0).step(0.05)
            .name('Distance Threshold')
            .onChange(() => {
                this.buildMetaClusters();
            });
        folder.open();

        const visFolder = this.gui.addFolder("Visualization Settings");

        // Slider for Highlight Factor (0.0 to 1.0)
        visFolder.add(this, 'highlightFactor', 0.0, 1.0)
            .name('Highlight Intensity')
            .step(0.1) // Steps of 0.1, matching your old '1' key logic
            .onChange((/** @type {number} */ value) => {
                console.log("Highlight factor changed:", value);
            });

        // Checkbox for Show Only Highlight
        visFolder.add(this, 'wholeVolume')
            .name('Whole volume')
            .onChange((/** @type {boolean} */ value) => {
                console.log("Whole volume:", value);
            });

        visFolder.open();
    }

    async render() {
        // Create rotation matrix
        const rotationMatrix = new THREE.Matrix4();
        rotationMatrix.makeRotationFromEuler(
            new THREE.Euler(this.rotationX, this.rotationY, 0, 'XYZ')
        );

        // Inverse view matrix (camera transform)
        const inverseViewMatrix = rotationMatrix
        const windowWidth = 1.0;
        const windowLevel = 0.5;

        const uniformData = new Float32Array(20);// 16 matrix + 4 floats = 20
        uniformData.set(inverseViewMatrix.elements, 0);
        uniformData[16] = windowWidth;
        uniformData[17] = windowLevel;
        uniformData[18] = this.highlightFactor ?? 1.0;
        uniformData[19] = this.wholeVolume ? 0.0 : 1.0;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // render pass setup
        this.colorAttachment.view = this.context.getCurrentTexture().createView();

        const commandEncoder = this.device.createCommandEncoder();
        {
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [this.colorAttachment]
            });

            renderPass.setVertexBuffer(0, this.sliceVB);
            renderPass.setIndexBuffer(this.sliceIB, "uint16");
            renderPass.setPipeline(this.slicPipeline);
            renderPass.setBindGroup(0, this.slicBindGroup);
            renderPass.drawIndexed(6);
            renderPass.end();
        }
        this.device.queue.submit([commandEncoder.finish()]);
    }
}