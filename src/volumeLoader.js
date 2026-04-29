// Loader for 3D .vol files with an ASCII header and a byte matrix payload.
// - Fetches the file
// - Parses header to get width/height/depth (if possible)
// - Reads voxel data as Uint8
// - Normalizes intensities to [0, 1]
//
/**
 * @typedef {Object} VolumeDimensions
 * @property {number} width  - number of voxels in X
 * @property {number} height - number of voxels in Y
 * @property {number} depth  - number of voxels in Z
 */

/**
 * @typedef {Object} VolumeData
 * @property {Float32Array} data     - normalized values in [0, 1]
 * @property {VolumeDimensions} dims - volume dimensions
 * @property {ArrayBuffer} raw       - original raw data
 */

/**
 * @typedef {Object} LoadVolOptions
 * @property {number} [width]              - override width if header parsing fails
 * @property {number} [height]             - override height if header parsing fails
 * @property {number} [depth]              - override depth if header parsing fails
 */

const VolumeLoader = {
    /**
     * Load and normalize a 3D .vol file.
     *
     * @param {string} url - URL/path to the .vol file
     * @param {LoadVolOptions} [options]
     * @returns {Promise<VolumeData>}
     */
    async loadVol(url, options = {}) {
        const {
            width: overrideW,
            height: overrideH,
            depth: overrideD,
        } = options;

        // Fetch the file as an ArrayBuffer
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load volume from "${url}": ${response.status} ${response.statusText}`);
        }
        const buffer = await response.arrayBuffer();

        const totalBytes = buffer.byteLength;
        const probe = new Uint8Array(buffer, 0, Math.min(64 * 1024, totalBytes));
        const headerEnd = findHeaderEnd(probe);

        const headerText = new TextDecoder("ascii").decode(new Uint8Array(buffer, 0, headerEnd));

        // Extract dimensions from the header text
        let dims = {
            width: overrideW ?? null,
            height: overrideH ?? null,
            depth: overrideD ?? null,
        };

        if (!dims.width || !dims.height || !dims.depth) {
            const guessed = guessDimsFromHeader(headerText);
            dims = {
                width: dims.width ?? guessed.width,
                height: dims.height ?? guessed.height,
                depth: dims.depth ?? guessed.depth,
            };
        }

        if (!dims.width || !dims.height || !dims.depth) {
            throw new Error(
                "[VolumeLoader] Could not determine volume dimensions from header. " +
                "Please provide width/height/depth in options."
            );
        }

        const voxelCount = dims.width * dims.height * dims.depth;
        const expectedDataBytes = voxelCount; // Uint8 = 1 byte per voxel
        const available = totalBytes - headerEnd;

        if (available < expectedDataBytes) {
            console.warn(`[VolumeLoader] Data shorter than declared. Need ${expectedDataBytes}, have ${available}. Truncating dims.`);
        }
        if (available > expectedDataBytes) {
            console.warn(`[VolumeLoader] Extra trailing bytes (${available - expectedDataBytes}). Using first ${expectedDataBytes} after header.`);
        }

        const rawFull = new Uint8Array(buffer, headerEnd, expectedDataBytes);

        // Normalize to [0,1]
        const norm = new Float32Array(rawFull.length);
        for (let i = 0; i < rawFull.length; i++) {
            const v = rawFull[i];        // 0–255 byte
            norm[i] = v / 255.0;   // match r8unorm exactly
        }
        return {
            data: norm,
            dims: { width: dims.width, height: dims.height, depth: dims.depth },
            raw: rawFull.slice().buffer,
        };
    },
};

/**
 * @param {Uint8Array} bytes
 * @returns {number} Index after the header (start of binary data).
 */
function findHeaderEnd(bytes) {
    // Try CRLF terminator: "\r\n.\r\n"
    for (let i = 3; i < bytes.length; i++) {
        if (
            bytes[i - 3] === 13 /*\r*/ &&
            bytes[i - 2] === 10 /*\n*/ &&
            bytes[i - 1] === 46 /*'.'*/ &&
            bytes[i] === 13 /*\r*/ &&
            bytes[i + 1] === 10 /*\n*/
        ) {
            return i + 2; // after final \r\n
        }
    }
    // Try LF terminator: "\n.\n"
    for (let i = 2; i < bytes.length; i++) {
        if (bytes[i - 2] === 10 /*\n*/ && bytes[i - 1] === 46 /*'.'*/ && bytes[i] === 10 /*\n*/) {
            return i + 1;
        }
    }
    // Try line with single dot followed by CRLF or LF
    for (let i = 1; i < bytes.length; i++) {
        if (bytes[i - 1] === 46 /*'.'*/ && (bytes[i] === 10 /*\n*/ || bytes[i] === 13 /*\r*/)) {
            // Advance past optional CRLF pair
            let j = i + 1;
            if (bytes[i] === 13 /*\r*/ && bytes[j] === 10 /*\n*/) j++;
            return j;
        }
    }
    // Fallback: stop at first non-printable after a newline
    let lastNL = -1;
    for (let i = 0; i < bytes.length; i++) {
        const b = bytes[i];
        if (b === 10 /*\n*/) lastNL = i;
        const isPrintable = (b >= 32 && b <= 126) || b === 9 || b === 10 || b === 13;
        if (!isPrintable) return (lastNL >= 0) ? (lastNL + 1) : i;
    }
    return bytes.length;
}

/**
 * Very generic dimension guessing:
 * - Look for a line that contains 3 integers: "nx ny nz"
 * - If multiple candidates, take the first one
 *
 * @param {string} header
 * @returns {{width:number|null,height:number|null,depth:number|null}}
 */
function guessDimsFromHeader(header) {
    console.log("[VolumeLoader] Guessing dimensions from header...");
    const lines = header.split(/\r?\n/);

    let width = null;
    let height = null;
    let depth = null;

    for (const line of lines) {
        // X: 256
        let m = line.match(/^\s*X:\s*(\d+)/i);
        if (m) {
            width = parseInt(m[1], 10);
            continue;
        }

        // Y: 256
        m = line.match(/^\s*Y:\s*(\d+)/i);
        if (m) {
            height = parseInt(m[1], 10);
            continue;
        }

        // Z: 256
        m = line.match(/^\s*Z:\s*(\d+)/i);
        if (m) {
            depth = parseInt(m[1], 10);
            continue;
        }
    }

    if (width != null && height != null && depth != null) {
        console.log("[VolumeLoader] Parsed dimensions from header:", width, height, depth);
        return { width, height, depth };
    }

    // Fallback: nothing found
    return { width: null, height: null, depth: null };
}