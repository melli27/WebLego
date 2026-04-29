/** @type {import("d3")} */

/**
 * @typedef {Object} MetaClusterData
 * @property {number|string} id
 * @property {number} voxelCount
 * @property {MetaClusterData[]} [children]
 * @property {boolean} [isLeaf]
 * @property {number} [metaIndex]
 */

/**
 * @typedef {Object} TreeNode
 * @property {MetaClusterData} data
 * @property {TreeNode[]} [children]
 * @property {TreeNode[]} [_children] Stashed children for collapse
 * @property {TreeNode} [parent]
 * @property {number} x
 * @property {number} y
 * @property {number} x0 Previous x for animation
 * @property {number} y0 Previous y for animation
 * @property {number} depth
 * @property {number} [id]
 */

class MetaClusterTreeViz {
  /**
   * @param {string} containerId
   * @param {MetaClusterData} rootData
   * @param {function(MetaClusterData): void} onNodeClick
   */
  constructor(containerId, rootData, onNodeClick) {
    this.containerId = containerId;
    this.data = rootData;
    this.onNodeClick = onNodeClick;
    this.margin = { top: 20, right: 90, bottom: 30, left: 90 };
    this.duration = 500;

    /** @type {any} */
    this.root = null; // Will be initialized in init()

    // Counter for node IDs
    this.i = 0;

    this.init();
  }

  init() {
    const container = document.querySelector(this.containerId);
    if (!container) return;

    const width = container.clientWidth - this.margin.left - this.margin.right || 800;
    //const height = 800; 
    const height = container.clientHeight || 800;

    this.svg = d3.select(this.containerId).append("svg")
      .attr("width", "100%")
      .attr("height", height)
      /** @param {any} e */
      .call(d3.zoom().on("zoom", (e) => {
        this.svg.attr("transform", e.transform);
      }))
      .append("g")
      //.attr("transform", "translate(" + this.margin.left + "," + this.margin.top + ")");
      .attr("transform", "translate(" + this.margin.left + "," + (height / 2) + ")");

    this.treemap = d3.tree().nodeSize([25, 150]);

    // Convert raw data to D3 hierarchy
    this.root = /** @type {any} */ (d3.hierarchy(this.data, (/** @type {MetaClusterData} */ d) => d.children));

    this.root = this.root.children[0]; // Skip the artificial root
    this.root.x0 = height / 2;
    this.root.y0 = 0;

    // Collapse all nodes below depth 1 initially
    if (this.root.children) {
      this.root.children.forEach((/** @type {TreeNode} */ d) => this.collapse(d));
    }

    this.update(this.root);
  }

  /**
   * @param {TreeNode} d 
   */
  collapse(d) {
    if (d.children) {
      d._children = d.children;
      d._children.forEach((/** @type {TreeNode} */ c) => this.collapse(c));
      d.children = null;
    }
  }

  /**
   * @param {TreeNode} source 
   */
  update(source) {
    // Assigns x and y position for the nodes
    const treeData = this.treemap(this.root);

    // Compute the new tree layout.
    /** @type {TreeNode[]} */
    // @ts-ignore
    const nodes = treeData.descendants();

    /** @type {any[]} */
    const links = treeData.descendants().slice(1);

    // Normalize for fixed-depth.
    nodes.forEach(d => { d.y = d.depth * 100; });

    /* NODES */
    //
    // Update the nodes...
    const node = this.svg.selectAll('g.node')
      // @ts-ignore
      .data(nodes, (/** @type {TreeNode} */ d) => d.id || (d.id = ++this.i));

    // Enter any new nodes at the parents previous position.
    const nodeEnter = node.enter().append('g')
      .attr('class', 'node')
      .attr('transform', (/** @type {any} */ d) => "translate(" + source.y0 + "," + source.x0 + ")")
      .on('click', (/** @type {Event} */ e, /** @type {TreeNode} */ d) => this.click(e, d));

    // Add Circle for the nodes
    nodeEnter.append('circle')
      .attr('class', 'node')
      .attr('r', 1e-6)
      .style("fill", (/** @type {TreeNode} */ d) => d._children ? "#888" : "#333");

    // Add Labels for the nodes
    nodeEnter.append('text')
      .attr("dy", ".35em")
      .attr("x", (/** @type {TreeNode} */ d) => d.children || d._children ? -13 : 13)
      .attr("text-anchor", (/** @type {TreeNode} */ d) => d.children || d._children ? "end" : "start")
      .text((/** @type {TreeNode} */ d) => {
        const vox = d.data.voxelCount ? (d.data.voxelCount / 1000).toFixed(1) + 'k' : '0';
        // @ts-ignore
        return `ID:${d.data.id} (${vox})`;
      });

    // UPDATE
    const nodeUpdate = nodeEnter.merge(node);

    // Transition to the proper position for the node
    nodeUpdate.transition()
      .duration(this.duration)
      .attr('transform', (/** @type {TreeNode} */ d) => "translate(" + d.y + "," + d.x + ")");

    // Update the circle attributes
    nodeUpdate.select('circle.node')
      .attr('r', 6)
      .style("fill", (/** @type {TreeNode} */ d) => d._children ? "#aaa" : "#333")
      .attr('cursor', 'pointer');

    // Remove any exiting nodes
    const nodeExit = node.exit().transition()
      .duration(this.duration)
      .attr('transform', (/** @type {any} */ d) => "translate(" + source.y + "," + source.x + ")")
      .remove();

    nodeExit.select('circle').attr('r', 1e-6);

    /* LINKS */
    //
    // Update the links
    const link = this.svg.selectAll('path.link')
      // @ts-ignore
      .data(links, (/** @type {TreeNode} */ d) => d.id);

    // Enter any new links at the parents previous position.
    const linkEnter = link.enter().insert('path', "g")
      .attr("class", "link")
      .attr('d', (/** @type {any} */ d) => {
        const o = { x: source.x0, y: source.y0 };
        return this.diagonal(o, o);
      });

    const linkUpdate = linkEnter.merge(link);

    // Transition back to the parent element position
    linkUpdate.transition()
      .duration(this.duration)
      .attr('d', (/** @type {TreeNode} */ d) => this.diagonal(d, /** @type {TreeNode} */(d.parent)));

    // Remove any exiting links
    link.exit().transition()
      .duration(this.duration)
      .attr('d', (/** @type {any} */ d) => {
        const o = { x: source.x, y: source.y };
        return this.diagonal(o, o);
      })
      .remove();

    // Store the old positions for transition.
    nodes.forEach(d => {
      d.x0 = d.x;
      d.y0 = d.y;
    });
  }

  /**
   * Creates a curved path from parent to child
   * @param {{x: number, y: number}} s 
   * @param {{x: number, y: number}} d 
   */
  diagonal(s, d) {
    // If parent is null (root), path to itself
    if (!d) d = s;

    return `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`;
  }

  /**
   * Toggle children on click
   * @param {Event} event 
   * @param {TreeNode} d 
   */
  click(event, d) {
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children = d._children;
      d._children = null;
    }

    // Highlight logic
    this.svg.selectAll("circle").classed("selected", false);

    const target = /** @type {Element} */ (event.currentTarget);
    d3.select(target).select("circle").classed("selected", true);

    if (this.onNodeClick) {
      this.onNodeClick(d.data);
    }

    this.update(d);
  }
}