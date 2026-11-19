//! Barnes–Hut octree for O(n log n) gravity.
//!
//! Bodies are bucketed into octants until each cell holds a single body
//! (a leaf) or shrinks below `MIN_NODE_SIZE` (a collapsed leaf). Every
//! internal node stores the total mass and center of mass of its subtree,
//! letting a distant cluster be treated as a single point mass.

use super::*;

/// Cells smaller than this are treated as coincident: further subdivision
/// would not separate bodies, so they collapse into one high-multiplicity leaf.
pub(super) const MIN_NODE_SIZE: f64 = 1e-9;

/// An octree of depth `d` needs at most `7·d + 1` pending nodes on the
/// DFS stack; `MIN_NODE_SIZE` bounds effective depth (~50) well below this.
const MAX_STACK: usize = 1024;

/// One octree cell. `body` encodes the node kind:
///   `>= 0` — leaf holding a single body index,
///   `-1`   — internal node with children,
///   `-2`   — collapsed leaf of coincident bodies (treated as one mass).
pub(super) struct Node {
    pub(super) cx: f64,
    pub(super) cy: f64,
    pub(super) cz: f64,
    pub(super) size: f64,
    pub(super) m: f64,
    pub(super) com_x: f64,
    pub(super) com_y: f64,
    pub(super) com_z: f64,
    pub(super) body: i32,
    /// Child node indices in `Tree::nodes`, or `-1` when absent. Octant `q`
    /// is high-x if bit 0 is set, high-y if bit 1, high-z if bit 2, so the
    /// first four are the old quadtree's order, at low z.
    pub(super) children: [i32; 8],
}

impl Node {
    pub(super) fn empty() -> Self {
        Self {
            cx: 0.0,
            cy: 0.0,
            cz: 0.0,
            size: 0.0,
            m: 0.0,
            com_x: 0.0,
            com_y: 0.0,
            com_z: 0.0,
            body: -1,
            children: [-1; 8],
        }
    }
}

/// Flat arena of all nodes. Index 0 is the root after `build` returns.
pub struct Tree {
    pub(super) nodes: Vec<Node>,
}

/// Unstable partition of `idx` so that all elements satisfying `pred` come
/// first. Returns the split point. Works in place with a single pass.
pub(super) fn partition_in_place(idx: &mut [u32], mut pred: impl FnMut(u32) -> bool) -> usize {
    let mut lo = 0;
    let mut hi = idx.len();
    while lo < hi {
        if pred(idx[lo]) {
            lo += 1;
        } else {
            hi -= 1;
            idx.swap(lo, hi);
        }
    }
    lo
}

impl Tree {
    /// Build the full tree. The root cell is a square covering the body
    /// bounding box with margin, so all bodies fall strictly inside.
    pub fn build(b: &Bodies) -> Self {
        let n = b.len();
        if n == 0 {
            return Tree { nodes: Vec::new() };
        }
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..n {
            xmin = xmin.min(b.x[i]);
            xmax = xmax.max(b.x[i]);
            ymin = ymin.min(b.y[i]);
            ymax = ymax.max(b.y[i]);
            zmin = zmin.min(b.z[i]);
            zmax = zmax.max(b.z[i]);
        }
        let cx = (xmin + xmax) * 0.5;
        let cy = (ymin + ymax) * 0.5;
        let cz = (zmin + zmax) * 0.5;
        let size = (xmax - xmin).max(ymax - ymin).max(zmax - zmin).max(1.0) * 0.5 + 1.0;
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let nodes = build_subtree(b, &mut idx, cx, cy, cz, size, 0);
        Tree { nodes }
    }

    /// Acceleration on body `i` plus a suggested timestep limit.
    ///
    /// Walks the tree with an explicit stack, accepting a node either as a
    /// distant cluster (opened by the `theta` criterion) or a single-body
    /// leaf. The returned timestep limit is the tightest of the local
    /// dynamical time and the pair crossing time over all accepted nodes.
    #[inline]
    pub(super) fn accel_for(&self, b: &Bodies, i: usize) -> (f64, f64, f64, f64) {
        let xi = b.x[i];
        let yi = b.y[i];
        let zi = b.z[i];
        let mi = b.m[i];
        let vxi = b.vx[i];
        let vyi = b.vy[i];
        let vzi = b.vz[i];
        let theta2 = BH_THETA * BH_THETA;

        let mut stack = [0u32; MAX_STACK];
        let mut sp = 1usize;
        let (mut sx, mut sy, mut sz) = (0.0f64, 0.0f64, 0.0f64);

        // Running maxima of 1/t_dyn²/G and v²/d² over accepted nodes;
        // converted to a timestep by `dt_limit` at the end.
        let mut s_max = 0.0f64;
        let mut w_max = 0.0f64;

        while sp > 0 {
            sp -= 1;
            let n = &self.nodes[stack[sp] as usize];
            // Empty quadrants carry zero mass: skip them so they neither
            // cost traversal nor feed (m_i + 0)/d³ into the timestep limit.
            if n.m == 0.0 {
                continue;
            }

            // Leaves point to a real body position; internal nodes use the
            // aggregated center of mass.
            let leaf = n.body >= 0;
            let (nx, ny, nz) = if leaf {
                let j = n.body as usize;
                (b.x[j], b.y[j], b.z[j])
            } else {
                (n.com_x, n.com_y, n.com_z)
            };

            let dx = nx - xi;
            let dy = ny - yi;
            let dz = nz - zi;
            let d2 = dx * dx + dy * dy + dz * dz + SOFT2;

            // Same acceptance rule for the serial and parallel build so the
            // opening criterion is independent of how the tree was assembled.
            let accept = if leaf {
                if n.body as usize == i {
                    continue;
                }
                true
            } else if n.body == -2 {
                // Collapsed coincident cluster: only safe to open the
                // softening core, otherwise its members act as one mass.
                if d2 < 4.0 * SOFT2 {
                    continue;
                }
                true
            } else {
                let s = n.size + n.size;
                s * s < theta2 * d2
            };

            if accept {
                // 1/d³ as  inv_d2 · sqrt(inv_d2)  with  inv_d2 = 1/d².
                // Reusing inv_d2 makes the pair timestep terms (below)
                // multiply-only.
                let inv_d2 = 1.0 / d2;
                let inv = inv_d2 * inv_d2.sqrt();
                let f = G * n.m * inv;
                sx += f * dx;
                sy += f * dy;
                sz += f * dz;

                // 1/t_dyn² / G = (mi + mj) / d³
                let s = (mi + n.m) * inv;
                if s > s_max {
                    s_max = s;
                }

                if leaf {
                    let j = n.body as usize;
                    let dvx = b.vx[j] - vxi;
                    let dvy = b.vy[j] - vyi;
                    let dvz = b.vz[j] - vzi;
                    // v² / d²
                    let w = (dvx * dvx + dvy * dvy + dvz * dvz) * inv_d2;
                    if w > w_max {
                        w_max = w;
                    }
                }
            } else {
                // Push children in fixed order; popping reverses it, so
                // force sums always reduce in the same order.
                for q in 0..8 {
                    let c = n.children[q];
                    if c >= 0 {
                        debug_assert!(sp < stack.len(), "BH traversal stack overflow");
                        if sp < stack.len() {
                            stack[sp] = c as u32;
                            sp += 1;
                        }
                    }
                }
            }
        }
        (sx, sy, sz, dt_limit(s_max, w_max))
    }
}

/// Recursively build one subtree. Child indices are initially local to the
/// returned vector; the caller fixes them up when concatenating.
///
/// Moments are computed bottom-up from the child roots, so each body
/// contributes to the reduction only once per level it passes through.
pub(super) fn build_subtree(
    b: &Bodies,
    indices: &mut [u32],
    cx: f64,
    cy: f64,
    cz: f64,
    size: f64,
    depth: usize,
) -> Vec<Node> {
    let n_here = indices.len();

    // Empty octant: no node. The parent keeps its child slot at -1.
    if n_here == 0 {
        return Vec::new();
    }

    // Single body: leaf.
    if n_here == 1 {
        let i = indices[0] as usize;
        let m = b.m[i];
        let (com_x, com_y, com_z) = if m > 0.0 {
            (b.x[i], b.y[i], b.z[i])
        } else {
            (cx, cy, cz)
        };
        return vec![Node {
            cx,
            cy,
            cz,
            size,
            m,
            com_x,
            com_y,
            com_z,
            body: indices[0] as i32,
            children: [-1; 8],
        }];
    }

    // Coincident bodies below the resolution limit: collapse into one
    // leaf carrying the group's aggregate mass and center of mass.
    if size < MIN_NODE_SIZE {
        let mut total_m = 0.0;
        let mut com_x = 0.0;
        let mut com_y = 0.0;
        let mut com_z = 0.0;
        for &i in indices.iter() {
            let i = i as usize;
            total_m += b.m[i];
            com_x += b.m[i] * b.x[i];
            com_y += b.m[i] * b.y[i];
            com_z += b.m[i] * b.z[i];
        }
        let (com_x, com_y, com_z) = if total_m > 0.0 {
            (com_x / total_m, com_y / total_m, com_z / total_m)
        } else {
            (cx, cy, cz)
        };
        return vec![Node {
            cx,
            cy,
            cz,
            size,
            m: total_m,
            com_x,
            com_y,
            com_z,
            body: -2,
            children: [-1; 8],
        }];
    }

    // Split the slice into eight octants by partitioning z, then y within
    // each half, then x within each quarter. O(n_here) per level. Octant
    // order is x | y << 1 | z << 2.
    let nz = partition_in_place(indices, |i| b.z[i as usize] < cz);
    let (zlo, zhi) = indices.split_at_mut(nz);
    let mut slices: Vec<&mut [u32]> = Vec::with_capacity(8);
    for half in [zlo, zhi] {
        let ny = partition_in_place(half, |i| b.y[i as usize] < cy);
        let (ylo, yhi) = half.split_at_mut(ny);
        for quarter in [ylo, yhi] {
            let nx = partition_in_place(quarter, |i| b.x[i as usize] < cx);
            let (xlo, xhi) = quarter.split_at_mut(nx);
            slices.push(xlo);
            slices.push(xhi);
        }
    }
    let h = size * 0.5;
    let centre = |q: usize| {
        (
            if q & 1 == 0 { cx - h } else { cx + h },
            if q & 2 == 0 { cy - h } else { cy + h },
            if q & 4 == 0 { cz - h } else { cz + h },
        )
    };

    // Parallelise only the top levels: below that, the dispatch cost
    // exceeds the work in any one octant.
    let par = depth < 3 && n_here >= 512;
    let trees: Vec<Vec<Node>> = if par {
        use rayon::prelude::*;
        slices
            .into_par_iter()
            .enumerate()
            .map(|(q, sl)| {
                let (qx, qy, qz) = centre(q);
                build_subtree(b, sl, qx, qy, qz, h, depth + 1)
            })
            .collect()
    } else {
        slices
            .into_iter()
            .enumerate()
            .map(|(q, sl)| {
                let (qx, qy, qz) = centre(q);
                build_subtree(b, sl, qx, qy, qz, h, depth + 1)
            })
            .collect()
    };

    // Aggregate moments from the child subtree roots.
    let mut total_m = 0.0;
    let mut com_x = 0.0;
    let mut com_y = 0.0;
    let mut com_z = 0.0;
    for t in trees.iter() {
        if let Some(root) = t.first() {
            total_m += root.m;
            com_x += root.m * root.com_x;
            com_y += root.m * root.com_y;
            com_z += root.m * root.com_z;
        }
    }
    let (com_x, com_y, com_z) = if total_m > 0.0 {
        (com_x / total_m, com_y / total_m, com_z / total_m)
    } else {
        (cx, cy, cz)
    };

    // Concatenate: parent first, then each child in octant order with
    // its internal indices shifted by the running offset.
    let mut out = Vec::with_capacity(1 + trees.iter().map(|t| t.len()).sum::<usize>());
    out.push(Node {
        cx,
        cy,
        cz,
        size,
        m: total_m,
        com_x,
        com_y,
        com_z,
        body: -1,
        children: [-1; 8],
    });
    for (q, mut sub) in trees.into_iter().enumerate() {
        if sub.is_empty() {
            continue;
        }
        let offset = out.len() as i32;
        out[0].children[q] = offset;
        for node in sub.iter_mut() {
            for c in node.children.iter_mut() {
                if *c >= 0 {
                    *c += offset;
                }
            }
        }
        out.append(&mut sub);
    }
    out
}

/// Combine the largest `1/t_dyn²/G` and `v²/d²` seen into a step limit.
#[inline]
pub(super) fn dt_limit(s_max: f64, w_max: f64) -> f64 {
    let t_dyn = (1.0 / (G * s_max)).sqrt();
    let t_vel = (1.0 / w_max).sqrt();
    (ETA_GRAV * t_dyn).min(ETA_VEL * t_vel)
}
