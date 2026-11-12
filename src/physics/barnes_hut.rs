//! Barnes-Hut quadtree.

use super::*;

pub(super) const MIN_NODE_SIZE: f64 = 1e-9;

pub(super) struct Node {
    pub(super) cx: f64,
    pub(super) cy: f64,
    pub(super) size: f64,
    pub(super) m: f64,
    pub(super) com_x: f64,
    pub(super) com_y: f64,
    pub(super) body: i32,
    pub(super) children: [i32; 4],
}

impl Node {
    pub(super) fn empty() -> Self {
        Self {
            cx: 0.0,
            cy: 0.0,
            size: 0.0,
            m: 0.0,
            com_x: 0.0,
            com_y: 0.0,
            body: -1,
            children: [-1; 4],
        }
    }
}

pub struct Tree {
    pub(super) nodes: Vec<Node>,
}

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

// ---- Fix 5: parallel tree build. Each subtree is built independently
// ---- and concatenated at the root with child offsets patched in the
// ---- fixed q0..q3 order. The resulting node array is bit-identical to
// ---- the serial version.
impl Tree {
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
        for i in 0..n {
            xmin = xmin.min(b.x[i]);
            xmax = xmax.max(b.x[i]);
            ymin = ymin.min(b.y[i]);
            ymax = ymax.max(b.y[i]);
        }
        let cx = (xmin + xmax) * 0.5;
        let cy = (ymin + ymax) * 0.5;
        let size = (xmax - xmin).max(ymax - ymin).max(1.0) * 0.5 + 1.0;
        let mut idx: Vec<u32> = (0..n as u32).collect();
        let nodes = build_subtree(b, &mut idx, cx, cy, size, 0);
        Tree { nodes }
    }

    #[inline]
    pub(super) fn accel_for(&self, b: &Bodies, i: usize) -> (f64, f64, f64) {
        let (xi, yi, mi) = (b.x[i], b.y[i], b.m[i]);
        let (vxi, vyi) = (b.vx[i], b.vy[i]);
        let theta2 = BH_THETA * BH_THETA;
        let mut stack = [0u32; 256];
        let mut sp = 1usize;
        let (mut sx, mut sy) = (0.0, 0.0);
        // Pair time-step limits without per-node divisions. With
        // inv = 1 / d^3:  1/t_dyn^2 = G (mi + mj) inv,  v^2 / d^2 = v^2 inv d.
        let mut s_max = 0.0f64;
        let mut w_max = 0.0f64;
        while sp > 0 {
            sp -= 1;
            let n = &self.nodes[stack[sp] as usize];
            let dx = n.com_x - xi;
            let dy = n.com_y - yi;
            let d2 = dx * dx + dy * dy + SOFT2;
            let accept = if n.body >= 0 {
                if n.body as usize == i {
                    continue;
                }
                true
            } else if n.body == -2 {
                if d2 < 4.0 * SOFT2 {
                    continue;
                }
                true
            } else {
                (n.size * 2.0) * (n.size * 2.0) < theta2 * d2
            };
            if accept {
                let d = d2.sqrt();
                let inv = 1.0 / (d2 * d);
                let f = G * n.m * inv;
                sx += f * dx;
                sy += f * dy;
                let s = (mi + n.m) * inv;
                if s > s_max {
                    s_max = s;
                }
                if n.body >= 0 {
                    let j = n.body as usize;
                    let dvx = b.vx[j] - vxi;
                    let dvy = b.vy[j] - vyi;
                    let w = (dvx * dvx + dvy * dvy) * inv * d;
                    if w > w_max {
                        w_max = w;
                    }
                }
            } else {
                for q in 0..4 {
                    if n.children[q] >= 0 && sp < stack.len() {
                        stack[sp] = n.children[q] as u32;
                        sp += 1;
                    }
                }
            }
        }
        (sx, sy, dt_limit(s_max, w_max))
    }
}

/// Build one subtree in pre-order. The returned nodes still have child
/// indices local to this subtree; the caller offsets them when merging.
pub(super) fn build_subtree(
    b: &Bodies,
    indices: &mut [u32],
    cx: f64,
    cy: f64,
    size: f64,
    depth: usize,
) -> Vec<Node> {
    let mut total_m = 0.0;
    let mut com_x = 0.0;
    let mut com_y = 0.0;
    for &i in indices.iter() {
        let i = i as usize;
        total_m += b.m[i];
        com_x += b.m[i] * b.x[i];
        com_y += b.m[i] * b.y[i];
    }
    let (com_x, com_y) = if total_m > 0.0 {
        (com_x / total_m, com_y / total_m)
    } else {
        (cx, cy)
    };
    let single = indices.len() == 1;
    let coincident = !single && size < MIN_NODE_SIZE;
    let root = Node {
        cx,
        cy,
        size,
        m: total_m,
        com_x,
        com_y,
        body: if single {
            indices[0] as i32
        } else if coincident {
            -2
        } else {
            -1
        },
        children: [-1; 4],
    };
    if indices.is_empty() || single || coincident {
        return vec![root];
    }
    let ny = partition_in_place(indices, |i| b.y[i as usize] < cy);
    // Snapshot the length *before* `split_at_mut` borrows `indices`
    // mutably; the sub-slices are still live below.
    let n_here = indices.len();
    let (low, high) = indices.split_at_mut(ny);
    let nx_low = partition_in_place(low, |i| b.x[i as usize] < cx);
    let nx_high = partition_in_place(high, |i| b.x[i as usize] < cx);
    let (q0, q1) = low.split_at_mut(nx_low);
    let (q2, q3) = high.split_at_mut(nx_high);
    let h = size * 0.5;

    // Parallelise the top few levels. Below that the slices are small
    // enough that dispatch cost dominates the work.
    let par = depth < 3 && n_here >= 512;
    let trees: [Vec<Node>; 4] = if par {
        let ((t0, t1), (t2, t3)) = rayon::join(
            || {
                rayon::join(
                    || build_subtree(b, q0, cx - h, cy - h, h, depth + 1),
                    || build_subtree(b, q1, cx + h, cy - h, h, depth + 1),
                )
            },
            || {
                rayon::join(
                    || build_subtree(b, q2, cx - h, cy + h, h, depth + 1),
                    || build_subtree(b, q3, cx + h, cy + h, h, depth + 1),
                )
            },
        );
        [t0, t1, t2, t3]
    } else {
        [
            build_subtree(b, q0, cx - h, cy - h, h, depth + 1),
            build_subtree(b, q1, cx + h, cy - h, h, depth + 1),
            build_subtree(b, q2, cx - h, cy + h, h, depth + 1),
            build_subtree(b, q3, cx + h, cy + h, h, depth + 1),
        ]
    };

    let mut out: Vec<Node> = Vec::with_capacity(1 + trees.iter().map(|t| t.len()).sum::<usize>());
    out.push(root);
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

/// Step limit from the largest `1/t_dyn^2 / G` and `v^2 / d^2` seen.
#[inline]
pub(super) fn dt_limit(s_max: f64, w_max: f64) -> f64 {
    let t_dyn = (1.0 / (G * s_max)).sqrt();
    let t_vel = (1.0 / w_max).sqrt();
    (ETA_GRAV * t_dyn).min(ETA_VEL * t_vel)
}
