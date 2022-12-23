use std::ops::Range;

use glam::Vec3;

#[derive(Clone, Debug)]
pub(crate) struct BVH {
    pub(crate) head: usize,
    pub(crate) nodes: Vec<BVHNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BVHNode {
    pub center: Vec3,
    pub radius: f32,
    pub child: Option<usize>,  // index of child node
    pub next: Option<usize>,   // index of child node
    pub leaf: Option<usize>,   // index of object
    pub parent: Option<usize>, // index of parent
}

impl BVH {
    /// Merges `other` into `self` and returns the changed indices.
    pub fn merge_in_place(&mut self, mut other: BVH) -> (Vec<Range<usize>>, bool) {
        let initial_head = self.head;
        let (self_head, other_head) = (&self.nodes[self.head], &other.nodes[other.head]);

        let distance = self_head.center.distance(other_head.center);

        let mut buffer_copy_ranges = vec![];
        let initial_len = self.nodes.len();

        let append_other_to_self = |s: &mut BVH, o: BVH| {
            let l = s.nodes.len();

            let iter = o.nodes.into_iter().map(|mut n| {
                n.child.as_mut().map(|x| *x += l);
                n.next.as_mut().map(|x| *x += l);
                n
            });

            s.nodes.extend(iter);
        };

        let is_self_in_other = distance + self_head.radius <= other_head.radius;
        let is_other_in_self = distance + other_head.radius <= self_head.radius;

        if is_self_in_other {
            let (mut right_child, center, radius) = other.get_best_fit(&self.nodes[self.head]);

            right_child += self.nodes.len();

            let left_child = self.head;

            let other_head = other.head + self.nodes.len();
            append_other_to_self(self, other);

            let l = self.nodes.len();

            self.nodes.push(BVHNode {
                center,
                radius,
                child: Some(left_child),
                next: self.nodes[right_child].next,
                leaf: None,
                parent: self.nodes[right_child].parent,
            });

            if let Some(p) = self.nodes[right_child].parent {
                buffer_copy_ranges.push(p..(p + 1));

                let child = self.nodes[p].child.as_mut().unwrap();

                // remapping "pointer" to right_child to point to current node
                if *child == right_child {
                    *child = l;
                } else {
                    let c = *child;
                    self.nodes[c].next = Some(l);
                }
            }
            
            // left child
            self.nodes[left_child].next = Some(right_child);
            self.nodes[left_child].parent = Some(l);
            // right child
            self.nodes[right_child].parent = Some(l);

            if right_child == other_head {
                self.head = l;
            } else {
                self.head = other_head;
            };
        } else if is_other_in_self {
            let (right_child, center, radius) = self.get_best_fit(&other.nodes[other.head]);

            let left_child = other.head + self.nodes.len();

            append_other_to_self(self, other);

            let l = self.nodes.len();

            self.nodes.push(BVHNode {
                center,
                radius,
                child: Some(left_child),
                next: self.nodes[right_child].next,
                leaf: None,
                parent: self.nodes[right_child].parent,
            });

            if let Some(p) = self.nodes[right_child].parent {
                buffer_copy_ranges.push(p..(p + 1));

                let child = self.nodes[p].child.as_mut().unwrap();

                // remapping "pointer" to right_child to point to current node
                if *child == right_child {
                    *child = l;
                } else {
                    let c = *child;
                    self.nodes[c].next = Some(l);
                }
            }

            // left child
            self.nodes[left_child].next = Some(right_child);
            self.nodes[left_child].parent = Some(l);
            // right child
            self.nodes[right_child].parent = Some(l);

            if right_child == self.head {
                self.head = l;
            }
        } else {
            let radius = self_head.calculate_radius(other_head);
            let center = self_head.calculate_center(other_head, radius);

            let left_child = other.head + self.nodes.len();
            let right_child = self.head;
            
            append_other_to_self(self, other);

            let l = self.nodes.len();

            // adds parent node (index of new node = l)
            self.nodes.push(BVHNode {
                center,
                radius,
                child: Some(left_child),
                next: None,
                leaf: None,
                parent: None,
            });
            
            // left child
            self.nodes[left_child].next = Some(right_child);
            self.nodes[left_child].parent = Some(l);
            // right child
            self.nodes[right_child].parent = Some(l);

            self.head = l;
        }

        buffer_copy_ranges.push(initial_len..self.nodes.len());
        (buffer_copy_ranges, initial_head != self.head)
    }

    /// Gets the best fit for a given node (in `self`) to replace.
    ///
    /// *Warning:* assumes `node` fits within `self`.
    fn get_best_fit(&mut self, node: &BVHNode) -> (usize, Vec3, f32) {
        // find the smallest node that the given node can form a boundary with, without changing the former's parent boundary
        let mut best_fit = (
            self.head,
            self.nodes[self.head].center,
            self.nodes[self.head].radius,
        );

        let mut curr_idx = match self.nodes[self.head].child {
            Some(x) => x,
            None => return best_fit,
        };

        loop {
            let m = &self.nodes[curr_idx];

            let new_radius = m.calculate_radius(node);
            let new_center = m.calculate_center(node, new_radius);

            let parent = &self.nodes[m.parent.unwrap()];

            let does_not_deform_parent =
                parent.center.distance(new_center) + new_radius <= parent.radius;

            if !does_not_deform_parent {
                curr_idx = match m.next {
                    Some(x) => x,
                    None => break,
                };

                continue;
            }

            if new_radius < best_fit.2 {
                best_fit = (curr_idx, new_center, new_radius);
            }

            curr_idx = match m.child {
                Some(x) => x,
                None => break,
            };
        }

        best_fit
    }
}

impl BVHNode {
    fn calculate_radius(&self, other: &BVHNode) -> f32 {
        self.radius
            .max(other.radius)
            .max((self.center.distance(other.center) + self.radius + other.radius) / 2.0)
    }

    fn calculate_center(&self, other: &BVHNode, parent_radius: f32) -> Vec3 {
        let mut array = [self, other];
        array.sort_by(|a, b| a.radius.total_cmp(&b.radius));
        let [smaller, larger] = array;

        let (c1, c2) = (smaller.center, larger.center);
        let a = ((parent_radius - larger.radius) / self.center.distance(other.center)).max(0.0);
        c1 * a + c2 * (1.0 - a)
    }
}

impl From<BVH> for crate::shaders::ty::BoundingVolumeHierarchy {
    fn from(source: BVH) -> Self {
        let mut bvh = crate::shaders::ty::BoundingVolumeHierarchy {
            head: source.head as u32 + 1, // node 0 is empty
            ..Default::default()
        };

        let mut raw_source = source
            .nodes
            .into_iter()
            .map(|n| crate::shaders::ty::Bounds {
                center: n.center.to_array(),
                radiusSquared: n.radius * n.radius,
                child: n.child.map(|x| x as u32 + 1).unwrap_or(0),
                next: n.next.map(|x| x as u32 + 1).unwrap_or(0),
                leaf: n.leaf.map(|x| x as u32).unwrap_or(0),
                _dummy0: [0u8; 4],
            })
            .collect::<Vec<_>>();

        bvh.nodes[0] = Default::default();
        bvh.nodes[1..=raw_source.len()].swap_with_slice(&mut raw_source);
        bvh
    }
}
