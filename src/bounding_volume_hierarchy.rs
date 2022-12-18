use std::ops::Range;

use glam::Vec3;
use ptree::{print_tree, TreeBuilder};

#[derive(Clone, Debug)]
pub(crate) struct BVH {
    pub(crate) head: usize,
    pub(crate) nodes: Vec<BVHNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BVHNode {
    pub center: Vec3,
    pub radius: f32,
    pub left: Option<usize>,   // index of child node
    pub right: Option<usize>,  // index of child node
    pub leaf: Option<usize>,   // index of object
    pub parent: Option<usize>, // index of parent
}

impl BVH {
    /// Merges `other` into `self` and returns the changed indices.
    pub fn merge(&mut self, mut other: BVH) -> (Vec<Range<usize>>, bool) {
        let initial_head = self.head;
        let (self_head, other_head) = (&self.nodes[self.head], &other.nodes[other.head]);

        let distance = self_head.center.distance(other_head.center);

        let mut buffer_copy_ranges = Vec::new();
        let initial_len = self.nodes.len();

        let append_other_to_self = |s: &mut BVH, o: BVH| {
            let l = s.nodes.len();

            let iter = o.nodes.into_iter().map(|mut n| {
                n.left.as_mut().map(|x| *x += l);
                n.right.as_mut().map(|x| *x += l);
                n.parent.as_mut().map(|x| *x += l);
                n
            });

            s.nodes.extend(iter);
        };

        // if self is contained within other
        if distance + self_head.radius <= other_head.radius {
            let (index, center, radius) = other.get_best_fit(&self.nodes[self.head]);

            self.nodes.push(BVHNode {
                center,
                radius,
                left: Some(index + self.nodes.len() + 1),
                right: Some(self.head),
                leaf: None,
                parent: other.nodes[index].parent.map(|x| x + self.nodes.len()),
            });

            let len = self.nodes.len();
            self.head = if index == other.head {
                len - 1
            } else {
                other.head + len
            };
            append_other_to_self(self, other);

            if let Some(p) = self.nodes[index + len].parent {
                buffer_copy_ranges.push(p..(p + 1));

                let parent = &mut self.nodes[p];

                if parent.left == Some(index) {
                    parent.left.as_mut().map(|x| *x = len - 1);
                } else {
                    parent.right.as_mut().map(|x| *x = len - 1);
                }
            }

            let BVHNode { left, right, .. } = self.nodes[len - 1];
            self.nodes[left.unwrap()].parent = Some(len - 1);
            self.nodes[right.unwrap()].parent = Some(len - 1);
        }
        // if other is contained within self
        else if distance + other_head.radius <= self_head.radius {
            let (index, center, radius) = self.get_best_fit(&other.nodes[other.head]);

            self.nodes.push(BVHNode {
                center,
                radius,
                left: Some(index),
                right: Some(other.head + self.nodes.len() + 1),
                leaf: None,
                parent: self.nodes[index].parent,
            });

            let len = self.nodes.len();
            if index == self.head {
                self.head = len - 1;
            }
            append_other_to_self(self, other);

            if let Some(p) = self.nodes[index].parent {
                buffer_copy_ranges.push(p..(p + 1));

                let parent = &mut self.nodes[p];

                if parent.left == Some(index) {
                    parent.left.as_mut().map(|x| *x = len - 1);
                } else {
                    parent.right.as_mut().map(|x| *x = len - 1);
                }
            }

            let BVHNode { left, right, .. } = self.nodes[len - 1];
            self.nodes[left.unwrap()].parent = Some(len - 1);
            self.nodes[right.unwrap()].parent = Some(len - 1);
        }
        // if neither is contained within the other
        else {
            let radius = self_head.get_parent_radius(other_head);
            let center = self_head.get_parent_center(other_head, radius);

            self.nodes.push(BVHNode {
                center,
                radius,
                left: Some(other.head + 1 + self.nodes.len()),
                right: Some(self.head),
                leaf: None,
                parent: None,
            });

            let len = self.nodes.len();
            self.head = self.nodes.len() - 1;
            append_other_to_self(self, other);

            let BVHNode { left, right, .. } = self.nodes[len - 1];
            self.nodes[left.unwrap()].parent = Some(len - 1);
            self.nodes[right.unwrap()].parent = Some(len - 1);
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

        if self.nodes[self.head].left.is_none() {
            return best_fit;
        }

        let mut stack = vec![
            self.nodes[self.head].left.unwrap(),
            self.nodes[self.head].right.unwrap(),
        ];

        while !stack.is_empty() {
            let i = stack.pop().unwrap();
            let m = &self.nodes[i];

            let new_radius = m.get_parent_radius(node);
            let new_center = m.get_parent_center(node, new_radius);

            let parent = &self.nodes[m.parent.unwrap()];

            let does_not_deform_parent =
                parent.center.distance(new_center) + new_radius <= parent.radius;
            
            if does_not_deform_parent {
                m.left.map(|l| stack.push(l));
                m.right.map(|r| stack.push(r));

                if new_radius < best_fit.2 {
                    best_fit = (i, new_center, new_radius);
                }
            }
        }
        
        best_fit
    }

    #[allow(dead_code)]
    pub fn pretty_print(&self) {
        let mut tree = TreeBuilder::new("Bounding Volume Hierarchy".to_string());
        self.traverse_tree(&self.nodes[self.head], &mut tree);
        print_tree(&tree.build()).unwrap();
    }

    #[allow(dead_code)]
    fn traverse_tree(&self, node: &BVHNode, printer: &mut TreeBuilder) {
        if node.leaf.is_some() {
            return;
        }

        let left = &self.nodes[node.left.unwrap()];
        printer.begin_child(format!("{}, {}", left.center, left.radius));
        self.traverse_tree(left, printer);
        printer.end_child();

        let right = &self.nodes[node.right.unwrap()];
        printer.begin_child(format!("{}, {}", right.center, right.radius));
        self.traverse_tree(right, printer);
        printer.end_child();
    }
}

impl BVHNode {
    fn get_parent_radius(&self, other: &BVHNode) -> f32 {
        self.radius
            .max(other.radius)
            .max((self.center.distance(other.center) + self.radius + other.radius) / 2.0)
    }

    fn get_parent_center(&self, other: &BVHNode, parent_radius: f32) -> Vec3 {
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
                left: n.left.map(|x| x as u32 + 1).unwrap_or(0), // TODO: figure out why swapping left and right breaks the thing
                right: n.right.map(|x| x as u32 + 1).unwrap_or(0),
                leaf: n.leaf.map(|x| x as u32).unwrap_or(0),
                parent: n.parent.map(|x| x as u32 + 1).unwrap_or(0),
            })
            .collect::<Vec<_>>();

        bvh.volumes[0] = Default::default();
        bvh.volumes[1..=raw_source.len()].swap_with_slice(&mut raw_source);
        bvh
    }
}
