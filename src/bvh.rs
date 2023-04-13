use std::collections::{HashMap, HashSet};

use glam::Vec3;
use itertools::Itertools;

#[derive(Clone, Debug)]
pub(crate) struct CpuBVH {
    pub(crate) root: usize,
    pub(crate) nodes: Vec<CpuNode>,
}

impl From<CpuNode> for CpuBVH {
    fn from(value: CpuNode) -> Self {
        Self {
            root: 0,
            nodes: vec![value.into()],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CpuNode {
    pub position: Vec3,
    pub radius: f32,
    pub child: Option<usize>,    // index of child node
    pub next: Option<usize>,     // index of child node
    pub material: Option<usize>, // index of object
    pub parent: Option<usize>,   // index of parent
}

#[derive(Clone, Copy)]
enum CpuNodePtrType {
    Parent,
    Previous,
}

#[derive(Clone)]
struct CpuNodePtr {
    index: usize,
    ptr_type: CpuNodePtrType,
}

#[derive(Clone, Copy, PartialEq, Default, Debug)]
struct Sphere {
    position: Vec3,
    radius: f32,
}

impl From<(Vec3, f32)> for Sphere {
    fn from((position, radius): (Vec3, f32)) -> Self {
        Sphere { position, radius }
    }
}

#[derive(Clone, Copy, Debug)]
struct BestFitReplace {
    index: usize,
    sphere: Sphere,
}

type InsertIndex = usize;

impl CpuBVH {
    fn root(&self) -> &CpuNode {
        &self.nodes[self.root]
    }

    fn append_other(&mut self, other: CpuBVH) {
        let l = self.nodes.len();

        let iter = other.nodes.into_iter().map(|mut n| {
            n.child.as_mut().map(|x| *x += l);
            n.next.as_mut().map(|x| *x += l);
            n
        });

        self.nodes.extend(iter);
    }

    fn push_new_node(&mut self, child_0_idx: usize, best_fit: BestFitReplace) {
        self.nodes.push(CpuNode {
            position: best_fit.sphere.position,
            radius: best_fit.sphere.radius,
            child: Some(child_0_idx),
            next: self.nodes[best_fit.index].next,
            material: None,
            parent: self.nodes[best_fit.index].parent,
        });
    }

    // FIXME: spheres are invisible occasionally due to faulty boundaries
    /// Merges `other` into `self`.
    pub fn merge_in_place(&mut self, other: CpuBVH) {
        let is_self_in_other = other.root().sphere().contains(&self.root().sphere());

        let other_root_idx = other.root + self.nodes.len();

        self.append_other(other);

        let (root_idx, target_idx) = if is_self_in_other {
            (other_root_idx, self.root)
        } else {
            (self.root, other_root_idx)
        };

        let result = self.best_fit(root_idx, self.nodes[target_idx].sphere());
        let best_fit = match result {
            Ok(bf) => bf,
            Err(target_idx) => {
                self.insert_node_as_child(root_idx, target_idx);
                return;
            }
        };

        self.push_new_node(target_idx, best_fit);

        self.rewire_child_1_pointer(best_fit.index, self.nodes.len() - 1);
        self.rewire_child_pointers(target_idx, best_fit.index, self.nodes.len() - 1);

        self.resize_parents_recursive(self.nodes.len() - 1);

        let new_node_surrounds_all = self.nodes[self.nodes.len() - 1].parent.is_none();
        let other_surrounds_all = self.nodes[other_root_idx].parent.is_none();

        if new_node_surrounds_all {
            self.root = self.nodes.len() - 1;
        } else if other_surrounds_all {
            self.root = other_root_idx;
        }
    }

    // TODO:
    // fn optimise(&mut self) {

    // }

    // fn merge_node_with_parent(&mut self, target_idx: usize) {
    //     let CpuNodePtr { index, ptr_type } = self.pointer_to_node(target_idx);

    //     match ptr_type {
    //         CpuNodePtrType::Parent => self.nodes[index].child = self.nodes[target_idx].next,
    //         CpuNodePtrType::Previous => self.nodes[index].next = self.nodes[target_idx].next,
    //     }

    //     let children = self.children_of_node(target_idx);
    //     for c in children {
    //         self.insert_node_as_child(c, self.nodes[target_idx].parent.unwrap());
    //     }
    // }

    fn insert_node_as_child(&mut self, insert_target_idx: usize, target_idx: usize) {
        let prev_child_idx = self.nodes[target_idx].child;
        self.nodes[target_idx].child = Some(insert_target_idx);
        self.nodes[insert_target_idx].next = prev_child_idx;
    }

    fn rewire_child_1_pointer(&mut self, child_1_idx: usize, new_idx: usize) {
        if self.nodes[child_1_idx].parent.is_none() {
            return;
        }

        let CpuNodePtr { index, ptr_type } = self.pointer_to_node(child_1_idx);

        match ptr_type {
            CpuNodePtrType::Parent => self.nodes[index].child = Some(new_idx),
            CpuNodePtrType::Previous => self.nodes[index].next = Some(new_idx),
        }
    }

    fn rewire_child_pointers(&mut self, child_0_idx: usize, child_1_idx: usize, new_idx: usize) {
        self.nodes[child_0_idx].next = Some(child_1_idx);
        self.nodes[child_0_idx].parent = Some(new_idx);
        self.nodes[child_1_idx].parent = Some(new_idx);
    }

    fn best_fit(&self, root_idx: usize, target: Sphere) -> Result<BestFitReplace, InsertIndex> {
        let mut best_fit: Result<BestFitReplace, InsertIndex> = Err(root_idx);

        let mut current_idx = root_idx;

        loop {
            let current = &self.nodes[current_idx];

            if current.sphere().contains(&target) {
                if best_fit.is_err() && current.radius < self.nodes[best_fit.unwrap_err()].radius {
                    best_fit = Err(current_idx);
                }

                current_idx = match current.child.or(current.next) {
                    Some(x) => x,
                    None => break,
                };
                continue;
            }

            let surrounding_sphere = current.sphere().calculate_sphere(&target);

            // if it makes the surface area of the BVH smaller (SAH)...
            if best_fit.is_err() || surrounding_sphere.radius < best_fit.unwrap().sphere.radius {
                best_fit = Ok(BestFitReplace {
                    index: current_idx,
                    sphere: surrounding_sphere,
                });
            }

            current_idx = match current.child.or(current.next) {
                Some(x) => x,
                None => break,
            };
        }

        best_fit
    }

    fn children_of_node(&self, target_idx: usize) -> Vec<usize> {
        let mut children = vec![];
        let mut idx = self.nodes[target_idx].child.unwrap();

        while self.nodes[idx].parent == Some(target_idx) {
            children.push(idx);
            match self.nodes[idx].next {
                Some(next) => idx = next,
                None => break,
            }
        }

        children
    }

    fn pointer_to_node(&self, target_idx: usize) -> CpuNodePtr {
        let parent_idx = self.nodes[target_idx].parent.unwrap();
        let parent = &self.nodes[parent_idx];

        let child_idx = parent.child.unwrap();
        if child_idx == target_idx {
            return CpuNodePtr {
                index: parent_idx,
                ptr_type: CpuNodePtrType::Parent,
            };
        }

        let mut idx = child_idx;
        while let Some(next_idx) = self.nodes[idx].next {
            if next_idx == target_idx {
                return CpuNodePtr {
                    index: idx,
                    ptr_type: CpuNodePtrType::Previous,
                };
            }
            idx = next_idx;
        }

        panic!("Node not found.");
    }

    fn largest_surrounding_sphere(&self, parent: usize) -> Sphere {
        let spheres = self
            .children_of_node(parent)
            .iter()
            .map(|&idx| self.nodes[idx].sphere())
            .collect();

        Sphere::largest_surrounding_sphere(spheres)
    }

    fn resize_parents_recursive(&mut self, node_idx: usize) {
        let parent_idx = match self.nodes[node_idx].parent {
            Some(p) => p,
            None => return,
        };

        let new_sphere = self.largest_surrounding_sphere(parent_idx);

        let old_radius = self.nodes[parent_idx].radius;

        if old_radius != new_sphere.radius {
            self.nodes[parent_idx].position = new_sphere.position;
            self.nodes[parent_idx].radius = new_sphere.radius;

            self.resize_parents_recursive(parent_idx);
        }
    }

    #[allow(dead_code)]
    pub fn graphify(&self) {
        let mut graph = petgraph::graph::Graph::<usize, &str>::new();
        let mut vertices = HashSet::new();
        let mut edges = vec![];

        let mut stack = vec![self.root];
        vertices.insert(self.root);

        while let Some(current) = stack.pop() {
            let node = &self.nodes[current];

            if let Some(x) = node.next {
                if !vertices.contains(&x) {
                    stack.push(x);
                    vertices.insert(x);
                }

                if self.nodes[x].parent == node.parent {
                    edges.push((current, x, "sibling"));
                } else {
                    edges.push((current, x, "next"));
                }
            }

            if let Some(x) = node.child {
                if !vertices.contains(&x) {
                    stack.push(x);
                    vertices.insert(x);
                }
                edges.push((current, x, "child"));
            }
        }

        let vertex_map = vertices
            .into_iter()
            .map(|v| (v, graph.add_node(v)))
            .collect::<HashMap<_, _>>();

        edges.iter().for_each(|(start, end, tag)| {
            graph.add_edge(vertex_map[start], vertex_map[end], tag);
        });

        println!("{}", petgraph::dot::Dot::new(&graph));
    }
}

impl CpuNode {
    fn sphere(&self) -> Sphere {
        Sphere {
            position: self.position,
            radius: self.radius,
        }
    }
}

impl Sphere {
    fn contains(&self, other: &Self) -> bool {
        let distance = self.position.distance(other.position);
        distance + other.radius <= self.radius
    }

    #[allow(dead_code)]
    fn contains_point(&self, point: Vec3) -> bool {
        let distance = self.position.distance(point);
        distance <= self.radius
    }

    fn calculate_sphere(&self, other: &Self) -> Self {
        let radius = self.calculate_radius(other);
        let position = self.calculate_position(other, radius);
        Self { position, radius }
    }

    fn calculate_radius(&self, other: &Self) -> f32 {
        self.radius
            .max(other.radius)
            .max((self.position.distance(other.position) + self.radius + other.radius) * 0.5)
    }

    fn calculate_position(&self, other: &Self, parent_radius: f32) -> Vec3 {
        let (s, l) = if self.radius < other.radius {
            (self, other)
        } else {
            (other, self)
        };

        let a = ((parent_radius - l.radius) / s.position.distance(l.position)).max(0.0);
        s.position * a + l.position * (1.0 - a)
    }

    // FIXME: gets called with only one item in `spheres`
    fn largest_surrounding_sphere(spheres: Vec<Sphere>) -> Sphere {
        spheres
            .iter()
            .combinations(2)
            .map(|vec| (vec[0], vec[1]))
            .map(|(&a, b)| a.calculate_sphere(b))
            .max_by(|a, b| a.radius.total_cmp(&b.radius))
            .unwrap()
    }
}

impl From<CpuBVH> for crate::shaders::ty::GpuBVH {
    fn from(source: CpuBVH) -> Self {
        let mut bvh = crate::shaders::ty::GpuBVH {
            root: source.root as u32 + 1, // node 0 is empty
            ..Default::default()
        };

        let mut raw_source = source
            .nodes
            .into_iter()
            .map(|n| crate::shaders::ty::Bounds {
                position: n.position.to_array(),
                radiusSquared: n.radius * n.radius,
                child: n.child.map(|x| x as u32 + 1).unwrap_or(0),
                next: n.next.map(|x| x as u32 + 1).unwrap_or(0),
                material: n.material.map(|x| x as u32 + 1).unwrap_or(0),
                radius: n.radius,
            })
            .collect::<Vec<_>>();

        bvh.nodes[0] = Default::default();
        bvh.nodes[1..=raw_source.len()].swap_with_slice(&mut raw_source);
        bvh
    }
}
