use std::sync::Arc;

use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    Version, VulkanLibrary,
};

pub(crate) fn create_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);

    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            engine_version: Version::V1_3,
            ..Default::default()
        },
    )
    .unwrap()
}
