use std::sync::Arc;

use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    Version, VulkanLibrary,
};

pub(crate) fn create_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);

    #[cfg(debug_assertions)]
    let required_extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..required_extensions
    };

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
