use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::ImageAccess,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
};

use crate::image::ImageViews;

pub fn create(device: Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::R16G16B16A16_UNORM,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: Store,
                format: Format::D32_SFLOAT,
                samples: 1
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth},
        },
    )
    .unwrap()
}

pub fn frame_buffer(render_pass: Arc<RenderPass>, image_views: ImageViews) -> Arc<Framebuffer> {
    let dimensions = image_views.render.image().dimensions();

    Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![image_views.render.clone(), image_views.depth.clone()],
            extent: dimensions.width_height(),
            layers: 1,
            ..Default::default()
        },
    )
    .unwrap()
}
