use std::sync::Arc;

use vulkano::{
    device::Device,
    format::Format,
    image::ImageViewAbstract,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
};

pub fn create(device: Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                load: DontCare,
                store: Store,
                format: Format::R16G16B16A16_UNORM,
                samples: 1,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap()
}

pub fn frame_buffer(
    render_pass: Arc<RenderPass>,
    render_image_view: Arc<dyn ImageViewAbstract>,
) -> Arc<Framebuffer> {
    let dimensions = render_image_view.image().dimensions();

    Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![render_image_view],
            extent: dimensions.width_height(),
            layers: 1,
            ..Default::default()
        },
    )
    .unwrap()
}
