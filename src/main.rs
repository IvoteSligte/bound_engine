use fps_counter::FPSCounter;
use glam::*;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};
use winit::{
    dpi::{PhysicalSize, Size},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub(crate) mod defines;
pub(crate) mod pipelines;
pub(crate) mod shader_objects;
pub(crate) mod shader_wrappers;
pub(crate) mod shaders;

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::VULKAN,
        dx12_shader_compiler: Dx12Compiler::Dxc {
            dxil_path: None,
            dxc_path: None,
        },
    });

    let event_loop = EventLoop::new();
    let window = create_window(&event_loop);
    window.set_cursor_visible(false);

    let surface = unsafe { instance.create_surface(&window) }.unwrap();

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: Some("gpu"),
                features: Features::empty(),
                limits: Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    let shaders = shaders::Shaders::new(&device);

    let pipelines = pipelines::Pipelines::new(&device, &shaders);

    let objects = shader_wrappers::ShaderObjects::new(
        &device,
        &queue,
        UVec2::from_array(window.inner_size().into()),
    );

    let config = surface
        .get_default_config(
            &adapter,
            window.inner_size().width,
            window.inner_size().height,
        )
        .unwrap();

    surface.configure(&device, &config);

    let mut eh = winit_event_helper::EventHelper::new(EventHelperData {
        fps_counter: FPSCounter::new(),
        state: State {
            instance,
            adapter,
            device,
            surface,
            config,
            window,
            shaders,
            pipelines,
            objects,
        },
    });

    let mut callbacks = winit_event_helper::Callbacks::<EventHelperData>::empty();

    callbacks.window.resized(|eh, size| {
        let State {
            config,
            surface,
            device,
            window,
            ..
        } = &mut eh.state;

        config.width = size.width;
        config.height = size.height;

        surface.configure(device, config);
        window.request_redraw();
    });

    callbacks
        .window
        .inputs
        .just_pressed(winit::event::VirtualKeyCode::Escape, |eh| eh.request_quit());

    event_loop.run(move |event, _, control_flow| {
        if eh.data.window.quit().is_some() {
            *control_flow = ControlFlow::Exit;
        }

        if !eh.update(&callbacks, &event) {
            return;
        }

        println!("{}", eh.fps_counter.tick());

        // surface.configure(&device, &config);

        // let frame = surface.get_current_texture().unwrap();
    })
}

#[derive(Debug)]
struct State {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    surface: Surface,
    config: SurfaceConfiguration,
    window: winit::window::Window,
    shaders: shaders::Shaders,
    pipelines: pipelines::Pipelines,
    objects: shader_wrappers::ShaderObjects,
}

#[derive(Debug)]
struct EventHelperData {
    fps_counter: FPSCounter,
    state: State,
}

fn create_window(event_loop: &EventLoop<()>) -> winit::window::Window {
    WindowBuilder::new()
        .with_maximized(true)
        .with_decorations(false)
        .with_visible(true)
        .with_resizable(false)
        .with_inner_size(Size::Physical(PhysicalSize::new(1920, 1080))) // TODO: support for other resolutions
        .build(event_loop)
        .unwrap()
}
