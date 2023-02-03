use std::{f32::consts::FRAC_PI_2, io::Cursor, sync::Arc, time::Instant};

use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use image::io::Reader as ImageReader;
use vulkano::command_buffer::PrimaryCommandBufferAbstract;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
        QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageUsage, ImmutableImage,
        MipmapsCount, SwapchainImage,
    },
    impl_vertex,
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    memory::allocator::{MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            render_pass::PipelineRenderingCreateInfo,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::StoreOp,
    sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    let event_loop = EventLoop::new();

    let library = VulkanLibrary::new().unwrap();

    #[cfg(feature = "vulkan-debug")]
    {
        println!("List of Vulkan debugging layers available to use:");
        let layers = library.layer_properties().unwrap();
        for l in layers {
            println!("\t{}", l.name());
        }
    }

    let required_extensions = vulkano_win::required_extensions(&library);
    #[cfg(feature = "vulkan-debug")]
    let enabled_extensions = required_extensions.union(&InstanceExtensions {
        ext_debug_utils: true,
        ..InstanceExtensions::empty()
    });
    #[cfg(not(feature = "vulkan-debug"))]
    let enabled_extensions: InstanceExtensions = required_extensions;

    #[cfg(feature = "vulkan-debug")]
    let layers = vec![
        "VK_LAYER_KHRONOS_validation".to_owned(),
        // "VK_LAYER_LUNARG_api_dump".to_owned(),
    ];
    #[cfg(not(feature = "vulkan-debug"))]
    let layers = vec![];

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions,
            enumerate_portability: true,
            enabled_layers: layers,
            ..Default::default()
        },
    )
    .expect("Failed to create vulkan instance");

    let _debug_callback = unsafe {
        DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity {
                    error: true,
                    warning: true,
                    information: true,
                    verbose: true,
                    ..DebugUtilsMessageSeverity::empty()
                },
                message_type: DebugUtilsMessageType {
                    general: true,
                    validation: true,
                    performance: true,
                    ..DebugUtilsMessageType::empty()
                },
                ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                    let severity = if msg.severity.error {
                        "error"
                    } else if msg.severity.warning {
                        "warning"
                    } else if msg.severity.information {
                        "information"
                    } else if msg.severity.verbose {
                        "verbose"
                    } else {
                        panic!("no-impl");
                    };

                    let ty = if msg.ty.general {
                        "general"
                    } else if msg.ty.validation {
                        "validation"
                    } else if msg.ty.performance {
                        "performance"
                    } else {
                        panic!("no-impl");
                    };

                    println!(
                        "{} {} {}: {}",
                        msg.layer_prefix.unwrap_or("unknown"),
                        ty,
                        severity,
                        msg.description
                    );
                }))
            },
        )
        .ok()
    };

    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .expect("Failed to build a vulkan surface");

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        khr_dynamic_rendering: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.graphics && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("No suitable physical device found");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_features: Features {
                dynamic_rendering: true,
                ..Features::empty()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage {
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct Vertex {
        position: [f32; 3],
    }
    impl_vertex!(Vertex, position);

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct InstanceData {
        position_offset: [f32; 3],
        color: [f32; 3],
    }
    impl_vertex!(InstanceData, position_offset, color);

    let vertices = [
        // 0: A
        Vertex {
            position: [0.0, 0.0, 0.0],
        },
        // 1: B
        Vertex {
            position: [0.0, 1.0, 0.0],
        },
        // 2: C
        Vertex {
            position: [1.0, 1.0, 0.0],
        },
        // 3: D
        Vertex {
            position: [1.0, 0.0, 0.0],
        },
        // 4: E
        Vertex {
            position: [1.0, 0.0, 1.0],
        },
        // 5: F
        Vertex {
            position: [1.0, 1.0, 1.0],
        },
        // 6: G
        Vertex {
            position: [0.0, 0.0, 1.0],
        },
        // 7: H
        Vertex {
            position: [0.0, 1.0, 1.0],
        },
    ];
    #[rustfmt::skip]
    let indices = vec![
        // ABC
        0u16, 1u16, 2u16,
        // DAB
        3u16, 0u16, 2u16,
        // DCE
        3u16, 2u16, 4u16,
        // CEF
        2u16, 4u16, 5u16,
        // BAG
        1u16, 0u16, 6u16,
        // BGH
        1u16, 6u16, 7u16,
        // GHF
        6u16, 7u16, 5u16,
        // GFE
        6u16, 5u16, 4u16,
        // ADG
        0u16, 3u16, 6u16,
        // GDE
        6u16, 3u16, 4u16,
        // BCH
        1u16, 2u16, 7u16,
        // CHF
        2u16, 7u16, 5u16,
    ];

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        vertices,
    )
    .unwrap();

    let instances = vec![
        InstanceData {
            position_offset: [0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        // InstanceData {
        //     position_offset: [1.0, 0.0],
        //     color: [0.0, 1.0, 0.0],
        // },
        // InstanceData {
        //     position_offset: [0.0, 1.0],
        //     color: [1.0, 0.0, 1.0],
        // },
        // InstanceData {
        //     position_offset: [0.0, 2.0],
        //     color: [0.0, 1.0, 1.0],
        // },
    ];
    let instance_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        instances,
    )
    .unwrap();

    let index_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            index_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        indices,
    )
    .unwrap();

    let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(
        memory_allocator.clone(),
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::empty()
        },
        MemoryUsage::Upload,
    );

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let pipeline = GraphicsPipeline::start()
        .render_pass(PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(swapchain.image_format())],
            depth_attachment_format: Some(Format::D16_UNORM),
            ..Default::default()
        })
        .vertex_input_state(
            BuffersDefinition::new()
                .vertex::<Vertex>()
                .instance::<InstanceData>(),
        )
        .input_assembly_state(InputAssemblyState::new())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .build(device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let dimensions = images[0].dimensions().width_height();
    let depth_format: Format = Format::D16_UNORM;
    // device
    // .physical_device()
    // .surface_formats(&surface, Default::default())
    // .iter()
    // .flat_map(|formats| formats)
    // .find(
    //     |(format, _)| match device.physical_device().format_properties(*format) {
    //         Ok(props) if props.linear_tiling_features.depth_stencil_attachment => true,
    //         _ => false,
    //     },
    // )
    // .unwrap()
    // .0;
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(&memory_allocator, dimensions, depth_format).unwrap(),
    )
    .unwrap();

    let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);

    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut uploads = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let texture = {
        let bytes = include_bytes!("texture.jpg").to_vec();
        let mut image = ImageReader::new(Cursor::new(bytes));
        image.set_format(image::ImageFormat::Jpeg);
        let image = image.decode().unwrap();
        let dimensions = ImageDimensions::Dim2d {
            width: image.width(),
            height: image.height(),
            array_layers: 1,
        };
        let immutable_image = ImmutableImage::from_iter(
            &memory_allocator,
            image.into_rgba8().to_vec(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            &mut uploads,
        )
        .unwrap();

        ImageView::new_default(immutable_image).unwrap()
    };

    let sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: vulkano::sampler::Filter::Linear,
            min_filter: vulkano::sampler::Filter::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            ..Default::default()
        },
    )
    .unwrap();

    let layout0 = pipeline.layout().set_layouts().get(0).unwrap();
    let textures_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout0.clone(),
        [WriteDescriptorSet::image_view_sampler(0, texture, sampler)],
    )
    .unwrap();

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(
        uploads
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .boxed(),
    );

    let position = Vector3::new(-0.5, -0.5, -0.5);

    let rotation_start = Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *control_flow = ControlFlow::Exit,
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Left),
                            ..
                        },
                    ..
                },
            ..
        } => {
            todo!();
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Right),
                            ..
                        },
                    ..
                },
            ..
        } => {
            todo!();
        }
        Event::RedrawEventsCleared => {
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            let dimensions = window.inner_size();
            if dimensions.width == 0 || dimensions.height == 0 {
                return;
            }

            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {e:?}"),
                };
                swapchain = new_swapchain;
                attachment_image_views = window_size_dependent_setup(&new_images, &mut viewport);
                recreate_swapchain = false;
            }

            let uniform_buffer_subbuffer = {
                let elapsed = rotation_start.elapsed();
                let rotation =
                    elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                let w = swapchain.image_extent()[0] as f32;
                let h = swapchain.image_extent()[1] as f32;
                let aspect_ratio = w / h;
                let near = 0.01;
                let far = 100.0;
                let fov = Rad(FRAC_PI_2);
                let proj = cgmath::perspective(fov, aspect_ratio, near, far);

                let eye = Point3::new(0.0, 1.0, -2.0);
                let center = Point3::new(0.0, 0.0, 0.0);
                let up = Vector3::new(0.0, -1.0, 0.0);
                let view = Matrix4::look_at_rh(eye, center, up);

                let scale = Matrix4::from_scale(0.7);

                let uniform_data = vs::ty::Data {
                    world: (Matrix4::from(rotation) * Matrix4::from_translation(position)).into(),
                    view: (view * scale).into(),
                    proj: proj.into(),
                };

                uniform_buffer.from_data(uniform_data).unwrap()
            };

            let layout1 = pipeline.layout().set_layouts().get(1).unwrap();
            let instances_set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout1.clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
            )
            .unwrap();

            let (image_index, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {e:?}"),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .begin_rendering(RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        // Clear the content of this attachment at the beginning of the rendering
                        load_op: vulkano::render_pass::LoadOp::Clear,
                        // Store the rendering result in the attachment image (instead of
                        // discarding it)
                        store_op: vulkano::render_pass::StoreOp::Store,
                        clear_value: Some([1.0, 1.0, 1.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(
                            attachment_image_views[image_index as usize].clone(),
                        )
                    })],
                    depth_attachment: Some(RenderingAttachmentInfo {
                        load_op: vulkano::render_pass::LoadOp::Clear,
                        store_op: StoreOp::DontCare,
                        clear_value: Some([1.0, 1.0, 1.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(depth_buffer.clone())
                    }),
                    ..Default::default()
                })
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    vec![textures_set.clone(), instances_set.clone()],
                )
                .bind_vertex_buffers(0, (vertex_buffer.clone(), instance_buffer.clone()))
                .bind_index_buffer(index_buffer.clone())
                .draw_indexed(
                    index_buffer.len() as u32,
                    instance_buffer.len() as u32,
                    0,
                    0,
                    0,
                )
                .unwrap()
                .end_rendering()
                .unwrap();

            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => panic!("Failed to flush future: {e:?}"),
            }
        }
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView<SwapchainImage>>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path:"src/vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/frag.glsl",
    }
}
