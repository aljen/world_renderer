use std::path::Path;

use bevy::{
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
    math::Vec2,
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::{
        mesh::Indices,
        render_resource::{Extent3d, PrimitiveTopology, TextureDimension, TextureFormat},
        settings::{WgpuFeatures, WgpuSettings},
        texture::ImageSettings,
    },
    window::PresentMode,
};
use bevy_egui::{egui, EguiContext};

use bevy_inspector_egui::{
    egui::{Align2, Color32, RichText},
    WorldInspectorPlugin,
};

use gdal::*;

use noise::{NoiseFn, Perlin, Seedable};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// reference: https://www.youtube.com/playlist?list=PLFt_AvWsXl0eBW2EiBtl_sxmDtSgZBxB3

#[derive(Component)]
pub struct Terrain;

#[derive(PartialEq, Clone, Copy)]
pub struct GeneratorConfig {
    seed: u64,

    noise_scale: f64,
    octaves: i32,
    persistance: f64,
    lacunarity: f64,
}

#[derive(PartialEq, Eq, Clone)]
pub struct GeoTifConfig {
    filename: String,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum ConfigType {
    Generator,
    GeoTif,
}

pub struct WorldStats {
    triangles: usize,
    vertices: usize,
}

pub struct TerrainLayer {
    height: f32,
    color: Color,
}

#[derive(PartialEq, Clone)]
pub struct WorldConfig {
    width: usize,
    height: usize,
    z_scale: f64,

    auto_update: bool,
    wireframe: bool,

    world_type: ConfigType,

    generator: GeneratorConfig,
    geo_tif: GeoTifConfig,

    offset: Vec2,
}

pub struct WorldAssets {
    terrain_mesh: Option<Handle<Mesh>>,
    terrain_material: Option<Handle<StandardMaterial>>,
}

fn generate_terrain_mesh(
    world_config: &WorldConfig,
    world_stats: &mut WorldStats,
) -> (Mesh, Image) {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let mut texture = Image::new(
        Extent3d {
            width: world_config.width as u32,
            height: world_config.height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        Vec::new(),
        TextureFormat::Rgba8Unorm,
    );

    let texture_data_size = world_config.width * world_config.height * 4;
    texture.data.resize(texture_data_size, 0);

    generate_terrain_perlin(&mut mesh, &mut texture.data, world_config, world_stats);

    (mesh, texture)
}

fn generate_terrain_geotif(mesh: &mut Mesh, world_config: &WorldConfig) {
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let n_vertices = world_config.width * world_config.height;

    let half_width = world_config.width / 2;
    let half_height = world_config.height / 2;

    vertices.reserve(n_vertices);
    indices.reserve(n_vertices);

    println!("Loading geotif..");

    let path = Path::new(&world_config.geo_tif.filename);
    let dataset = Dataset::open(path).unwrap();
    println!("dataset description: {:?}", dataset.description());
    let band = dataset.rasterband(1).unwrap();
    let buf = band
        .read_as::<u32>((0, 0), (256, 256), (256, 256), None)
        .unwrap();

    println!("{:?}", buf.size);

    println!("Loaded geotif");

    for y in 0..(world_config.height as i32) {
        for x in 0..(world_config.width as i32) {
            let vx = (x - half_width as i32) as f32;
            let vy = (y - half_height as i32) as f32;
            // let vz = tiff.get_value_at(y as usize, x as usize) as f32 * world_config.z_scale as f32;
            let vz = buf.data[y as usize * world_config.width + x as usize] as f32
                * world_config.z_scale as f32;
            // if vy == 0.0 {
            //     println!("v: {}, {}, {}", vx, vy, vz);
            // }

            vertices.push([vx, vy, vz]);
        }
    }

    for y in 0..world_config.height - 1 {
        for x in 0..world_config.width - 1 {
            let t0_x0 = x;
            let t0_y0 = y;
            let t0_x1 = x + 1;
            let t0_y1 = y + 1;
            let t0_x2 = x;
            let t0_y2 = y + 1;

            let t1_x0 = x + 1;
            let t1_y0 = y + 1;
            let t1_x1 = x;
            let t1_y1 = y;
            let t1_x2 = x + 1;
            let t1_y2 = y;

            let t0_index0 = (t0_y0 * world_config.width + t0_x0) as u32;
            let t0_index1 = (t0_y1 * world_config.width + t0_x1) as u32;
            let t0_index2 = (t0_y2 * world_config.width + t0_x2) as u32;

            let t1_index0 = (t1_y0 * world_config.width + t1_x0) as u32;
            let t1_index1 = (t1_y1 * world_config.width + t1_x1) as u32;
            let t1_index2 = (t1_y2 * world_config.width + t1_x2) as u32;

            indices.push(t0_index0);
            indices.push(t0_index1);
            indices.push(t0_index2);

            indices.push(t1_index0);
            indices.push(t1_index1);
            indices.push(t1_index2);
        }
    }

    mesh.set_indices(Some(Indices::U32(indices)));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);

    mesh.duplicate_vertices();
    mesh.compute_flat_normals();
}

fn generate_terrain_perlin(
    mesh: &mut Mesh,
    texture_data: &mut Vec<u8>,
    world_config: &WorldConfig,
    world_stats: &mut WorldStats,
) {
    let n_vertices = world_config.width * world_config.height;

    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n_vertices);
    let mut indices: Vec<u32> = Vec::with_capacity(n_vertices);

    let half_width = world_config.width / 2;
    let half_height = world_config.height / 2;

    let perlin = Perlin::new();
    perlin.set_seed(world_config.generator.seed as u32);
    let mut rand = ChaCha8Rng::seed_from_u64(world_config.generator.seed);

    let mut octave_offsets: Vec<Vec2> = Vec::with_capacity(world_config.generator.octaves as usize);

    let layers: Vec<TerrainLayer> = vec![
        // water deep
        TerrainLayer {
            height: 0.3,
            color: Color::rgb_u8(51, 100, 197),
        },
        // water shallow
        TerrainLayer {
            height: 0.4,
            color: Color::rgb_u8(57, 106, 203),
        },
        // sand
        TerrainLayer {
            height: 0.45,
            color: Color::rgb_u8(210, 208, 125),
        },
        // grass
        TerrainLayer {
            height: 0.55,
            color: Color::rgb_u8(86, 152, 23),
        },
        // grass 2
        TerrainLayer {
            height: 0.6,
            color: Color::rgb_u8(62, 107, 18),
        },
        // rock
        TerrainLayer {
            height: 0.7,
            color: Color::rgb_u8(90, 69, 60),
        },
        // rock 2
        TerrainLayer {
            height: 0.9,
            color: Color::rgb_u8(75, 60, 53),
        },
        // snow
        TerrainLayer {
            height: 1.0,
            color: Color::ANTIQUE_WHITE,
        },
    ];

    for _ in 0..world_config.generator.octaves {
        let offset_x = rand.gen_range(-100_000.0..=100_000.0) + world_config.offset.x;
        let offset_y = rand.gen_range(-100_000.0..=100_000.0) + world_config.offset.y;
        octave_offsets.push(Vec2::new(offset_x, offset_y));
    }

    let mut z_min = f32::MAX;
    let mut z_max = f32::MIN;

    for y in 0..(world_config.height as i32) {
        for x in 0..(world_config.width as i32) {
            let mut amplitude: f64 = 1.0;
            let mut frequency: f64 = 1.0;
            let mut z: f64 = 0.0;

            (0..world_config.generator.octaves as usize).for_each(|i| {
                let sample_x = x as f64 / world_config.generator.noise_scale * frequency
                    + octave_offsets[i].x as f64;
                let sample_y = y as f64 / world_config.generator.noise_scale * frequency
                    + octave_offsets[i].y as f64;

                let value = perlin.get([sample_x, sample_y]);

                z += value * amplitude;

                amplitude *= world_config.generator.persistance;
                frequency *= world_config.generator.lacunarity;
            });

            vertices.push([
                (x - half_width as i32) as f32,
                (y - half_height as i32) as f32,
                z as f32,
            ]);

            if (z as f32) < z_min {
                z_min = z as f32;
        }

            if (z as f32) > z_max {
                z_max = z as f32;
    }
        }
    }

    let z_diff = z_max - z_min;
    let z_diff = if z_diff > 0.0 { z_diff } else { 1.0 };

    println!("z_min: {} z_max: {} z_diff: {}", z_min, z_max, z_diff);

    for y in 0..(world_config.height as i32) {
        for x in 0..(world_config.width as i32) {
            let index = (y as usize * world_config.width + x as usize) as usize;

            let z = vertices[index][2];
            let normalized_z = z + z_min.abs();
            let normalized_z = normalized_z / z_diff;

            vertices[index][2] = z * world_config.z_scale as f32;

            let index = index * 4;

            for layer in &layers {
                if normalized_z <= layer.height {
                    texture_data[index] = (layer.color.r() * 255.0) as u8;
                    texture_data[index + 1] = (layer.color.g() * 255.0) as u8;
                    texture_data[index + 2] = (layer.color.b() * 255.0) as u8;
                    texture_data[index + 3] = (layer.color.a() * 255.0) as u8;
                    break;
                }
            }

            uvs.push([
                x as f32 / world_config.width as f32,
                y as f32 / world_config.height as f32,
            ]);
        }
    }

    for y in 0..world_config.height - 1 {
        for x in 0..world_config.width - 1 {
            let t0_x0 = x;
            let t0_y0 = y;
            let t0_x1 = x + 1;
            let t0_y1 = y + 1;
            let t0_x2 = x;
            let t0_y2 = y + 1;

            let t1_x0 = x + 1;
            let t1_y0 = y + 1;
            let t1_x1 = x;
            let t1_y1 = y;
            let t1_x2 = x + 1;
            let t1_y2 = y;

            let t0_index0 = (t0_y0 * world_config.width + t0_x0) as u32;
            let t0_index1 = (t0_y1 * world_config.width + t0_x1) as u32;
            let t0_index2 = (t0_y2 * world_config.width + t0_x2) as u32;

            let t1_index0 = (t1_y0 * world_config.width + t1_x0) as u32;
            let t1_index1 = (t1_y1 * world_config.width + t1_x1) as u32;
            let t1_index2 = (t1_y2 * world_config.width + t1_x2) as u32;

            indices.push(t0_index0);
            indices.push(t0_index1);
            indices.push(t0_index2);

            indices.push(t1_index0);
            indices.push(t1_index1);
            indices.push(t1_index2);
        }
    }

    mesh.set_indices(Some(Indices::U32(indices)));

    if mesh.contains_attribute(Mesh::ATTRIBUTE_POSITION) {
        mesh.remove_attribute(Mesh::ATTRIBUTE_POSITION);
    }
    if mesh.contains_attribute(Mesh::ATTRIBUTE_NORMAL) {
        mesh.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
    }
    if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
        mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0);
    }
    if mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT) {
        mesh.remove_attribute(Mesh::ATTRIBUTE_TANGENT);
    }
    if mesh.contains_attribute(Mesh::ATTRIBUTE_COLOR) {
        mesh.remove_attribute(Mesh::ATTRIBUTE_COLOR);
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    mesh.duplicate_vertices();
    mesh.compute_flat_normals();

    world_stats.vertices = mesh.count_vertices();
    world_stats.triangles = world_stats.vertices / 3;
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut world_assets: ResMut<WorldAssets>,
    mut world_stats: ResMut<WorldStats>,
    world_config: Res<WorldConfig>,
) {
    // commands.spawn_bundle(Camera3dBundle {
    //     transform: Transform::from_xyz(0.0, 0.0, 100.0).looking_at(Vec3::ZERO, Vec3::X),
    //     ..default()
    // });
    commands.spawn_bundle(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 250.0, 100.0).looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });

    let (terrain_mesh, terrain_image) = generate_terrain_mesh(&world_config, &mut world_stats);

    let terrain_mesh = meshes.add(terrain_mesh);
    world_assets.terrain_mesh = Some(terrain_mesh.clone());

    let terrain_image: Handle<Image> = images.add(terrain_image);

    let terrain_material: Handle<StandardMaterial> = materials.add(StandardMaterial {
        base_color_texture: Some(terrain_image),
        unlit: true,
        alpha_mode: AlphaMode::Opaque,
        ..default()
    });
    world_assets.terrain_material = Some(terrain_material.clone());

    commands
        .spawn_bundle(PbrBundle {
            mesh: terrain_mesh,
            material: terrain_material,
            ..default()
        })
        .insert(Terrain)
        .insert(Name::new("Terrain".to_string()));
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::ANTIQUE_WHITE))
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 1.0 / 5.0f32,
            // brightness: 1.0,
        })
        .insert_resource(WindowDescriptor {
            title: "WorldRenderer".to_string(),
            width: 1920.0,
            height: 1080.0,
            // position: WindowPosition::At(Vec2::new(0.0, 50.0)),
            position: WindowPosition::Centered(MonitorSelection::Primary),
            present_mode: PresentMode::Immediate,
            ..default()
        })
        .insert_resource(WgpuSettings {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..default()
        })
        .insert_resource(ImageSettings::default_nearest())
        .insert_resource(WorldConfig {
            width: 512,
            height: 512,
            z_scale: 50.0,
            auto_update: true,
            wireframe: false,
            world_type: ConfigType::GeoTif,
            generator: GeneratorConfig {
                seed: 0,
                noise_scale: 100.0,
                octaves: 8,
                persistance: 0.5,
                lacunarity: 2.0,
            },
            geo_tif: GeoTifConfig {
                filename: "data/ISKO_PL_4326_5m_N54_E18.tif".to_string(),
            },
            offset: Vec2::ZERO,
        })
        .insert_resource(WorldAssets {
            terrain_mesh: None,
            terrain_material: None,
        })
        .insert_resource(WorldStats {
            triangles: 0,
            vertices: 0,
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(WireframePlugin)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .add_system(world_generator_ui)
        .add_system(debug_stats_ui)
        .add_system(bevy::window::close_on_esc)
        .run();
}

pub fn world_generator_ui(
    mut egui_context: ResMut<EguiContext>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut world_config: ResMut<WorldConfig>,
    mut world_stats: ResMut<WorldStats>,
    world_assets: Res<WorldAssets>,
) {
    egui::Window::new("Terrain")
        .anchor(Align2::RIGHT_BOTTOM, egui::vec2(-15.0, -15.0))
        .show(egui_context.ctx_mut(), |ui| {
            let old_world = world_config.clone();
            let mut changed = false;

            ui.separator();

            egui::Grid::new("GeneratorGrid").show(ui, |ui| {
                ui.label("Width:");
                ui.add(egui::Slider::new(&mut world_config.width, 128..=2048).clamp_to_range(true));
                ui.end_row();

                ui.label("Height:");
                ui.add(
                    egui::Slider::new(&mut world_config.height, 128..=2048).clamp_to_range(true),
                );
                ui.end_row();

                ui.label("Height Scale:");
                ui.add(
                    egui::Slider::new(&mut world_config.z_scale, 0.001..=128.0)
                        .clamp_to_range(true),
                );
                ui.end_row();

                ui.label("Noise Scale:");
                ui.add(
                    egui::Slider::new(&mut world_config.generator.noise_scale, 0.001..=128.0)
                        .clamp_to_range(true),
                );
                ui.end_row();

                ui.label("Seed:");
                ui.add(egui::DragValue::new(&mut world_config.generator.seed));
                ui.end_row();

                ui.label("Octaves:");
                ui.add(
                    egui::Slider::new(&mut world_config.generator.octaves, 0..=10)
                        .clamp_to_range(true),
                );
                ui.end_row();

                ui.label("Persistance:");
                ui.add(
                    egui::Slider::new(&mut world_config.generator.persistance, 0.0..=1.0)
                        .clamp_to_range(true),
                );
                ui.end_row();

                ui.label("Lacunarity:");
                ui.add(
                    egui::Slider::new(&mut world_config.generator.lacunarity, 1.0..=10.0)
                        .clamp_to_range(true),
                );
                ui.end_row();

                ui.label("Offset:");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut world_config.offset.x));
                    ui.add(egui::DragValue::new(&mut world_config.offset.y));
                });
                ui.end_row();

                ui.checkbox(&mut world_config.auto_update, "Auto update");
                if ui
                    .button(RichText::new("Generate").color(Color32::LIGHT_GREEN))
                    .clicked()
                {
                    changed = true;
                }
                ui.end_row();
            });

            changed |= world_config.auto_update && *world_config != old_world;

            if changed {
                let terrain_mesh = world_assets.terrain_mesh.clone();
                let terrain_mesh = terrain_mesh.unwrap();
                let mesh = meshes.get_mut(&terrain_mesh);
                let mesh = mesh.unwrap();

                let terrain_material = world_assets.terrain_material.clone();
                let terrain_material = terrain_material.unwrap();
                let material = materials.get_mut(&terrain_material);
                let material = material.unwrap();
                let image_handle = material.base_color_texture.clone();
                let image_handle = image_handle.unwrap();
                let opt_image = images.get_mut(&image_handle);
                let image = opt_image.unwrap();

                if world_config.width != old_world.width || world_config.height != old_world.height
                {
                    let texture_data_size = world_config.width * world_config.height * 4;
                    image.texture_descriptor.size = Extent3d {
                        width: world_config.width as u32,
                        height: world_config.height as u32,
                        depth_or_array_layers: 1,
                    };
                    image.data.resize(texture_data_size, 0);
                }

                generate_terrain_perlin(mesh, &mut image.data, &world_config, &mut world_stats);
            }
        });
}

pub fn debug_stats_ui(
    mut egui_context: ResMut<EguiContext>,
    mut world_config: ResMut<WorldConfig>,
    world_stats: Res<WorldStats>,
    query_terrain: Query<(Entity, &Terrain, Option<&Wireframe>)>,
    diagnostics: Res<Diagnostics>,
) {
    let fps = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS).unwrap();
    let frame_count = diagnostics
        .get(FrameTimeDiagnosticsPlugin::FRAME_COUNT)
        .unwrap();
    let frame_time = diagnostics
        .get(FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .unwrap();

    let current_fps = if let Some(fps) = fps.measurement() {
        fps.value as u32
    } else {
        0
    };
    let average_fps = if let Some(fps) = fps.measurement() {
        fps.value as u32
    } else {
        0
    };
    let frame_time = if let Some(frame_time) = frame_time.measurement() {
        frame_time.value * 1000.0
    } else {
        0.0
    };
    let frame_count = if let Some(frame_count) = frame_count.measurement() {
        frame_count.value as u32
    } else {
        0
    };

    let wireframe = world_config.wireframe;

    egui::Window::new("Stats")
        .anchor(Align2::RIGHT_TOP, egui::vec2(-15.0, 15.0))
        .show(egui_context.ctx_mut(), |ui| {
            egui::Grid::new("StatsGrid").show(ui, |ui| {
                ui.label("FPS");
                ui.label(format!("{}", current_fps));
                ui.end_row();

                ui.label("FPS (avg)");
                ui.label(format!("{}", average_fps));
                ui.end_row();

                ui.label("Frame (ms)");
                ui.label(format!("{:.2}", frame_time));
                ui.end_row();

                ui.label("Frame");
                ui.label(format!("{}", frame_count));
                ui.end_row();

                ui.label("Vertices");
                ui.label(format!("{}", world_stats.vertices));
                ui.end_row();

                ui.label("Triangles");
                ui.label(format!("{}", world_stats.triangles));
                ui.end_row();

                // ui.label("Wireframe");
                ui.checkbox(&mut world_config.wireframe, "Wireframe");
                ui.end_row();
            });
        });

    if wireframe != world_config.wireframe {
        for (entity, terrain, wireframe_tag) in query_terrain.iter() {
            if let Some(wireframe_tag) = wireframe_tag {
            } else {
            }
        }
    }
}
