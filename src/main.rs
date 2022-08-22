use std::path::Path;

use bevy::{
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
    math::Vec2,
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    render::{
        mesh::Indices,
        render_resource::PrimitiveTopology,
        settings::{WgpuFeatures, WgpuSettings},
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
}

fn generate_terrain_mesh(world_config: &WorldConfig, world_stats: &mut WorldStats) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    generate_terrain_perlin(&mut mesh, world_config);

    mesh
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
    world_config: &WorldConfig,
    world_stats: &mut WorldStats,
) {
    let n_vertices = world_config.width * world_config.height;

    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
    let mut indices: Vec<u32> = Vec::with_capacity(n_vertices);

    let half_width = world_config.width / 2;
    let half_height = world_config.height / 2;

    let perlin = Perlin::new();
    perlin.set_seed(world_config.generator.seed as u32);
    let mut rand = ChaCha8Rng::seed_from_u64(world_config.generator.seed);

    let mut octave_offsets: Vec<Vec2> = Vec::with_capacity(world_config.generator.octaves as usize);

    for _ in 0..world_config.generator.octaves {
        let offset_x = rand.gen_range(-100_000.0..=100_000.0) + world_config.offset.x;
        let offset_y = rand.gen_range(-100_000.0..=100_000.0) + world_config.offset.y;
        octave_offsets.push(Vec2::new(offset_x, offset_y));
    }

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
                (z * world_config.z_scale) as f32,
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
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);

    mesh.duplicate_vertices();
    mesh.compute_flat_normals();

    world_stats.vertices = mesh.count_vertices();
    world_stats.triangles = world_stats.vertices / 3;
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
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

    let terrain_mesh = meshes.add(generate_terrain_mesh(&world_config, &mut world_stats));

    world_assets.terrain_mesh = Some(terrain_mesh.clone());

    // commands.spawn_bundle(InfiniteGridBundle {
    //     grid: InfiniteGrid {
    //         ..Default::default()
    //     },
    //     ..Default::default()
    // });

    commands
        .spawn_bundle(PbrBundle {
            mesh: terrain_mesh,
            material: materials.add(Color::ALICE_BLUE.into()),
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
        .insert_resource(WorldAssets { terrain_mesh: None })
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
    mut world_config: ResMut<WorldConfig>,
    mut world_stats: ResMut<WorldStats>,
    world_assets: Res<WorldAssets>,
) {
    egui::Window::new("Terrain")
        .anchor(Align2::RIGHT_BOTTOM, egui::vec2(-15.0, -15.0))
        .show(egui_context.ctx_mut(), |ui| {
            let old_world = world_config.clone();

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
                    let terrain_mesh = world_assets.terrain_mesh.clone();
                    let terrain_mesh = terrain_mesh.unwrap();
                    let mesh = meshes.get_mut(&terrain_mesh);
                    let mesh = mesh.unwrap();
                    generate_terrain_perlin(mesh, &world_config);
                }
                ui.end_row();
            });

            let changed = *world_config != old_world;

            if changed && world_config.auto_update {
                let terrain_mesh = world_assets.terrain_mesh.clone();
                let terrain_mesh = terrain_mesh.unwrap();
                let mesh = meshes.get_mut(&terrain_mesh);
                let mesh = mesh.unwrap();
                generate_terrain_perlin(mesh, &world_config);
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
