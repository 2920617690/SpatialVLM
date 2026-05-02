import json
import math
import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError("Expected '-- <config_path>' after Blender arguments.")
    idx = argv.index("--")
    if idx + 1 >= len(argv):
        raise RuntimeError("Missing config path after '--'.")
    return Path(argv[idx + 1]).resolve()


CONFIG = json.loads(_parse_args().read_text(encoding="utf-8"))
OUTPUT_ROOT = Path(CONFIG["output"]["root"]).resolve()


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render_settings():
    scene = bpy.context.scene
    render_cfg = CONFIG["render"]
    engine = render_cfg["engine"]
    try:
        scene.render.engine = engine
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = render_cfg["resolution_x"]
    scene.render.resolution_y = render_cfg["resolution_y"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = render_cfg.get("transparent_background", False)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    scene.view_settings.exposure = -0.8
    scene.view_settings.gamma = 1.0
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = render_cfg["samples"]
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = render_cfg["samples"]


def add_ground_and_lights(rng):
    ground_color = tuple(rng.choice(CONFIG["style"]["ground_palette"]))
    bg_color = tuple(rng.choice(CONFIG["style"]["background_palette"]))

    bpy.ops.mesh.primitive_plane_add(size=24.0, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "GroundPlane"
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if bsdf:
        bsdf.inputs["Base Color"].default_value = ground_color
        bsdf.inputs["Roughness"].default_value = 0.95
        bsdf.inputs["Specular IOR Level"].default_value = 0.15
    plane.data.materials.append(mat)

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = bg_color
        bg_node.inputs["Strength"].default_value = 0.28

    bpy.ops.object.light_add(type="SUN", location=(4, -5, 8))
    sun = bpy.context.object
    sun.data.energy = 1.3 + rng.random() * 0.6
    sun.rotation_euler = Euler((math.radians(46 + rng.uniform(-8, 8)), math.radians(rng.uniform(-6, 6)), math.radians(rng.uniform(18, 76))))

    bpy.ops.object.light_add(type="AREA", location=(-5, 4, 5))
    fill = bpy.context.object
    fill.data.energy = 220.0 + rng.random() * 90.0
    fill.data.size = 7.0
    fill.rotation_euler = Euler((math.radians(68), 0, math.radians(-122)))

    bpy.ops.object.light_add(type="AREA", location=(4, -3, 4))
    rim = bpy.context.object
    rim.data.energy = 90.0 + rng.random() * 30.0
    rim.data.size = 4.0
    rim.rotation_euler = Euler((math.radians(70), 0, math.radians(45)))


def _pick_camera_mode(scene_index):
    cycle = CONFIG["scene"].get("camera_mode_cycle")
    if cycle:
        return cycle[scene_index % len(cycle)]
    return "oblique"


def add_camera(rng, scene_index):
    scene_cfg = CONFIG["scene"]
    mode = _pick_camera_mode(scene_index)
    if mode == "topdown":
        az = math.radians(rng.uniform(*scene_cfg["topdown_azimuth_range"]))
        el = math.radians(rng.uniform(*scene_cfg["topdown_elevation_range"]))
        dist = rng.uniform(*scene_cfg["topdown_distance_range"])
        lens = 44
        look_at = Vector((0.0, 0.0, 0.55))
    else:
        az = math.radians(rng.uniform(*scene_cfg["camera_azimuth_range"]))
        el = math.radians(rng.uniform(*scene_cfg["camera_elevation_range"]))
        dist = rng.uniform(*scene_cfg["camera_distance_range"])
        lens = 38
        look_at = Vector((0.0, 0.0, 0.7))
    x = dist * math.cos(el) * math.cos(az)
    y = dist * math.cos(el) * math.sin(az)
    z = dist * math.sin(el) + 0.6

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.location = (x, y, z)
    direction = look_at - Vector((x, y, z))
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = lens
    camera.data.clip_start = 0.1
    camera.data.clip_end = 100.0
    return camera, mode


def _sample_positions(rng, num_objects):
    scene_cfg = CONFIG["scene"]
    min_r = scene_cfg["placement_radius_min"]
    max_r = scene_cfg["placement_radius_max"]
    min_pair = scene_cfg["min_pair_distance"]
    layout_cycle = scene_cfg.get("layout_modes") or ["ring"]
    layout_mode = rng.choice(layout_cycle)

    if layout_mode == "line":
        positions = []
        base_y = rng.uniform(-1.0, 1.0)
        xs = sorted(rng.uniform(-max_r, max_r) for _ in range(num_objects))
        jitter_y = [base_y + rng.uniform(-0.45, 0.45) for _ in range(num_objects)]
        for x, y in zip(xs, jitter_y):
            positions.append((x, y))
        return positions, layout_mode

    if layout_mode == "arc":
        positions = []
        start_angle = rng.uniform(-1.5, -0.5)
        end_angle = rng.uniform(0.5, 1.5)
        angles = [start_angle + (end_angle - start_angle) * i / max(num_objects - 1, 1) for i in range(num_objects)]
        for angle in angles:
            radius = rng.uniform((min_r + max_r) * 0.45, max_r)
            positions.append((radius * math.cos(angle), radius * math.sin(angle)))
        return positions, layout_mode

    if layout_mode == "cluster_pair":
        positions = []
        cluster_centers = [(-1.8, 0.8), (1.8, -0.6)]
        for i in range(num_objects):
            cx, cy = cluster_centers[i % 2]
            positions.append((cx + rng.uniform(-0.7, 0.7), cy + rng.uniform(-0.7, 0.7)))
        return positions, layout_mode

    if layout_mode == "corner_spread":
        anchors = [(-max_r * 0.7, -max_r * 0.5), (max_r * 0.65, -max_r * 0.45), (-max_r * 0.45, max_r * 0.65), (max_r * 0.55, max_r * 0.55)]
        rng.shuffle(anchors)
        positions = []
        for i in range(num_objects):
            ax, ay = anchors[i % len(anchors)]
            positions.append((ax + rng.uniform(-0.45, 0.45), ay + rng.uniform(-0.45, 0.45)))
        return positions, layout_mode

    positions = []
    for _ in range(num_objects):
        for _ in range(300):
            angle = rng.uniform(0, math.pi * 2)
            radius = rng.uniform(min_r, max_r)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            if all(math.dist((x, y), (px, py)) >= min_pair for px, py in positions):
                positions.append((x, y))
                break
    return positions, layout_mode


def _material_setup(obj, color_rgba, material_cfg):
    mat = bpy.data.materials.new(name=f"{obj.name}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if bsdf:
        bsdf.inputs["Base Color"].default_value = tuple(color_rgba)
        bsdf.inputs["Metallic"].default_value = material_cfg["metallic"]
        bsdf.inputs["Roughness"].default_value = material_cfg["roughness"]
        bsdf.inputs["Specular IOR Level"].default_value = 0.45 if material_cfg["name"] == "rubber" else 0.65
        bsdf.inputs["Coat Weight"].default_value = 0.03 if material_cfg["name"] == "rubber" else 0.0
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def _create_primitive(shape, location, scale_value, rotation_z_deg):
    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(location=(location[0], location[1], scale_value))
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(location=(location[0], location[1], scale_value), segments=48, ring_count=24)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(location=(location[0], location[1], scale_value), vertices=48)
    elif shape == "cone":
        bpy.ops.mesh.primitive_cone_add(location=(location[0], location[1], scale_value), vertices=48)
    else:
        raise ValueError(f"Unknown primitive shape: {shape}")
    obj = bpy.context.object
    obj.scale = (scale_value, scale_value, scale_value)
    obj.rotation_euler[2] = math.radians(rotation_z_deg)
    return obj


def _project_bbox(obj, camera, scene):
    points = []
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        co_ndc = world_to_camera_view(scene, camera, world_corner)
        points.append(co_ndc)
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    if max(xs) < 0 or min(xs) > 1 or max(ys) < 0 or min(ys) > 1:
        return None
    x1 = max(0.0, min(xs))
    x2 = min(1.0, max(xs))
    y1 = max(0.0, min(ys))
    y2 = min(1.0, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, 1.0 - y2, x2, 1.0 - y1]


def _center_from_bbox(bbox):
    return [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]


def build_scene(split, scene_index, rng):
    clear_scene()
    setup_render_settings()
    add_ground_and_lights(rng)
    camera, camera_mode = add_camera(rng, scene_index)
    scene = bpy.context.scene

    obj_cfg = CONFIG["objects"]
    num_objects = rng.randint(CONFIG["scene"]["min_objects"], CONFIG["scene"]["max_objects"])
    positions, layout_mode = _sample_positions(rng, num_objects)
    shape_choices = rng.choices(obj_cfg["shapes"], k=num_objects)
    size_choices = rng.choices(obj_cfg["sizes"], k=num_objects)
    palette = CONFIG["style"]["object_palette"]
    color_choices = rng.sample(palette, k=min(len(palette), num_objects))
    material_choices = rng.choices(CONFIG["style"]["materials"], k=num_objects)

    scene_objects = []
    used_descriptors = set()
    for index, ((x, y), shape, size_cfg, material_cfg) in enumerate(zip(positions, shape_choices, size_choices, material_choices)):
        color_cfg = color_choices[index % len(color_choices)]
        descriptor = f"{color_cfg['name']} {material_cfg['name']} {shape}"
        if descriptor in used_descriptors:
            alt_choices = [item for item in palette if f"{item['name']} {material_cfg['name']} {shape}" not in used_descriptors]
            if alt_choices:
                color_cfg = rng.choice(alt_choices)
                descriptor = f"{color_cfg['name']} {material_cfg['name']} {shape}"
        rot_z = rng.uniform(0, 360)
        obj = _create_primitive(shape, (x, y), size_cfg["scale"], rot_z)
        _material_setup(obj, color_cfg["rgba"], material_cfg)
        bbox = _project_bbox(obj, camera, scene)
        if bbox is None:
            continue
        used_descriptors.add(descriptor)
        scene_objects.append(
            {
                "object_id": f"obj_{index}",
                "shape": shape,
                "material": material_cfg["name"],
                "size": size_cfg["name"],
                "color": color_cfg["name"],
                "descriptor": descriptor,
                "bbox": bbox,
                "center": _center_from_bbox(bbox),
                "location_3d": [x, y, size_cfg["scale"]],
                "scale_3d": [size_cfg["scale"], size_cfg["scale"], size_cfg["scale"]],
                "rotation_z": rot_z
            }
        )
    if len(scene_objects) < 4:
        raise RuntimeError("Scene has too few visible primitive objects.")

    scene_id = f"{split}_{scene_index:06d}"
    image_path = OUTPUT_ROOT / "images" / split / f"{scene_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)

    metadata = {
        "scene_id": scene_id,
        "split": split,
        "image_path": str(image_path),
        "scene_objects": scene_objects,
        "camera": {
            "mode": camera_mode,
            "location": list(camera.location),
            "rotation_euler": list(camera.rotation_euler),
        },
        "layout_mode": layout_mode,
    }
    metadata_path = OUTPUT_ROOT / "scene_metadata" / split / f"{scene_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    seed = CONFIG["seed"]
    split_offsets = {"train": 0, "val": 1000, "test": 2000}
    for split, count in CONFIG["splits"].items():
        rng = random.Random(seed + split_offsets.get(split, 5000))
        success = 0
        attempts = 0
        max_attempts = max(count * 5, 20)
        while success < count and attempts < max_attempts:
            attempts += 1
            try:
                build_scene(split, success, rng)
                print(f"[blender-clevr] rendered split={split} scene={success} attempt={attempts}")
                success += 1
            except Exception as exc:
                print(f"[blender-clevr] retry split={split} attempt={attempts} reason={exc}")
        if success < count:
            raise RuntimeError(f"Failed to render enough scenes for split={split}: got {success}, need {count}")


if __name__ == "__main__":
    main()
