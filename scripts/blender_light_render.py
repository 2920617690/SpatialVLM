import json
import math
import os
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


def _load_config():
    path = _parse_args()
    return json.loads(path.read_text(encoding="utf-8"))


CONFIG = _load_config()
ASSET_ROOT = Path(CONFIG["asset_root"]).resolve()
OUTPUT_ROOT = Path(CONFIG["output"]["root"]).resolve()
MANIFEST = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
ALLOWED_ASSET_KEYS = CONFIG.get("allowed_asset_keys") or list(MANIFEST.keys())


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
    available_engines = {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"}
    if engine not in available_engines:
        engine = "BLENDER_EEVEE_NEXT"
    try:
        scene.render.engine = engine
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = render_cfg["resolution_x"]
    scene.render.resolution_y = render_cfg["resolution_y"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = render_cfg.get("transparent_background", False)
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = render_cfg["samples"]
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = render_cfg["samples"]


def add_ground_and_lights(rng):
    ground_color = tuple(rng.choice(CONFIG["style"]["ground_palette"]))
    bg_color = tuple(rng.choice(CONFIG["style"]["background_palette"]))

    bpy.ops.mesh.primitive_plane_add(size=22.0, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "GroundPlane"
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = ground_color
        bsdf.inputs["Roughness"].default_value = 0.85
    plane.data.materials.append(mat)

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = bg_color
        bg_node.inputs["Strength"].default_value = 0.9

    bpy.ops.object.light_add(type="SUN", location=(5, -5, 8))
    sun = bpy.context.object
    sun.data.energy = 3.0 + rng.random() * 1.5
    sun.rotation_euler = Euler((math.radians(45 + rng.uniform(-10, 12)), math.radians(rng.uniform(-10, 10)), math.radians(rng.uniform(10, 80))))

    bpy.ops.object.light_add(type="AREA", location=(-4, 3, 6))
    fill = bpy.context.object
    fill.data.energy = 40.0 + rng.random() * 30.0
    fill.data.size = 5.0
    fill.rotation_euler = Euler((math.radians(60), 0, math.radians(-120)))


def add_camera(rng):
    scene_cfg = CONFIG["scene"]
    az = math.radians(rng.uniform(*scene_cfg["camera_azimuth_range"]))
    el = math.radians(rng.uniform(*scene_cfg["camera_elevation_range"]))
    dist = rng.uniform(*scene_cfg["camera_distance_range"])
    look_at = Vector((0.0, 0.0, 0.9))
    x = dist * math.cos(el) * math.cos(az)
    y = dist * math.cos(el) * math.sin(az)
    z = dist * math.sin(el) + 0.6

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.location = (x, y, z)
    direction = look_at - Vector((x, y, z))
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 35
    camera.data.clip_start = 0.1
    camera.data.clip_end = 100.0
    return camera


def _sample_positions(rng, num_objects):
    scene_cfg = CONFIG["scene"]
    min_r = scene_cfg["placement_radius_min"]
    max_r = scene_cfg["placement_radius_max"]
    min_pair = scene_cfg["min_pair_distance"]
    positions = []
    for _ in range(num_objects):
        for _ in range(200):
            angle = rng.uniform(0, math.pi * 2)
            radius = rng.uniform(min_r, max_r)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            if all(math.dist((x, y), (px, py)) >= min_pair for px, py in positions):
                positions.append((x, y))
                break
    return positions


def _category_from_asset(asset_key):
    if asset_key.startswith("human_"):
        return "person"
    return asset_key


def _ensure_principled_material(obj, rgba):
    mesh_objects = [child for child in obj.children_recursive if child.type == "MESH"]
    if obj.type == "MESH":
        mesh_objects.append(obj)
    for mesh_obj in mesh_objects:
        if not mesh_obj.data.materials:
            mat = bpy.data.materials.new(name=f"{mesh_obj.name}_mat")
            mat.use_nodes = True
            mesh_obj.data.materials.append(mat)
        for mat in mesh_obj.data.materials:
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = tuple(rgba)
                bsdf.inputs["Roughness"].default_value = 0.55


def import_glb(filepath, location=(0, 0, 0), scale=1.0, rotation_z_deg=0):
    existing = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(filepath))
    new_objects = [obj for name, obj in bpy.data.objects.items() if name not in existing]
    if not new_objects:
        return None
    new_set = set(id(o) for o in new_objects)
    roots = [o for o in new_objects if o.parent is None or id(o.parent) not in new_set]
    if len(roots) > 1:
        bpy.ops.object.empty_add(location=location)
        parent_empty = bpy.context.object
        for r in roots:
            r.parent = parent_empty
        target = parent_empty
    else:
        target = roots[0]

    coords = []
    for obj in new_objects:
        if obj.type == "MESH":
            for corner in obj.bound_box:
                coords.append(obj.matrix_world @ Vector(corner))
    if coords:
        xs = [c.x for c in coords]
        ys = [c.y for c in coords]
        zs = [c.z for c in coords]
        bbox_size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        eff_scale = (2.0 / bbox_size) * scale if bbox_size > 0 else scale
    else:
        eff_scale = scale

    target.scale = (eff_scale, eff_scale, eff_scale)
    target.location = location
    target.rotation_euler[2] = math.radians(rotation_z_deg)
    return target


def _project_bbox(obj, camera, scene):
    points = []
    mesh_objects = [child for child in obj.children_recursive if child.type == "MESH"]
    if obj.type == "MESH":
        mesh_objects.append(obj)
    for mesh_obj in mesh_objects:
        for corner in mesh_obj.bound_box:
            world_corner = mesh_obj.matrix_world @ Vector(corner)
            co_ndc = world_to_camera_view(scene, camera, world_corner)
            points.append(co_ndc)
    if not points:
        return None
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


def _bbox_area(bbox):
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _center_from_bbox(bbox):
    return [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]


def build_scene(split, scene_index, rng):
    clear_scene()
    setup_render_settings()
    add_ground_and_lights(rng)
    camera = add_camera(rng)
    scene = bpy.context.scene

    asset_keys = [key for key in ALLOWED_ASSET_KEYS if key in MANIFEST]
    if len(asset_keys) == 0:
        raise RuntimeError("No valid asset keys available for Blender-light rendering.")
    num_objects = rng.randint(CONFIG["scene"]["min_objects"], CONFIG["scene"]["max_objects"])
    positions = _sample_positions(rng, num_objects)
    chosen = rng.choices(asset_keys, k=num_objects)

    scene_objects = []
    palette = CONFIG["style"]["object_palette"]
    color_choices = rng.sample(palette, k=min(len(palette), num_objects))
    for index, (asset_key, (x, y)) in enumerate(zip(chosen, positions)):
        asset_info = MANIFEST[asset_key]
        scale = rng.uniform(0.8, 1.25)
        rot_z = rng.uniform(0, 360)
        target = import_glb(ASSET_ROOT / asset_info["filename"], location=(x, y, 0.0), scale=scale, rotation_z_deg=rot_z)
        if target is None:
            continue
        color_info = color_choices[index % len(color_choices)]
        _ensure_principled_material(target, color_info["rgba"])
        bbox = _project_bbox(target, camera, scene)
        if bbox is None:
            continue
        if _bbox_area(bbox) > 0.80:
            continue
        center = _center_from_bbox(bbox)
        if any(
            abs(existing["center"][0] - center[0]) < 1.0e-3
            and abs(existing["center"][1] - center[1]) < 1.0e-3
            for existing in scene_objects
        ):
            continue
        scene_objects.append(
            {
                "object_id": f"obj_{index}",
                "asset_key": asset_key,
                "category": _category_from_asset(asset_key),
                "descriptor": f"{color_info['name']} {_category_from_asset(asset_key)}",
                "color": color_info["name"],
                "shape": _category_from_asset(asset_key),
                "size": "medium",
                "bbox": bbox,
                "center": center,
                "location_3d": [x, y, 0.0],
                "scale_3d": [target.scale[0], target.scale[1], target.scale[2]],
                "rotation_z": rot_z
            }
        )
    if len(scene_objects) < 3:
        raise RuntimeError("Scene has too few visible objects after rendering setup.")

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
            "location": list(camera.location),
            "rotation_euler": list(camera.rotation_euler),
        },
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
        max_attempts = max(count * 20, 40)
        while success < count and attempts < max_attempts:
            attempts += 1
            try:
                build_scene(split, success, rng)
                print(f"[blender-light] rendered split={split} scene={success} attempt={attempts}")
                success += 1
            except Exception as exc:
                print(f"[blender-light] retry split={split} attempt={attempts} reason={exc}")
        if success < count:
            raise RuntimeError(f"Failed to render enough scenes for split={split}: got {success}, need {count}")


if __name__ == "__main__":
    main()
