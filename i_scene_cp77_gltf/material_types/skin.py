from ..assetio.paths import backslash_depot_key
from ..bartmoss.scalar import clamp01
from ..blender import shader_nodes
from ..blender.shader_nodes import assign_if_changed, set_default_if_changed
from ..core.log import get_logger
from .bundles import smoothstep_socket, tangent_normal_blend_socket
from .graph import (
    mix_color_socket,
    mix_scalar_socket,
    separate_color_node,
)
from .mat_common import (
    MaterialTypeBase,
    add_group_node,
    create_relative_texture_node,
    find_input,
    get_or_build_node_group,
    new_labeled_node,
    param_color,
    param_float,
    param_texture_path,
)
from .normal_math import (
    decode_cp77_normal,
    encode_tangent_normal,
    tangent_to_normal_map,
)
from .skin_regions import DEFAULT_FACE_REGIONSET_PATH
from .skin_wrinkles import (
    WRINKLE_REGION_COUNT,
    blend_wrinkle_normal,
    build_wrinkle_influence,
)

_LOG = get_logger(__name__)

_WHITE = (1.0, 1.0, 1.0, 1.0)


def _source_xy(tree, color, label, location):
    split = separate_color_node(tree.nodes, tree.links, color, f"{label} RG", location)
    x2 = shader_nodes.math(
        tree.nodes,
        tree.links,
        "MULTIPLY",
        split.outputs["Red"],
        2.0,
        name=f"{label} X×2",
        location=(location[0] + 180, location[1] + 40),
    )
    x = shader_nodes.math(
        tree.nodes,
        tree.links,
        "SUBTRACT",
        x2,
        1.0,
        name=f"{label} X",
        location=(location[0] + 360, location[1] + 40),
    )
    y2 = shader_nodes.math(
        tree.nodes,
        tree.links,
        "MULTIPLY",
        split.outputs["Green"],
        2.0,
        name=f"{label} Y×2",
        location=(location[0] + 180, location[1] - 60),
    )
    y = shader_nodes.math(
        tree.nodes,
        tree.links,
        "SUBTRACT",
        y2,
        1.0,
        name=f"{label} Y",
        location=(location[0] + 360, location[1] - 60),
    )
    return x, y


def _atlas_uv(tree, uv, scale, region_scale, frame, label, location):
    nodes = tree.nodes
    links = tree.links
    scale_factor = shader_nodes.math(
        nodes,
        links,
        "ADD",
        region_scale,
        1.4 if frame == 0 else 1.0,
        name=f"{label} Region Scale",
        location=location,
    )
    total_scale = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        scale,
        scale_factor,
        name=f"{label} UV Scale",
        location=(location[0] + 160, location[1]),
    )
    scaled = shader_nodes.vector_math(
        nodes,
        links,
        "SCALE",
        uv,
        scale=total_scale,
        name=f"{label} Scaled UV",
        location=(location[0] + 320, location[1]),
    )
    frac = shader_nodes.vector_math(
        nodes,
        links,
        "FRACTION",
        scaled,
        name=f"{label} Repeat",
        location=(location[0] + 480, location[1]),
    )
    half = shader_nodes.vector_math(
        nodes,
        links,
        "MULTIPLY",
        frac,
        (0.5, 1.0, 1.0),
        name=f"{label} Atlas Half",
        location=(location[0] + 640, location[1]),
    )
    if frame == 0:
        return half
    return shader_nodes.vector_math(
        nodes,
        links,
        "ADD",
        half,
        (0.5, 0.0, 0.0),
        name=f"{label} Atlas Offset",
        location=(location[0] + 800, location[1]),
    )


# Texture parameters a skin samples (each named after its parameter), and whether each is non-colour data.
_TEXTURES = (
    ("Roughness", True),
    ("TintColorMask", True),
    ("Normal", True),
    ("Detailmap_Stretch", True),
    ("Detailmap_Squash", True),
    ("DetailNormal", True),
    ("Albedo", False),
    ("SecondaryAlbedo", False),
    ("Bloodflow", True),
    ("EmissiveMask", True),
)
_WRINKLE_TEXTURES = ("Detailmap_Stretch", "Detailmap_Squash", "Bloodflow")

SKIN_GROUP_NAME = "CP77 Skin v1"
MICRODETAIL_UV_GROUP_NAME = "CP77 Skin MicroDetail UV v1"
# Names of the skin's own nodes in a material, which apply_structure_values looks up.
_SKIN_NODE = "Skin"
_MICRODETAIL_UV_NODE = "MicroDetail UV"
_MICRODETAIL_NODES = ("MicroDetail 01", "MicroDetail 02")

_FLAT_NORMAL = (0.5, 0.5, 1.0, 1.0)

# Texture inputs of the skin group with the value each takes when its texture is missing. Each default reproduces
# what the shader assumes without that texture: roughness 0.5 with no metalness or detail mask, full tint, flat
# normals, no blood, an unmasked emissive.
_GROUP_TEXTURE_INPUTS = (
    ("Albedo", _WHITE),
    ("Roughness", (0.5, 0.0, 0.0, 1.0)),
    ("TintColorMask", (1.0, 0.0, 0.0, 1.0)),
    ("Normal", _FLAT_NORMAL),
    ("Detailmap_Stretch", _FLAT_NORMAL),
    ("Detailmap_Squash", _FLAT_NORMAL),
    ("DetailNormal", _FLAT_NORMAL),
    ("MicroDetail 01", _FLAT_NORMAL),
    ("MicroDetail 02", _FLAT_NORMAL),
    ("SecondaryAlbedo", _WHITE),
    ("Bloodflow", (0.0, 0.0, 0.0, 1.0)),
    ("EmissiveMask", _WHITE),
)

# Value inputs of the skin group, named after the material parameters: (socket type, name, default, min, max).
_GROUP_VALUE_INPUTS = (
    ("NodeSocketFloat", "SecondaryAlbedo Alpha", 0.0, 0.0, 1.0),
    ("NodeSocketColor", "TintColor", _WHITE, None, None),
    ("NodeSocketFloat", "TintScale", 0.0, -10.0, 10.0),
    ("NodeSocketFloat", "DetailNormalInfluence", 1.0, 0.0, 1.0),
    ("NodeSocketFloat", "MicroDetailInfluence", 0.0, 0.0, 1.0),
    ("NodeSocketFloat", "DetailRoughnessBiasMin", 0.0, 0.0, 1.0),
    ("NodeSocketFloat", "DetailRoughnessBiasMax", 1.0, 0.0, 1.0),
    ("NodeSocketFloat", "CavityIntensity", 0.0, 0.0, 10.0),
    ("NodeSocketFloat", "SecondaryAlbedoTintColorInfluence", 0.0, 0.0, 1.0),
    ("NodeSocketFloat", "SecondaryAlbedoInfluence", 0.0, 0.0, 1.0),
    ("NodeSocketColor", "BloodColor", (0.0, 0.0, 0.0, 1.0), None, None),
    ("NodeSocketFloat", "EmissiveEV", 0.0, -16.0, 16.0),
    # 1 when the material has a microdetail texture / wrinkle maps; 0 turns that part of the shader off.
    ("NodeSocketFloat", "Use MicroDetail", 0.0, 0.0, 1.0),
    ("NodeSocketFloat", "Use Wrinkles", 0.0, 0.0, 1.0),
    ("NodeSocketFloat", "Use Stretch Map", 0.0, 0.0, 1.0),
    ("NodeSocketFloat", "Use Squash Map", 0.0, 0.0, 1.0),
    # 1 when the material has a Bloodflow texture and wrinkle maps to drive it.
    ("NodeSocketFloat", "Use Bloodflow", 0.0, 0.0, 1.0),
)

_GROUP_OUTPUTS = (
    ("NodeSocketColor", "Base Color"),
    ("NodeSocketFloat", "Metallic"),
    ("NodeSocketFloat", "Roughness"),
    ("NodeSocketVector", "Normal"),
    ("NodeSocketFloat", "Emission Strength"),
)


def _is_blendable(template):
    return "blendable" in template or "blackwall" in template


def _values(data):
    """The skin's per-material values, as the build writes them into the skin group's inputs"""
    bias_min = param_float(data, "DetailRoughnessBiasMin", 0.0)
    bias_max = param_float(data, "DetailRoughnessBiasMax", 1.0)
    return {
        "DetailNormalInfluence": clamp01(
            param_float(data, "DetailNormalInfluence", 1.0)
        ),
        "MicroDetailUVScale01": max(
            0.0, param_float(data, "MicroDetailUVScale01", 1.0)
        ),
        "MicroDetailUVScale02": max(
            0.0, param_float(data, "MicroDetailUVScale02", 1.0)
        ),
        "MicroDetailInfluence": clamp01(param_float(data, "MicroDetailInfluence", 0.0)),
        "DetailRoughnessBiasMin": min(bias_min, bias_max),
        "DetailRoughnessBiasMax": max(bias_min, bias_max),
        "CavityIntensity": param_float(data, "CavityIntensity", 0.0),
        "TintColor": param_color(data, "TintColor", _WHITE),
        "TintScale": param_float(data, "TintScale", 0.0),
        "SecondaryAlbedoTintColorInfluence": param_float(
            data, "SecondaryAlbedoTintColorInfluence", 0.0
        ),
        "SecondaryAlbedoInfluence": param_float(data, "SecondaryAlbedoInfluence", 0.0),
        "BloodColor": param_color(data, "BloodColor", (0.0, 0.0, 0.0, 1.0)),
        "EmissiveEV": max(-16.0, min(16.0, param_float(data, "EmissiveEV", 0.0))),
    }


def _wrinkle_textures_present(data):
    return any(param_texture_path(data, key) for key in _WRINKLE_TEXTURES)


def _set_image(node, image):
    if node.image is not image:
        node.image = image


def _set_interface_defaults(group, specs):
    items = group.interface.items_tree
    for name, default, minimum, maximum in specs:
        item = items.get(name)
        if item is None:
            continue
        item.default_value = default
        if minimum is not None:
            item.min_value = minimum
        if maximum is not None:
            item.max_value = maximum


def _organize_interface(group, panels):
    """Put the group's inputs in panels, (panel name, closed by default, input names) in order; best effort"""
    interface = group.interface
    try:
        for name, closed, sockets in panels:
            panel = interface.new_panel(name=name)
            panel.default_closed = closed
            for index, socket_name in enumerate(sockets):
                item = interface.items_tree.get(socket_name)
                if item is not None and getattr(item, "in_out", "") == "INPUT":
                    interface.move_to_parent(item=item, parent=panel, to_position=index)
    except (AttributeError, RuntimeError, TypeError) as error:
        _LOG.debug("Skin group panels unavailable: %s", error)


def _build_microdetail_uv_group(group):
    nodes = group.nodes
    links = group.links
    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-1500, 0)
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (900, 0)
    _set_interface_defaults(
        group,
        (
            ("TintColorMask", (1.0, 0.0, 0.0, 1.0), None, None),
            ("MicroDetailUVScale01", 1.0, 0.0, 100.0),
            ("MicroDetailUVScale02", 1.0, 0.0, 100.0),
        ),
    )
    # TintColorMask blue is the microdetail UV scale region, in fifths.
    split = separate_color_node(
        nodes, links, group_in.outputs["TintColorMask"], "Tint Mask", (-1300, -200)
    )
    b5 = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        split.outputs["Blue"],
        5.0,
        name="Microdetail Scale ×5",
        location=(-1100, -200),
    )
    b_floor = shader_nodes.math(
        nodes, links, "FLOOR", b5, name="Microdetail Scale floor", location=(-940, -200)
    )
    region_scale = shader_nodes.math(
        nodes,
        links,
        "DIVIDE",
        b_floor,
        5.0,
        name="Microdetail Scale",
        location=(-780, -200),
    )
    md_uv_lerp = smoothstep_socket(
        group, region_scale, 0.0, 1.0, (-600, -200), "Microdetail UV Blend"
    )
    uv = group_in.outputs["UV"]
    uv1 = _atlas_uv(
        group,
        uv,
        group_in.outputs["MicroDetailUVScale01"],
        md_uv_lerp,
        0,
        "Microdetail 01",
        (-300, 100),
    )
    uv2 = _atlas_uv(
        group,
        uv,
        group_in.outputs["MicroDetailUVScale02"],
        md_uv_lerp,
        1,
        "Microdetail 02",
        (-300, -150),
    )
    links.new(uv1, group_out.inputs["UV 01"])
    links.new(uv2, group_out.inputs["UV 02"])


def microdetail_uv_group():
    return get_or_build_node_group(
        MICRODETAIL_UV_GROUP_NAME,
        (
            ("NodeSocketVector", "UV"),
            ("NodeSocketColor", "TintColorMask"),
            ("NodeSocketFloat", "MicroDetailUVScale01"),
            ("NodeSocketFloat", "MicroDetailUVScale02"),
        ),
        (("NodeSocketVector", "UV 01"), ("NodeSocketVector", "UV 02")),
        _build_microdetail_uv_group,
    )


def _build_skin_group(group):
    tree = group
    nodes = group.nodes
    links = group.links
    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-2400, 0)
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (2200, 0)
    _set_interface_defaults(
        group,
        tuple((name, default, None, None) for name, default in _GROUP_TEXTURE_INPUTS)
        + tuple(
            (name, default, minimum, maximum)
            for _socket, name, default, minimum, maximum in _GROUP_VALUE_INPUTS
        ),
    )

    _organize_interface(
        group,
        (
            (
                "Textures",
                False,
                tuple(name for name, _default in _GROUP_TEXTURE_INPUTS)
                + ("SecondaryAlbedo Alpha",),
            ),
            (
                "Parameters",
                False,
                tuple(
                    name
                    for _socket, name, *_rest in _GROUP_VALUE_INPUTS
                    if name != "SecondaryAlbedo Alpha" and not name.startswith("Use ")
                ),
            ),
            (
                "Switches",
                True,
                tuple(
                    name
                    for _socket, name, *_rest in _GROUP_VALUE_INPUTS
                    if name.startswith("Use ")
                ),
            ),
        ),
    )

    def inp(name):
        return group_in.outputs[name]

    use_micro = inp("Use MicroDetail")

    # Roughness texture is R=roughness, G=metalness, B=detail mask.
    rms = separate_color_node(
        nodes, links, inp("Roughness"), "Rough/Metal/Detail", (-1660, 80)
    )
    rough = rms.outputs["Red"]
    detail_mask = rms.outputs["Blue"]
    links.new(rms.outputs["Green"], group_out.inputs["Metallic"])
    # Microdetail only shows where the detail mask allows and the material has a microdetail texture.
    micro_mask = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        detail_mask,
        use_micro,
        name="Microdetail Mask",
        location=(620, -1240),
    )

    # TintColorMask is R=tint mask, G=microdetail region (in sixths).
    tint_split = separate_color_node(
        nodes, links, inp("TintColorMask"), "Tint/Microdetail Mask", (-1660, -180)
    )
    tint_mask = tint_split.outputs["Red"]
    g6 = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        tint_split.outputs["Green"],
        6.0,
        name="Microdetail Regions ×6",
        location=(-1460, -100),
    )
    g_floor = shader_nodes.math(
        nodes,
        links,
        "FLOOR",
        g6,
        name="Microdetail Regions floor",
        location=(-1300, -100),
    )
    region = shader_nodes.math(
        nodes,
        links,
        "DIVIDE",
        g_floor,
        6.0,
        name="Microdetail Regions",
        location=(-1140, -100),
    )
    mdr_scale = smoothstep_socket(
        tree, region, 0.0, 1.0, (-940, -100), "Microdetail Region Blend"
    )

    squash_raw, stretch_raw = build_wrinkle_influence(tree, location=(-2000, -2200))
    squash_scale = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        squash_raw,
        inp("Use Wrinkles"),
        name="Wrinkle Squash",
        location=(-1800, -2110),
    )
    stretch_scale = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        stretch_raw,
        inp("Use Wrinkles"),
        name="Wrinkle Stretch",
        location=(-1800, -2290),
    )

    # Apply normals from broadest deformation to finest surface detail.
    def decode(name, y, label):
        return decode_cp77_normal(
            tree, inp(name), strength=1.0, location=(-1640, y), label=label
        ).outputs["Tangent"]

    # The additive wrinkle blend bends a normal even toward a flat map, so a missing map is switched off.
    def wrinkle_factor(scale, use, label, y):
        return shader_nodes.math(
            nodes, links, "MULTIPLY", scale, inp(use), name=label, location=(-1560, y)
        )

    normal_tangent = decode("Normal", -520, "Skin Normal DirectX Decode")
    normal_tangent = blend_wrinkle_normal(
        tree,
        normal_tangent,
        decode("Detailmap_Stretch", -680, "Stretch Wrinkle DirectX Decode"),
        wrinkle_factor(stretch_scale, "Use Stretch Map", "Stretch Map Weight", -600),
        "Stretch Wrinkle Normal",
        (-1380, -640),
    )
    normal_tangent = blend_wrinkle_normal(
        tree,
        normal_tangent,
        decode("Detailmap_Squash", -840, "Squash Wrinkle DirectX Decode"),
        wrinkle_factor(squash_scale, "Use Squash Map", "Squash Map Weight", -760),
        "Squash Wrinkle Normal",
        (-1380, -800),
    )
    normal_tangent = tangent_normal_blend_socket(
        tree,
        normal_tangent,
        decode("DetailNormal", -1000, "Detail Normal DirectX Decode"),
        inp("DetailNormalInfluence"),
        (-1060, -900),
        "Detail Normal Blend",
    )

    micro1 = decode("MicroDetail 01", -1080, "Microdetail 01 DirectX Decode")
    micro2 = decode("MicroDetail 02", -1320, "Microdetail 02 DirectX Decode")
    micro1_source = _source_xy(
        tree, inp("MicroDetail 01"), "Microdetail 01 Source", (-360, -1080)
    )
    micro2_source = _source_xy(
        tree, inp("MicroDetail 02"), "Microdetail 02 Source", (-360, -1320)
    )
    delta = shader_nodes.vector_math(
        nodes,
        links,
        "SUBTRACT",
        micro1,
        micro2,
        name="Microdetail 01-02",
        location=(80, -1180),
    )
    micro_mix = shader_nodes.vector_math(
        nodes,
        links,
        "MULTIPLY_ADD",
        delta,
        mdr_scale,
        micro2,
        name="Microdetail Blend",
        location=(440, -1180),
    )
    influence = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        micro_mask,
        inp("MicroDetailInfluence"),
        name="Microdetail Normal Influence",
        location=(620, -1040),
    )
    normal_tangent = tangent_normal_blend_socket(
        tree,
        normal_tangent,
        micro_mix,
        influence,
        (800, -1080),
        "Microdetail Normal Blend",
    )

    def detail_roughness(pair, label, y):
        x, yy = pair
        one_plus_x = shader_nodes.math(
            nodes, links, "ADD", x, 1.0, name=f"{label} 1+x", location=(80, y)
        )
        one_plus_y = shader_nodes.math(
            nodes, links, "ADD", yy, 1.0, name=f"{label} 1+y", location=(80, y - 80)
        )
        y_term = shader_nodes.math(
            nodes,
            links,
            "MULTIPLY",
            yy,
            one_plus_y,
            name=f"{label} y(1+y)",
            location=(240, y - 80),
        )
        total = shader_nodes.math(
            nodes,
            links,
            "MULTIPLY_ADD",
            x,
            one_plus_x,
            y_term,
            name=f"{label} Sum",
            location=(400, y - 40),
        )
        total = shader_nodes.math(
            nodes,
            links,
            "MULTIPLY",
            total,
            2.5,
            name=f"{label} ×2.5",
            location=(560, y - 40),
        )
        return shader_nodes.math(
            nodes, links, "ADD", total, 0.2, name=label, location=(720, y - 40)
        )

    dr1 = detail_roughness(micro1_source, "Detail Roughness 01", -1480)
    dr2 = detail_roughness(micro2_source, "Detail Roughness 02", -1680)
    micro_rough_blend = mix_scalar_socket(
        nodes,
        links,
        "Microdetail Roughness Blend",
        (900, -1580),
        dr1,
        dr2,
        mdr_scale,
    )
    biased = mix_scalar_socket(
        nodes,
        links,
        "Detail Roughness Bias",
        (1080, -1580),
        inp("DetailRoughnessBiasMin"),
        inp("DetailRoughnessBiasMax"),
        micro_rough_blend,
    )
    rough_detail = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        rough,
        biased,
        name="Roughness × Detail",
        location=(1260, -1580),
    )
    rough = mix_scalar_socket(
        nodes,
        links,
        "Final Roughness",
        (1440, -1580),
        rough,
        rough_detail,
        micro_mask,
    )
    links.new(rough, group_out.inputs["Roughness"])

    encoded = encode_tangent_normal(
        tree, normal_tangent, location=(1300, -760), label="Skin Normal Encode"
    )
    normal = tangent_to_normal_map(
        tree, encoded, location=(1500, -760), label="Skin Normal"
    )
    links.new(normal, group_out.inputs["Normal"])

    # Cavity modifies albedo before tinting.
    albedo = inp("Albedo")
    detail_split = separate_color_node(
        nodes, links, inp("DetailNormal"), "Detail Source RG", (-1400, -1060)
    )
    detail_x2 = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        detail_split.outputs["Red"],
        2.0,
        name="Detail Source X x2",
        location=(-1220, -1020),
    )
    detail_source_x = shader_nodes.math(
        nodes,
        links,
        "SUBTRACT",
        detail_x2,
        1.0,
        name="Detail Source X",
        location=(-1040, -1020),
    )
    mask_sq = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        detail_mask,
        detail_mask,
        name="Detail Mask²",
        location=(900, -1840),
    )
    cavity = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        micro_rough_blend,
        mask_sq,
        name="Cavity Microdetail",
        location=(1080, -1840),
    )
    cavity = shader_nodes.math(
        nodes,
        links,
        "ADD",
        cavity,
        detail_source_x,
        name="Cavity + DetailNormal.x",
        location=(1260, -1840),
    )
    cavity_scale = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        inp("CavityIntensity"),
        0.1,
        name="CavityIntensity × 0.1",
        location=(1260, -1980),
    )
    cavity = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        cavity,
        cavity_scale,
        name="Cavity Intensity",
        location=(1440, -1840),
    )
    cavity10 = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        cavity,
        10.0,
        name="Cavity ×10",
        location=(1620, -1840),
    )
    cavity_inv = smoothstep_socket(
        tree, cavity10, 0.15, 0.0, (900, -2020), "Cavity Inverted"
    )
    cavity_inv = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        cavity_inv,
        0.01,
        name="Cavity Inverted ×0.01",
        location=(1880, -2020),
    )
    subtract = mix_color_socket(
        nodes,
        links,
        "Albedo - Cavity",
        albedo,
        (-1.0, -1.0, -1.0, 1.0),
        cavity,
        (-1320, 700),
        blend_type="ADD",
    )
    cavity_corrected = mix_color_socket(
        nodes,
        links,
        "Cavity Corrected",
        subtract,
        _WHITE,
        cavity_inv,
        (-1100, 700),
        blend_type="ADD",
    )
    albedo = mix_color_socket(
        nodes,
        links,
        "Cavity Mask",
        albedo,
        cavity_corrected,
        micro_mask,
        (-880, 700),
    )

    # A negative TintScale overlays the tint instead of multiplying it.
    tint_scale = inp("TintScale")
    tint_amount = shader_nodes.math(
        nodes, links, "ABSOLUTE", tint_scale, name="|TintScale|", location=(-800, 560)
    )
    tint_factor = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        tint_amount,
        tint_mask,
        name="Tint Scale × Mask",
        location=(-620, 560),
    )
    tint_multiply = mix_color_socket(
        nodes,
        links,
        "Tint Multiply",
        albedo,
        inp("TintColor"),
        1.0,
        (-620, 900),
        blend_type="MULTIPLY",
        clamp=True,
    )
    tint_overlay = mix_color_socket(
        nodes,
        links,
        "Tint Overlay",
        albedo,
        inp("TintColor"),
        1.0,
        (-620, 760),
        blend_type="OVERLAY",
        clamp=True,
    )
    overlay = shader_nodes.math(
        nodes,
        links,
        "LESS_THAN",
        tint_scale,
        0.0,
        name="TintScale < 0",
        location=(-800, 700),
    )
    tinted_full = mix_color_socket(
        nodes,
        links,
        "Tint Blend Mode",
        tint_multiply,
        tint_overlay,
        overlay,
        (-500, 820),
    )
    tinted = mix_color_socket(
        nodes, links, "Tint Mask", albedo, tinted_full, tint_factor, (-380, 700)
    )

    secondary = inp("SecondaryAlbedo")
    secondary_tint_factor = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        tint_factor,
        inp("SecondaryAlbedoTintColorInfluence"),
        name="Secondary Tint Mask",
        location=(-380, 1020),
    )
    secondary_mul = mix_color_socket(
        nodes,
        links,
        "Secondary × Tint",
        secondary,
        tinted,
        1.0,
        (-160, 1040),
        blend_type="MULTIPLY",
    )
    secondary_tinted = mix_color_socket(
        nodes,
        links,
        "Secondary Tint Influence",
        secondary,
        secondary_mul,
        secondary_tint_factor,
        (60, 1020),
    )
    sec_factor = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        inp("SecondaryAlbedo Alpha"),
        inp("SecondaryAlbedoInfluence"),
        name="Secondary Influence × Alpha",
        location=(280, 980),
    )

    blood_mask = separate_color_node(
        nodes, links, inp("Bloodflow"), "Bloodflow Mask", (-380, 1240)
    )
    blood_target = mix_color_socket(
        nodes,
        links,
        "Blood Influenced Tint",
        tinted,
        inp("BloodColor"),
        1.0,
        (-120, 1320),
        blend_type="ADD",
        clamp=True,
    )

    def cube(value, label, y):
        squared = shader_nodes.math(
            nodes,
            links,
            "MULTIPLY",
            value,
            value,
            name=f"{label}²",
            location=(-120, y),
        )
        return shader_nodes.math(
            nodes,
            links,
            "MULTIPLY",
            squared,
            value,
            name=f"{label}³",
            location=(60, y),
        )

    squash_blood = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        blood_mask.outputs["Red"],
        cube(squash_scale, "Bloodflow Squash", 1200),
        name="Bloodflow Squash Mask",
        location=(240, 1200),
    )
    stretch_blood = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        blood_mask.outputs["Green"],
        cube(stretch_scale, "Bloodflow Stretch", 1120),
        name="Bloodflow Stretch Mask",
        location=(240, 1120),
    )
    color = mix_color_socket(
        nodes,
        links,
        "Squash Bloodflow",
        tinted,
        blood_target,
        squash_blood,
        (460, 1280),
        clamp=True,
    )
    color = mix_color_socket(
        nodes,
        links,
        "Stretch Bloodflow",
        color,
        blood_target,
        stretch_blood,
        (680, 1280),
        clamp=True,
    )
    # Without bloodflow the tinted colour passes on unclamped.
    color = mix_color_socket(
        nodes,
        links,
        "Bloodflow",
        tinted,
        color,
        inp("Use Bloodflow"),
        (800, 1180),
    )
    color = mix_color_socket(
        nodes,
        links,
        "Final Skin Color",
        color,
        secondary_tinted,
        sec_factor,
        (920, 1020),
    )
    links.new(color, group_out.inputs["Base Color"])

    # Emission strength is max(0, 2^EmissiveEV - 1), masked by EmissiveMask red; 0 EV emits nothing.
    ev_linear = shader_nodes.math(
        nodes, links, "POWER", 2.0, inp("EmissiveEV"), name="2^EV", location=(880, 440)
    )
    ev_linear = shader_nodes.math(
        nodes, links, "SUBTRACT", ev_linear, 1.0, name="2^EV - 1", location=(1040, 440)
    )
    ev_linear = shader_nodes.math(
        nodes,
        links,
        "MAXIMUM",
        ev_linear,
        0.0,
        name="Emissive Linear",
        location=(1200, 440),
    )
    mask_split = separate_color_node(
        nodes, links, inp("EmissiveMask"), "Emissive Mask", (1100, 580)
    )
    emission = shader_nodes.math(
        nodes,
        links,
        "MULTIPLY",
        mask_split.outputs["Red"],
        ev_linear,
        name="Skin EmissiveEV",
        location=(1300, 580),
    )
    links.new(emission, group_out.inputs["Emission Strength"])


def skin_group():
    return get_or_build_node_group(
        SKIN_GROUP_NAME,
        tuple(("NodeSocketColor", name) for name, _default in _GROUP_TEXTURE_INPUTS)
        + tuple((socket, name) for socket, name, _d, _lo, _hi in _GROUP_VALUE_INPUTS),
        _GROUP_OUTPUTS,
        _build_skin_group,
    )


def _write_inputs(node, values):
    if node is None:
        return
    for name, value in values.items():
        socket = node.inputs.get(name)
        if socket is not None:
            set_default_if_changed(socket, value)


class Skin(MaterialTypeBase):
    """Cyberpunk skin: one shared "CP77 Skin" node group, its textures and its Principled BSDF

    Every skin material uses the same group. The material holds the textures (each Image Texture node is named
    after its parameter and feeds the group input of that name), the group's per-material values, and the
    Principled BSDF the group drives. A missing texture leaves its group input on a default that switches that
    part of the shader off.
    """

    def _texture_keys(self, data, blendable):
        """The texture parameters a build loads"""
        for key, non_color in _TEXTURES:
            if blendable and key in {"SecondaryAlbedo", "EmissiveMask"}:
                continue
            if key == "EmissiveMask" and param_float(data, "EmissiveEV", 0.0) == 0.0:
                continue
            yield key, non_color

    def _images(self, data, blendable):
        images = {
            key: self._load_relative_image(
                param_texture_path(data, key), non_color=non_color, error_label=key
            )
            for key, non_color in self._texture_keys(data, blendable)
        }
        micro_path = param_texture_path(data, "MicroDetail")
        images["MicroDetail"] = (
            self._load_relative_image(
                micro_path, non_color=True, error_label="MicroDetail"
            )
            if micro_path
            else None
        )
        return images

    def structure_key(self, Data):
        # Which textures load shape the material; every other parameter is a value on the skin group's inputs.
        data = Data if isinstance(Data, dict) else {}
        images = self._images(data, _is_blendable(self.material_template))
        return (
            "skin-group-v1",
            tuple((key, image is not None) for key, image in images.items()),
        )

    def _write_values(self, data, Mat, images):
        nodes = Mat.node_tree.nodes
        values = _values(data)
        skin_values = {
            name: value
            for name, value in values.items()
            if not name.startswith("MicroDetailUVScale")
        }
        skin_values["Use MicroDetail"] = 1.0 if images.get("MicroDetail") else 0.0
        wrinkles = _wrinkle_textures_present(data)
        skin_values["Use Wrinkles"] = 1.0 if wrinkles else 0.0
        for flag, key in (
            ("Use Stretch Map", "Detailmap_Stretch"),
            ("Use Squash Map", "Detailmap_Squash"),
        ):
            skin_values[flag] = 1.0 if images.get(key) is not None else 0.0
        skin_values["Use Bloodflow"] = (
            1.0 if wrinkles and images.get("Bloodflow") is not None else 0.0
        )
        _write_inputs(nodes.get(_SKIN_NODE), skin_values)
        _write_inputs(
            nodes.get(_MICRODETAIL_UV_NODE),
            {
                "MicroDetailUVScale01": values["MicroDetailUVScale01"],
                "MicroDetailUVScale02": values["MicroDetailUVScale02"],
            },
        )
        if wrinkles:
            Mat["cp77_skin_wrinkle_region_count"] = WRINKLE_REGION_COUNT
            Mat["cp77_skin_wrinkle_region_set"] = DEFAULT_FACE_REGIONSET_PATH
        assign_if_changed(Mat, "diffuse_color", (*values["TintColor"][:3], 1.0))

    def apply_structure_values(self, Data, Mat, slots):
        data = Data if isinstance(Data, dict) else {}
        nodes = Mat.node_tree.nodes
        images = self._images(data, _is_blendable(self.material_template))
        for key, image in images.items():
            if image is None:
                continue
            for name in _MICRODETAIL_NODES if key == "MicroDetail" else (key,):
                node = nodes.get(name)
                if node is not None:
                    _set_image(node, image)
        self._write_values(data, Mat, images)

    def create(self, Data, Mat):
        data = Data if isinstance(Data, dict) else {}
        template = backslash_depot_key(Mat.get("MaterialTemplate", ""))
        self.material_template = template
        is_blendable = _is_blendable(template)
        self.structure_slots = {}
        Mat.use_nodes = True
        tree = Mat.node_tree
        nodes = tree.nodes
        links = tree.links
        nodes.clear()

        texcoord = new_labeled_node(nodes, "ShaderNodeTexCoord", "UV", (-1500, 200))
        uv = texcoord.outputs["UV"]
        skin = add_group_node(tree, skin_group(), (-480, 300), label="CP77 Skin")
        skin.name = _SKIN_NODE
        skin.width = 320.0
        skin.hide = False
        # The group already gathers the surface inputs; bundling its links would only add nodes between it and the
        # Principled BSDF.
        skin["cp77_surface_bundle_boundary"] = True
        principled = new_labeled_node(
            nodes, "ShaderNodeBsdfPrincipled", "Principled BSDF", (0, 300)
        )
        output = new_labeled_node(
            nodes, "ShaderNodeOutputMaterial", "Material Output", (340, 300)
        )
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        for name in ("Base Color", "Metallic", "Roughness", "Normal"):
            links.new(skin.outputs[name], find_input(principled, name))
        emission = find_input(principled, "Emission Color", "Emission")
        if emission is not None:
            links.new(skin.outputs["Base Color"], emission)
        strength = find_input(principled, "Emission Strength")
        if strength is not None:
            links.new(skin.outputs["Emission Strength"], strength)
        subsurface = find_input(principled, "Subsurface Weight")
        if subsurface is not None:
            subsurface.default_value = 1.0
        scale = find_input(principled, "Subsurface Scale")
        if scale is not None:
            scale.default_value = 0.005

        # One texture column left of the group, in the order of its inputs.
        x, y = -1080, 900
        textures = {}
        for key, non_color in self._texture_keys(data, is_blendable):
            node = create_relative_texture_node(
                self, tree, data, key, uv, (x, y), non_color=non_color
            )
            if node is None:
                continue
            textures[key] = node
            links.new(node.outputs["Color"], skin.inputs[key])
            if key == "SecondaryAlbedo":
                links.new(node.outputs["Alpha"], skin.inputs["SecondaryAlbedo Alpha"])
            y -= 300

        micro_path = param_texture_path(data, "MicroDetail")
        micro_image = (
            self._load_relative_image(
                micro_path, non_color=True, error_label="MicroDetail"
            )
            if micro_path
            else None
        )
        if micro_image is not None:
            micro_uv = add_group_node(
                tree, microdetail_uv_group(), (-1380, y - 100), label="MicroDetail UV"
            )
            micro_uv.name = _MICRODETAIL_UV_NODE
            micro_uv.width = 220.0
            micro_uv.hide = False
            links.new(uv, micro_uv.inputs["UV"])
            tint_mask = textures.get("TintColorMask")
            if tint_mask is not None:
                links.new(tint_mask.outputs["Color"], micro_uv.inputs["TintColorMask"])
            for index, name in enumerate(_MICRODETAIL_NODES):
                tex = new_labeled_node(nodes, "ShaderNodeTexImage", name, (x, y))
                tex.image = micro_image
                tex.extension = "REPEAT"
                links.new(micro_uv.outputs[f"UV 0{index + 1}"], tex.inputs["Vector"])
                links.new(tex.outputs["Color"], skin.inputs[name])
                y -= 300

        images = {key: node.image for key, node in textures.items()}
        images["MicroDetail"] = micro_image
        self._write_values(data, Mat, images)
