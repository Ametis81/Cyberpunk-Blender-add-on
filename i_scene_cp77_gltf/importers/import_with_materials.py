import bpy
import os
import json
import time
from io_scene_gltf2.io.imp.gltf2_io_gltf import glTFImporter
from io_scene_gltf2.blender.imp.gltf2_blender_gltf import BlenderGlTF
from ..main.setup import MaterialBuilder
from ..main.bartmoss_functions import UV_by_bounds
from .import_from_external import *
from .attribute_import import manage_garment_support
from ..cyber_props import add_anim_props
from ..jsontool import jsonload
import traceback

def get_anim_info(animations):
    # Get animations
    #animations = gltf_importer.data.animations
    cp77_addon_prefs = bpy.context.preferences.addons['i_scene_cp77_gltf'].preferences
    for animation in animations:
        if not cp77_addon_prefs.non_verbose:
            print(f"Processing animation: {animation.name}")

        # Find an action whose name contains the animation name
        action = next((act for act in bpy.data.actions if act.name.startswith(animation.name + "_Armature")), None)

        if action:
            add_anim_props(animation, action)
            if not cp77_addon_prefs.non_verbose:
                print("Properties added to", action.name)
        else:
            if not cp77_addon_prefs.non_verbose:
                print("No action found for", animation.name)

    print('')
def objs_in_col(top_coll, objtype):
    return sum([len([o for o in col.objects if o.type==objtype]) for col in top_coll.children_recursive])+len([o for o in top_coll.objects if o.type==objtype])

existingMaterials = None
imported = None
appearances = None
collection = None

def CP77GLBimport(self, with_materials, remap_depot, exclude_unused_mats=True, image_format='png', filepath='', hide_armatures=True, import_garmentsupport=False, files=[], directory='', appearances=[]):
    cp77_addon_prefs = bpy.context.preferences.addons['i_scene_cp77_gltf'].preferences
    context=bpy.context
   # obj = None
    start_time = time.time()
    loadfiles=self.files
    DepotPath=cp77_addon_prefs
    appearances=self.appearances.split(",")
    if not cp77_addon_prefs.non_verbose:
        if ".anims.glb" in self.filepath:
            print('-------------------- Beginning Cyberpunk Animation Import --------------------')
            print('')
            print(f"Importing Animations From: {os.path.basename(self.filepath)}")
            print('')
        else:
            print('-------------------- Beginning Cyberpunk Model Import --------------------')
            print('')
            if with_materials==True:
                print(f"Importing: {os.path.basename(self.filepath)} with materials")
                print(f"Appearances to Import: {(', '.join(appearances))}")
            else:
                print(f"Importing: {os.path.basename(self.filepath)}")
            print('')
    # prevent crash if no directory supplied when using filepath
    if len(self.directory)>0:
        directory = self.directory
    else:
        directory = os.path.dirname(self.filepath)

    #if no files were supplied and a filepath is populate the files from the filepath
    if len(loadfiles)==0 and len(self.filepath)>0:
        f={}
        f['name']=os.path.basename(self.filepath)
        loadfiles=(f,)

    file_names=[]
    file_paths=[]
    for f in loadfiles:
        file_names.append( f['name'])
        file_paths.append(os.path.join(directory, f['name']))

    # check materials

    #Kwek: Gate this--do the block iff corresponding Material.json exist
    #Kwek: was tempted to do a try-catch, but that is just La-Z
    #Kwek: Added another gate for materials
    for f in loadfiles:
        filename=os.path.splitext(f['name'])[0]
        filepath = os.path.join(directory, f['name'])
        vers = bpy.app.version
        if vers[0] == 4 and vers[1] >= 2:
            gltf_importer = glTFImporter(filepath, { "files": None, "loglevel": 0, "import_pack_images" :True, "merge_vertices" :False, "import_shading" : 'NORMALS', "bone_heuristic":'BLENDER', "guess_original_bind_pose" : False, "import_user_extensions": "",'disable_bone_shape':False, 'bone_shape_scale_factor':1.0})
        else:
            gltf_importer = glTFImporter(filepath, { "files": None, "loglevel": 0, "import_pack_images" :True, "merge_vertices" :False, "import_shading" : 'NORMALS', "bone_heuristic":'BLENDER', "guess_original_bind_pose" : False, "import_user_extensions": "",'disable_bone_shape':False,})
        gltf_importer.read()
        gltf_importer.checks()
        existingMeshes = bpy.data.meshes.keys()

        current_file_base_path = os.path.splitext(filepath)[0]
        has_material_json = os.path.exists(current_file_base_path + ".Material.json")

        existingMaterials = bpy.data.materials.keys()
        BlenderGlTF.create(gltf_importer)

        imported=context.selected_objects #the new stuff should be selected

        # if we're not importing a Cyberpunk mesh, not all submesh names will start with submesh_00, and they will be nested weirdly.
        # we want to clean this up.
        imported_meshes = [obj for obj in imported if obj.type == "MESH"]
        imported_empties = [obj for obj in imported if obj.type == "EMPTY"]
        isExternalImport = len(imported_empties) > 0 or len([mesh.name for mesh in imported_meshes if mesh.name.startswith("submesh")]) != len(imported_meshes)

        if isExternalImport:
            CP77_cleanup_external_export(imported)


        imported= context.selected_objects #the new stuff should be selected
        if f['name'][:7]=='terrain':
            UV_by_bounds(imported)

        #create a collection by file name
        collection = bpy.data.collections.new(filename)
        bpy.context.scene.collection.children.link(collection)
        for o in imported:
            import_meshes_and_anims(collection, gltf_importer, hide_armatures, o)

        collection['orig_filepath']=filepath
        collection['numMeshChildren']=objs_in_col(collection, 'MESH')
        collection['numArmatureChildren']=objs_in_col(collection, 'ARMATURE')

        #for sketchfab exports, we want to keep our materials
        if not isExternalImport:
            for name in bpy.data.materials.keys():
                if name not in existingMaterials:
                    bpy.data.materials.remove(bpy.data.materials[name], do_unlink=True, do_id_user=True, do_ui_user=True)

        #Kwek: Gate this--do the block if corresponding Material.json exist
        #Kwek: was tempted to do a try-catch, but that is just La-Z
        #Kwek: Added another gate for materials
        DepotPath=None
        blender_4_scale_armature_bones()
        if ".anims.glb" in filepath:
            break
        else:
            if with_materials==True and has_material_json:
                matjsonpath = current_file_base_path + ".Material.json"
                DepotPath, json_apps, mats = jsonload(matjsonpath)
        if DepotPath == None:
            print('DepotPath not set')
            break

        #DepotPath = str(obj["MaterialRepo"])  + "\\"
        context=bpy.context
        if remap_depot and os.path.exists(context.preferences.addons[__name__.split('.')[0]].preferences.depotfolder_path):
            DepotPath = context.preferences.addons[__name__.split('.')[0]].preferences.depotfolder_path
            if not cp77_addon_prefs.non_verbose:
                print(f"Using depot path: {DepotPath}")
        DepotPath= DepotPath.replace('\\', os.sep)
        #json_apps=obj['Appearances']
        # fix the app names as for some reason they have their index added on the end.
        appkeys=[k for k in json_apps.keys()]
        for i,k in enumerate(appkeys):
            json_apps[k[:-1*len(str(i))]]=json_apps.pop(k)
        validmats={}
        #appearances = ({'name':'short_hair'},{'name':'02_ca_limestone'},{'name':'ml_plastic_doll'},{'name':'03_ca_senna'})
        #if appearances defined populate valid mats with the mats for them, otherwise populate with everything used.
        if len(appearances)>0 and 'ALL' not in appearances:
            if 'Default' in appearances:
                first_key = next(iter(json_apps))
                for m in json_apps[first_key]:
                    validmats[m] = True
            else:
                for key in json_apps.keys():
                    if key in appearances:
                        for m in json_apps[key]:
                            validmats[m]=True
        # there isnt always a default, so if none were listed, or ALL was used, or an invalid one add everything.
        if len(validmats)==0:
            for key in json_apps.keys():
                for m in json_apps[key]:
                    validmats[m]=True
        if with_materials:
            import_mats(current_file_base_path, DepotPath, exclude_unused_mats, existingMeshes, gltf_importer, image_format, mats, validmats)

        if import_garmentsupport:
            manage_garment_support(existingMeshes, gltf_importer)


    if not cp77_addon_prefs.non_verbose:
        print(f"GLB Import Time: {(time.time() - start_time)} Seconds")
        print('')
        print('-------------------- Finished importing Cyberpunk 2077 Model --------------------')

def import_mats(BasePath, DepotPath, exclude_unused_mats, existingMeshes, gltf_importer, image_format, mats, validmats):
    failedon = []
    cp77_addon_prefs = bpy.context.preferences.addons['i_scene_cp77_gltf'].preferences
    start_time = time.time()
    for mat in validmats.keys():
        for m in mats: #obj['Materials']:
            if m['Name'] != mat:
                continue
            if 'BaseMaterial' in m.keys():
                if 'GlobalNormal' in m['Data'].keys():
                    GlobalNormal = m['Data']['GlobalNormal']
                else:
                    GlobalNormal = 'None'
                if 'MultilayerMask' in m['Data'].keys():
                    MultilayerMask = m['Data']['MultilayerMask']
                else:
                    MultilayerMask = 'None'
                if 'DiffuseMap' in m['Data'].keys():
                    DiffuseMap = m['Data']['DiffuseMap']
                else:
                    DiffuseMap = 'None'

                validmats[mat] = {'Name': m['Name'], 'BaseMaterial': m['BaseMaterial'],
                                  'GlobalNormal': GlobalNormal, 'MultilayerMask': MultilayerMask,
                                  'DiffuseMap': DiffuseMap}
            else:
                print(m.keys())

    MatImportList = [k for k in validmats.keys()]
    Builder = MaterialBuilder(mats, DepotPath, str(image_format), BasePath)
    counter = 0
    bpy_mats = bpy.data.materials
    for name in bpy.data.meshes.keys():
        if name in existingMeshes or "Icosphere" in name:
            continue
        bpy.data.meshes[name].materials.clear()
        extras = gltf_importer.data.meshes[counter].extras

        # morphtargets don't have material names. Just use all of them.
        materialNames = None
        if extras is None or ("materialNames" not in extras.keys() or extras["materialNames"] is None):
            if BasePath.endswith(".morphtarget"):
                materialNames = validmats.keys()
            else:
                counter = counter + 1
                continue
        else:
            materialNames = extras["materialNames"]

        # Kwek: I also found that other material hiccups will cause the Collection to fail
        for matname in materialNames:

            if matname not in validmats.keys():
                # print("Pass")
                # print(matname, validmats.keys())
                pass
            else:
                # print('matname: ',matname, validmats[matname])
                m = validmats[matname]
                # Should create a list of mis that dont play nice with this and just check if the mat is using one.

                if matname in bpy_mats.keys() and 'glass' not in matname and 'MaterialTemplate' not in matname and 'Window' not in matname and matname[
                                                                                                                                                :5] != 'Atlas' and 'BaseMaterial' in \
                        bpy_mats[matname].keys() and bpy_mats[matname]['BaseMaterial'] == m['BaseMaterial'] and \
                        bpy_mats[matname]['GlobalNormal'] == m['GlobalNormal'] and bpy_mats[matname][
                    'MultilayerMask'] == m['MultilayerMask']:

                    bpy.data.meshes[name].materials.append(bpy_mats[matname])
                elif matname in bpy_mats.keys() and matname[:5] == 'Atlas' and bpy_mats[matname][
                    'BaseMaterial'] == m['BaseMaterial'] and bpy_mats[matname]['DiffuseMap'] == m['DiffuseMap']:
                    bpy.data.meshes[name].materials.append(bpy_mats[matname])
                else:
                    if matname in validmats.keys():
                        index = 0
                        for rawmat in mats:
                            if rawmat["Name"] == matname:
                                try:
                                    bpymat = Builder.create(mats, index)
                                    if bpymat:
                                        bpymat['BaseMaterial'] = validmats[matname]['BaseMaterial']
                                        bpymat['GlobalNormal'] = validmats[matname]['GlobalNormal']
                                        bpymat['MultilayerMask'] = validmats[matname]['MultilayerMask']
                                        bpymat['DiffuseMap'] = validmats[matname]['DiffuseMap']
                                        bpy.data.meshes[name].materials.append(bpymat)
                                        if 'no_shadows' in bpymat.keys() and bpymat['no_shadows']:
                                            bpy.data.objects[name].visible_shadow = False
                                except:
                                    # Kwek -- finally, even if the Builder couldn't find the materials, keep calm and carry on
                                    print(traceback.print_exc())
                                    failedon.append(matname)
                                    pass
                            index = index + 1

        counter = counter + 1
    if not cp77_addon_prefs.non_verbose:
        if len(failedon) == 0:
            print(f"Shader Setup Completed Succesfully in {(time.time() - start_time)} Seconds")
        else:
            print(f"Material Setup Failed on: {', '.join(failedon)}")
            print(f"Attempted Setup for {(time.time() - start_time)} seconds")

    if exclude_unused_mats:
        return

    index = 0
    for rawmat in mats:#obj["Materials"]:
        if rawmat["Name"] not in bpy.data.materials.keys() and (
                (rawmat["Name"] in MatImportList) or len(MatImportList) < 1):
            Builder.create(mats,index)
        index = index + 1

def blender_4_scale_armature_bones():
    vers = bpy.app.version
    if vers[0] >= 4:
        arms = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
        for arm in arms:
            for pb in arm.pose.bones:
                pb.custom_shape_scale_xyz[0] = .0175
                pb.custom_shape_scale_xyz[1] = .0175
                pb.custom_shape_scale_xyz[2] = .0175
                pb.use_custom_shape_bone_size = True


def import_meshes_and_anims(collection, gltf_importer, hide_armatures, o):
    # check if this is a Cyberpunk import or something else entirely


    for parent in o.users_collection:
        parent.objects.unlink(o)
    collection.objects.link(o)
    # if animations exist, don't hide the armature and get the extras properties
    # We should probably break the base import out into a separate function, have it check the gltf file and then send the info either to anim import or import with materials, but this works too
    animations = gltf_importer.data.animations
    meshes = gltf_importer.data.meshes
    if animations:
        get_anim_info(animations)
        bpy.context.scene.render.fps = 30
        if meshes and "Icosphere" not in mesh.name:
            if 'Armature' in o.name:
                o.hide_set(hide_armatures)
        else:
            if 'Armature' in o.name:
                pass

    else:
        if 'Armature' in o.name:
            o.hide_set(hide_armatures)