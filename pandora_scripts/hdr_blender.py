"""
Script to render IBL renders
Usage: blender <blend file> -P hdr_blender.py -b
"""
import os
import random
from math import radians
from os import listdir
from os.path import isfile, join

import bpy
import cycles
from mathutils import Euler, Matrix, Vector


import sys
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"

inp_dirs = [argv[0]] 
out_dirs = [argv[1]] 

pairs = list(zip(inp_dirs,out_dirs))
scn = bpy.context.scene
scn.render.engine = 'CYCLES'
scn.render.image_settings.file_format = 'OPEN_EXR'
scn.world.use_nodes = True
# Enabf_le GPU rendering.
# bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if "TITAN RX" == device.name:
        device.use = True
    else:
        device.use = False
# select world node tree
wd = scn.world
nt = bpy.data.worlds[wd.name].node_tree

for pair in pairs:
    inp, out = pair    
    inp = inp    
    os.makedirs(out,exist_ok=True)
    S = bpy.context.scene
    envFiles = [f for f in listdir(inp) if isfile(join(inp, f)) and f.endswith('.exr')]
    i = 0
    for env_map in envFiles:
        env_file = join(inp, env_map)
        bpy.ops.image.open(filepath=env_file, directory=inp, files=[
                           {"name": env_map, "name": env_map}], show_multiview=False)
        output_name = env_map.split('.')[0] +'.exr'
        
        backNode = nt.nodes['Environment Texture']
        backNode.image = bpy.data.images.load(env_file)

        bpy.context.scene.render.filepath = os.path.join(out, output_name)
        print("Output",os.path.join(out, output_name))
        bpy.ops.render.render(write_still=True)
