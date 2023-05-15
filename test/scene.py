import numpy as np
import os
from math import *
import json

from normal import *
import stealth as st


def load_scene(path: str, initialize_normal_map=False):
    assert os.path.exists(path)
    assert os.path.splitext(path)[1] == '.json'

    with open(path, 'r') as f:
        scene_json = json.loads(f.read().replace('\n',''))
    
    bsdfs = dict()
    for data in scene_json['bsdfs']:
        assert data['type'] == 'phong'
        bsdf = st.PhongBSDF()
        bsdf.albedo = data['albedo']
        bsdf.kd = data['kd']
        bsdf.n = data['n']
        bsdfs[data['name']] = bsdf
    
    scene = st.Scene()
    normal_map = dict()
    for data in scene_json['objects']:
        if data['type'] == 'plane':
            obj = st.Plane()
            scl = data['scale']
            pos = data['position']
            nrm = data['normal']
            obj.set_scale(scl[0],scl[1])
            obj.set_center(pos[0],pos[1],pos[2])
            obj.set_normal(nrm[0],nrm[1],nrm[2])
            if 'normal_map' in data:
                nmap = data['normal_map']
                if initialize_normal_map:
                    normals = create_normal_map(res=nmap['res'], type='none') + 3.0*create_normal_map(res=nmap['res'], type=nmap['type'])
                    normals /= np.linalg.norm(normals, axis=2)[:,:,None]
                    obj.set_normal_map(normals)
                    normal_map['tensor'] = obj.normal_map_tensor()
                    normal_map['plane'] = obj
                    normal_map['res'] = nmap['res']
                else:
                    obj.set_normal_map(create_normal_map(res=nmap['res'],type=nmap['type']))
        elif data['type'] == 'mesh':
            print('error: mesh loading has not been implemented yet')
            assert False ## TODO
        else:
            print('error: invalid object type =', data['type'])
            assert False
        
        scene.add_object(obj, bsdfs[data['bsdf']])
    
    renderer = st.Renderer()
    data = scene_json['light']
    wi = data['wi']
    Li = data['Li']
    renderer.set_light_wi(wi[0],wi[1],wi[2])
    renderer.light_Li = Li
    
    data = scene_json['options']
    renderer.depth = data['depth']
    renderer.resolution = data['res']
    
    return scene, renderer, normal_map


def load_pose(dir: str, id: int = 1):
    path = os.path.join(dir, f'camera_{id}.json')
    assert os.path.exists(path)

    with open(path, 'r') as f:
        data = json.loads(f.read().replace('\n',''))

    camera = st.Camera()
    pos = data['position']
    cen = data['look_at']
    camera.set_position(pos[0],pos[1],pos[2])
    camera.look_at(cen[0],cen[1],cen[2])
    camera.fov = data['fov']

    import imageio
    imageio.plugins.freeimage.download()
    img = None
    img_path = os.path.join(dir, f'image_{id}.exr')
    if os.path.exists(img_path):
        img = np.array(imageio.imread(img_path), dtype=np.float32)

    return camera, img


if __name__ == '__main__':
    scene, renderer, nmap = load_scene('./data/scene.json')
    print(scene, renderer, nmap)

    camera, img_gt = load_pose(dir='./data/gt', id=1)
    print(camera, img_gt)