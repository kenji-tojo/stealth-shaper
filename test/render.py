## This code was used to produce the ground truth images under ./data/gt/


import os
from math import *

import numpy as np
from PIL import Image 

from scene import *


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--spp', type=int, default=32, help='number of rays per pixel')
    parser.add_argument('--overwrite', action='store_true', help='overwriting existing image')
    args = parser.parse_args()

    spp: int = args.spp
    overwrite: bool = args.overwrite

    SCENE_DIR = './data/'

    scene, renderer, normal_map = load_scene(path=os.path.join(SCENE_DIR, 'scene.json'), initialize_normal_map=False)

    n_poses = len(list(filter(lambda x: os.path.splitext(x)[1] == '.json', os.listdir(os.path.join(SCENE_DIR, 'gt')))))
    print(f'rendering reference images for {n_poses} camera poses')

    GT_DIR = os.path.join(SCENE_DIR, 'gt')

    for pose_id in range(n_poses):
        camera, img = load_pose(dir=os.path.join(SCENE_DIR, 'gt'), id=pose_id)
        save_exr: bool = overwrite or img is None

        renderer.channels = 1
        renderer.spp = spp

        img = renderer.render(scene, camera)
        img = img.reshape(img.shape[0], img.shape[1])
        print(f'vmin = {np.min(img)}, vmax = {np.max(img)}')

        if save_exr:
            print(f'saving exr image for pose no.{pose_id}')
            import imageio
            imageio.plugins.freeimage.download()
            imageio.imwrite(os.path.join(GT_DIR, f'image_{pose_id}.exr'), img)

        Image.fromarray(np.round(img*255.).clip(0.,255.).astype(np.uint8)).save(os.path.join(GT_DIR, f'image_{pose_id}.png'))