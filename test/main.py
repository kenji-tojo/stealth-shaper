import os
from math import *

import numpy as np
from PIL import Image

from normal import *
from scene import *
import stealth as st


def save_image(img: np.ndarray, path: str) -> None:
    assert img.ndim == 2 or img.ndim == 3
    assert img.dtype == np.uint8 or img.dtype == np.float32 or img.dtype == np.float64
    assert img.dtype != np.uint8
    Image.fromarray(np.round(img*255.).clip(0.,255.).astype(np.uint8)).save(path)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--spp', type=int, default=128, help='number of rays per pixel')
    parser.add_argument('-i', '--n_iter', type=int, default=5, help='number of iterations')
    parser.add_argument('--alpha', type=float, default=1e-2, help='step size for adam optimizer')
    parser.add_argument('--DEBUG_pose_id', type=int, default=-1, help='pose id in the training (-1 is to use all camera poses)')
    parser.add_argument('--DEBUG_depth', type=int, default=-1, help='depth of path tracer (for debugging)')
    args = parser.parse_args()

    spp: int = args.spp
    alpha: float = args.alpha
    n_iter: int = args.n_iter
    target_pose_id: int = args.DEBUG_pose_id

    os.makedirs('./result/tmp', exist_ok=True)

    SCENE_DIR = './data/'
    GT_DIR = os.path.join(SCENE_DIR, 'gt')

    scene, renderer, normal_map = load_scene(path=os.path.join(SCENE_DIR, 'scene.json'), initialize_normal_map=True)
    assert normal_map is not None

    if args.DEBUG_depth >= 0: renderer.depth = args.DEBUG_depth

    adam = st.Adam()
    adam.alpha = alpha
    adam.add_parameters(normal_map['tensor'])
    
    renderer.channels = 1
    renderer.spp = spp

    n_poses = len(list(filter(lambda x: os.path.splitext(x)[1] == '.json', os.listdir(GT_DIR))))
    print(f'optimizing normal map using {n_poses} camera poses')
    poses = []
    for pose_id in range(n_poses):
        camera, img_gt = load_pose(dir=GT_DIR, id=pose_id)
        assert img_gt is not None
        poses.append((camera, img_gt))

        img = renderer.render(scene, camera)
        img = img.reshape(img.shape[0], img.shape[1])
        save_image(img, f'./result/original_{pose_id}.png')

    pose_indices = np.arange(len(poses), dtype=np.int32)

    rmse_record = [[] for _ in range(len(poses))]

    for iter in range(n_iter):
        print(f'--- iter #{iter+1}/{n_iter} ---')

        np.random.shuffle(pose_indices)

        for ii, pose_id in enumerate(pose_indices):
            if target_pose_id >= 0 and target_pose_id != pose_id:
                continue

            print(f'--- iter #{iter+1}/{n_iter}; pose #{ii+1}/{len(pose_indices)} (id: {pose_id}) ---')

            camera, img_gt = poses[pose_id]

            img = renderer.render(scene, camera)
            img = img.reshape(img.shape[0], img.shape[1])

            print(f'reference: vmin = {np.min(img_gt)}, vmax = {np.max(img_gt)}')
            print(f'rendering: vmin = {np.min(img)}, vmax = {np.max(img)}')

            diff = (img-img_gt)*255.
            rmse = sqrt(np.mean(np.square(diff)))
            print('rmse =', rmse)
            rmse_record[pose_id].append(rmse)

            Ae = diff / diff.size
            renderer.adjoint(Ae, scene, camera, normal_map['plane'], normal_map['tensor'])
            adam.step()

            ## visualizing intermediate results
            save_image(img, f'./result/tmp/rendering.png')
            import matplotlib.pyplot as plt
            plt.clf()
            plt.imshow(np.abs(diff), vmin=0, vmax=10)
            plt.savefig('./result/tmp/diff.png')

        normals = normal_map['tensor'].numpy().flatten()
        nrm_res = normal_map['res']
        nrm_count = nrm_res ** 2
        normals = np.concatenate((
            normals[nrm_count*0:nrm_count*1][:,None],
            normals[nrm_count*1:nrm_count*2][:,None],
            normals[nrm_count*2:nrm_count*3][:,None]
        ), axis=1)
        save_image(shade_normals(normals.reshape(nrm_res, nrm_res, 3)), './result/tmp/normal.png')


    ## visualizing the final result
    for pose_id in range(len(poses)):
        camera, img_gt = poses[pose_id]
        img = renderer.render(scene, camera)
        img = img.reshape(img.shape[0], img.shape[1])
        print(f'vmin = {np.min(img)}, vmax = {np.max(img)}')
        rmse = sqrt(np.mean(np.square((img-img_gt)*255.)))
        print('rmse =', rmse)
        rmse_record[pose_id].append(rmse)
        save_image(img, f'./result/optimized_{pose_id}.png')

    normals = normal_map['tensor'].numpy().flatten()
    nrm_res = normal_map['res']
    nrm_count = nrm_res ** 2
    normals = np.concatenate((
        normals[nrm_count*0:nrm_count*1][:,None],
        normals[nrm_count*1:nrm_count*2][:,None],
        normals[nrm_count*2:nrm_count*3][:,None]
    ), axis=1)
    save_image(shade_normals(normals.reshape(nrm_res, nrm_res, 3)), './result/optimized_normal.png')

    plt.clf()
    plt.title('Root Mean Squared Error')
    for pose_id in range(len(poses)):
        plt.plot(list(range(len(rmse_record[pose_id]))), rmse_record[pose_id], label=f'Pose {pose_id}')
    plt.legend()
    plt.savefig('./result/rmse.png')