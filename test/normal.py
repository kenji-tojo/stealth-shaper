import numpy as np
from skimage.transform import resize
import math
from PIL import Image


def create_normal_map(res: int, type='ball') -> np.ndarray:
    assert type in ['none', 'ball']

    width  = res*4
    height = res*4

    normals = np.zeros((width, height, 3), dtype=np.float32)

    for iw in range(width):
        for ih in range(height):
            if type == 'ball':
                x = -1.+2.*(iw+.5)/width
                y = -1.+2.*(ih+.5)/height
                z = -.5

                if x*x+y*y+z*z > 1.:
                    normals[iw,ih,2] = 1.
                    continue

                n = -np.array([x,y,-math.sqrt(1.-(x*x+y*y))], dtype=np.float32)
                norm = np.linalg.norm(n)
                assert abs(1.-norm) < 1e-5
                normals[iw,ih] = n/norm
            else:
                normals[iw,ih,2] = 1. ## unit z

    normals = resize(normals, (res,res,3), anti_aliasing=True)
    return normals / np.linalg.norm(normals, axis=2)[:,:,None]


def shade_normals(normals: np.ndarray) -> np.ndarray:
    img = normals.copy()
    img_shape = img.shape
    img = img.reshape(-1,3)

    vmin = .2
    vmax = .99
    for ip in range(len(img)):
        n = img[ip]
        n0 = .5*(1.+n[0])
        n1 = .5*(1.+n[1])
        n2 = .5*(1.+n[2])
        img[ip][0] = vmin + (vmax-vmin) * n0
        img[ip][1] = vmin + (vmax-vmin) * n1
        img[ip][2] = vmin + (vmax-vmin) * n2

    return img.reshape(img_shape)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=64, help='resolution of a normal map image')
    parser.add_argument('--albedo', type=float, default=.8, help='albedo for reflectance visualization')
    args = parser.parse_args()

    res: int = args.res
    print(f'creating normal map: resolution = {res}x{res}')
    normals = create_normal_map(res)

    os.makedirs('./result', exist_ok=True)

    Image.fromarray(np.round(shade_normals(normals)*255.).clip(0.,255.).astype(np.uint8)).save('./result/normal.png')