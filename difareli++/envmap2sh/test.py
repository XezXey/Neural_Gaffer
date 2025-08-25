import numpy as np
from sh_utils import get_shcoeff, unfold_sh_coeff, flatten_sh_coeff, apply_integrate_conv, sample_from_sh, genSurfaceNormals, cartesian_to_spherical, from_x_left_to_z_up
from tonemapper import TonemapHDR
import matplotlib.pyplot as plt
import skimage
import torchvision
import tqdm
from envmap import EnvironmentMap, rotation_matrix
import warnings
import torch as th
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--spl', default=None, help='Use a single point')
parser.add_argument('--axis', default='azimuth', help='Rotation axis')
parser.add_argument('--normal_map', default=None, help='Path to normal map')
args = parser.parse_args()

tonemapper = TonemapHDR()
ORDER = 2
map_name = "117_hdrmaps_com_free_2K.exr"
# map_name = "128_hdrmaps_com_free_2K.exr"
# map_name = "125_hdrmaps_com_free_2K.exr"

def render(hdr_image, normal_map):
    hdr_tm, _, _ = tonemapper(hdr_image)

    coeff = get_shcoeff(hdr_image, Lmax=2)
    sh = flatten_sh_coeff(coeff, max_sh_level=2)

    unfolded = unfold_sh_coeff(sh, max_sh_level=2)

    apply_integrated = apply_integrate_conv(unfolded.copy())

    if normal_map is None:
        normal_map = genSurfaceNormals(256).permute(1, 2, 0).cpu().numpy()  # H, W, C
        T = th.tensor([[0.,0.,1.],
                        [1.,0.,0.],
                        [0.,1.,0.]])                       # maps [x,y,z] -> [z,x,y]
        normal_map = th.einsum('ij,hwj->hwi', T, th.tensor(normal_map).float())
        normal_map = normal_map.cpu().numpy()
    else:
        normal_map = normal_map
        T = th.tensor([[0.,0.,1.],
                        [1.,0.,0.],
                        [0.,1.,0.]])                       # maps [x,y,z] -> [z,x,y]
        normal_map = th.einsum('ij,hwj->hwi', T, th.tensor(normal_map).float())
        normal_map = normal_map.cpu().numpy()
    
    # mask = (normal_map[..., 0:1] != 0)
    mask = 1.0

    theta, phi = cartesian_to_spherical(normal_map)
    shading = sample_from_sh(apply_integrated, lmax=ORDER, theta=theta, phi=phi)

    shading = np.float32(shading)
    shading, _, _ = tonemapper(shading) # tonemap
    shading = np.clip(shading, 0, 1)
    shading = skimage.img_as_ubyte(shading) / 255.

    
    return ((normal_map + 1) * 0.5) * mask, shading * mask

def generate_frame(hdr_image, i, axis, normal_map):
    # hdr_image_roll = np.roll(hdr_image.copy(), shift=-i, axis=1)
    # print(axis)
    rot_deg = i*np.pi/180
    dcm = rotation_matrix(azimuth=rot_deg if axis == 'azimuth' else 0,
                        elevation=rot_deg if axis == 'elevation' else 0,
                        roll=rot_deg if axis == 'roll' else 0)
    e = EnvironmentMap(hdr_image, 'latlong')
    e_rot = e.copy().rotate(dcm)
    hdr_image_rot = e_rot.data    # np.array of shape [H, W, 3], min: 0, max: 1
    normal_map, shading = render(hdr_image_rot, normal_map)
    print(np.max(normal_map), np.min(normal_map))
    print(np.max(shading), np.min(shading))
    tgt_w = normal_map.shape[1] + shading.shape[1]
    
    # resize hdr_image_roll but still preserve aspect ratio
    hdr_image_rot = skimage.transform.resize(hdr_image_rot, (normal_map.shape[0], tgt_w), anti_aliasing=True)
    hdr_image_rot, _, _ = tonemapper(hdr_image_rot)
    
    hdr_image_original = skimage.transform.resize(hdr_image, (normal_map.shape[0], tgt_w), anti_aliasing=True)
    hdr_image_original, _, _ = tonemapper(hdr_image_original)
    


    frame = np.concatenate((hdr_image_original, hdr_image_rot, 
                        np.concatenate((normal_map, shading), axis=1)), axis=0)
    
    # print(np.max(hdr_image_roll), np.min(hdr_image_roll))
    # print(np.max(normal_map), np.min(normal_map))
    # print(np.max(shading), np.min(shading))
    return frame

if __name__ == '__main__':
    hdr_map = f"/home/mint/Dev/DiFaReli++/TPAMI_baseline_MajorRevision/Neural_Gaffer/demo/environment_map_sample/{map_name}"
    hdr_image = skimage.io.imread(hdr_map)
    hdr_image = skimage.img_as_float(hdr_image)
    
    if args.spl is not None:
        print("[#] Using a single point light mode.")
        
        desired_pos = args.spl  # 'middle', 'left-top', 'right-top', 'left-bottom', 'right-bottom', 'left-middle', 'right-middle', 'middle-top', 'middle-bottom'
        max_idx = np.unravel_index(np.argmax(hdr_image, axis=None), hdr_image.shape)
        max_val = hdr_image[max_idx]
        hdr_image = np.zeros_like(hdr_image)
        H, W = hdr_image.shape[:2]
        xq = [W // 4, W // 2, (3 * W) // 4]  # left, middle, right
        yq = [H // 4, H // 2, (3 * H) // 4]  # top, middle, bottom

        pos_map = {
            'left-top':      (xq[0], yq[0]),
            'middle-top':    (xq[1], yq[0]),
            'right-top':     (xq[2], yq[0]),
            'left-middle':   (xq[0], yq[1]),
            'middle':        (xq[1], yq[1]),
            'right-middle':  (xq[2], yq[1]),
            'left-bottom':   (xq[0], yq[2]),
            'middle-bottom': (xq[1], yq[2]),
            'right-bottom':  (xq[2], yq[2]),
        }

        if desired_pos not in pos_map:
            raise ValueError(f"Unknown desired_pos: {desired_pos}")

        px, py = pos_map[desired_pos]
        hdr_image[py, px] = max_val
        pf = f'pl_{desired_pos}'
    else:
        print("[#] Using full HDR environment map.")
        pf = 'hdr'
        
    if args.normal_map is None:
        print("[#] Use generated normal map.")
        nm = 'gen'
        normal_map = None
    else: 
        normal_map = np.load(args.normal_map)
        # Assume shape is T x 3 x H x W
        assert normal_map.shape[1] == 3
        normal_map = normal_map.transpose(0, 2, 3, 1)
        assert np.all([np.allclose(x, normal_map[0], rtol=1e-03) for x in normal_map])
        normal_map = normal_map[0]
        nm = 'deca'
        print(f"[#] Loaded normal map from {args.normal_map}, shape: {normal_map.shape}")

    import multiprocessing as mp
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # 'pool.imap' applies the 'generate_frame' function to each item in 'shift_values'.
        # It's used here instead of 'pool.map' because it works better with tqdm's progress bar.
        # The list() wrapper collects all the results.
        shift_values = np.linspace(0, 360, 60).astype(int)
        frames = pool.starmap(generate_frame, [(hdr_image, i, args.axis, normal_map) for i in shift_values])
    frames = (np.stack(frames).clip(0, 1) * 255).astype(int)
    torchvision.io.write_video(f"out_{pf}_{args.axis}_{nm}.mp4", frames, fps=24)
