import numpy as np
from sh_utils import get_shcoeff, unfold_sh_coeff, flatten_sh_coeff, apply_integrate_conv, apply_integrate_conv_anyLmax, sample_from_sh, genSurfaceNormals, cartesian_to_spherical, from_x_left_to_z_up
from tonemapper import TonemapHDR
import matplotlib.pyplot as plt
import skimage
from PIL import Image
import torchvision
import tqdm
from envmap import EnvironmentMap, rotation_matrix
import warnings
import torch as th
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--spl', default=None, help='Use a single point')
parser.add_argument('--apl', default=None, help='Use an area point light')
parser.add_argument('--axis', default='azimuth', help='Rotation axis')
parser.add_argument('--normal_map', default=None, help='Path to normal map')
parser.add_argument('--Lmax', default=2, type=int, help='Maximum SH order')
args = parser.parse_args()

tonemapper = TonemapHDR()
ORDER = args.Lmax
map_name = "117_hdrmaps_com_free_2K.exr"
# map_name = "128_hdrmaps_com_free_2K.exr"
# map_name = "125_hdrmaps_com_free_2K.exr"

def render(hdr_image, face, Lmax):
    hdr_tm, _, _ = tonemapper(hdr_image)

    coeff = get_shcoeff(hdr_image, Lmax=Lmax)   # 3, 2, Lmax+1, Lmax+1
    # print("COEFF: ", coeff.shape)
    sh = flatten_sh_coeff(coeff, max_sh_level=Lmax) # 3, (Lmax+1)^2
    # print("SH: ", sh.shape)
    unfolded = unfold_sh_coeff(sh, max_sh_level=Lmax)   # 3, 2, Lmax+1, Lmax+1
    # print("UNFOLDED: ", unfolded.shape)

    # apply_integrated = apply_integrate_conv(unfolded.copy(), Lmax)
    apply_integrated = apply_integrate_conv_anyLmax(unfolded.copy(), Lmax)

    if face is None:
        normal_map_org, mask = genSurfaceNormals(256)  # H, W, C
        normal_map_org = normal_map_org.permute(1, 2, 0).cpu().numpy()
        normal_map = normal_map_org.copy()
        mask = mask.cpu().numpy()[..., None]
        T = th.tensor([[0.,0.,1.],
                        [1.,0.,0.],
                        [0.,1.,0.]])                       # maps [x,y,z] -> [z,x,y]
        normal_map = th.einsum('ij,hwj->hwi', T, th.tensor(normal_map).float()).cpu().numpy()
        normal_map = normal_map
    else:
        normal_map_org = face['normal_map']
        normal_map = normal_map_org.copy()
        mask = face['alpha_map']
        T = th.tensor([[0.,0.,1.],
                        [1.,0.,0.],
                        [0.,1.,0.]])                       # maps [x,y,z] -> [z,x,y]
        normal_map = th.einsum('ij,hwj->hwi', T, th.tensor(normal_map).float()).cpu().numpy()
        normal_map = normal_map
    

    theta, phi = cartesian_to_spherical(normal_map)
    shading = sample_from_sh(apply_integrated, lmax=ORDER, theta=theta, phi=phi)
    if face is not None:
        shading = shading * face['albedo']

    shading = np.float32(shading)
    # shading, _, _ = tonemapper(shading) # tonemap
    # shading = np.clip(shading, 0, 1)
    shading = skimage.img_as_ubyte(shading)
    
    # Shading with grey-scale
    shading_grey = np.array(Image.fromarray(shading.copy()).convert('L'))[..., None]
    shading_grey = np.repeat(shading_grey, 3, axis=2)
    # print(shading_grey.dtype, shading_grey.shape)
    
    shading = shading / 255.
    shading_grey = shading_grey / 255.
    
    return ((normal_map_org + 1) * 0.5) * mask, ((normal_map + 1) * 0.5) * mask, shading * mask, shading_grey * mask

def generate_frame(hdr_image, i, axis, face, Lmax):
    # hdr_image_roll = np.roll(hdr_image.copy(), shift=-i, axis=1)
    # print(axis)
    rot_deg = i*np.pi/180
    dcm = rotation_matrix(azimuth=rot_deg if axis == 'azimuth' else 0,
                        elevation=rot_deg if axis == 'elevation' else 0,
                        roll=rot_deg if axis == 'roll' else 0)
    e = EnvironmentMap(hdr_image, 'latlong')
    e_rot = e.copy().rotate(dcm)
    hdr_image_rot = e_rot.data    # np.array of shape [H, W, 3], min: 0, max: 1
    normal_map_org, normal_map, shading, shading_grey = render(hdr_image_rot, face, Lmax)
    # print(np.max(normal_map), np.min(normal_map))
    # print(np.max(shading), np.min(shading))
    # print(np.max(shading_grey), np.min(shading_grey))
    # print(normal_map.shape, shading.shape, shading_grey.shape)
    
    tgt_w = normal_map_org.shape[1] + normal_map.shape[1] + shading.shape[1] + shading_grey.shape[1]
    
    # resize hdr_image_roll but still preserve aspect ratio
    hdr_image_rot = skimage.transform.resize(hdr_image_rot, (normal_map.shape[0], tgt_w), anti_aliasing=True)
    hdr_image_rot, _, _ = tonemapper(hdr_image_rot)
    
    hdr_image_original = skimage.transform.resize(hdr_image, (normal_map.shape[0], tgt_w), anti_aliasing=True)
    hdr_image_original, _, _ = tonemapper(hdr_image_original)
    


    frame = np.concatenate((hdr_image_original, hdr_image_rot, 
                        np.concatenate((normal_map_org, normal_map, shading, shading_grey), axis=1)), axis=0)
    
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
        # hdr_image = np.ones_like(hdr_image) * (max_val * 1e-06)
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
    elif args.apl is not None:
        print("[#] Using an area point light mode.")
        desired_pos = args.apl  # e.g., 'left', 'middle', 'right'
        max_idx = np.unravel_index(np.argmax(hdr_image, axis=None), hdr_image.shape)
        max_val = hdr_image[max_idx]
        # hdr_image = np.zeros_like(hdr_image)
        hdr_image = np.ones_like(hdr_image) * (max_val * 0.01)
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
        radius = 30
        # Circle area light
        hdr_image[max(0, py-radius):min(H, py+radius), max(0, px-radius):min(W, px+radius)] = max_val
        pf = f'apl_{desired_pos}'
        
    else:
        print("[#] Using full HDR environment map.")
        pf = f'hdr_{map_name}'
        
    if args.normal_map is None:
        print("[#] Use generated normal map.")
        nm = 'gen'
        face = None
    else: 
        normal_map = np.load(args.normal_map)
        alpha_map = np.load(args.normal_map.replace('normal', 'alpha'))
        abledo = np.load(args.normal_map.replace('normal', 'albedo'))
        # Assume shape is T x 3 x H x W
        assert normal_map.shape[1] == 3
        assert abledo.shape[1] == 3
        assert alpha_map.shape[1] == 1
        normal_map = normal_map.transpose(0, 2, 3, 1)
        alpha_map = alpha_map.transpose(0, 2, 3, 1)
        albedo = abledo.transpose(0, 2, 3, 1)
        
        assert np.all([np.allclose(x, normal_map[0], rtol=1e-03) for x in normal_map])
        assert np.all([np.allclose(x, alpha_map[0], rtol=1e-03) for x in alpha_map])
        assert np.all([np.allclose(x, albedo[0], rtol=1e-03) for x in albedo])
        
        normal_map = normal_map[0]
        alpha_map = alpha_map[0]
        albedo = albedo[0]

        face = {'normal_map': normal_map, 'alpha_map': alpha_map, 'albedo': albedo}

        nm = 'deca'
        print(f"[#] Loaded normal map from {args.normal_map}, shape: {normal_map.shape}")

    import multiprocessing as mp
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # 'pool.imap' applies the 'generate_frame' function to each item in 'shift_values'.
        # It's used here instead of 'pool.map' because it works better with tqdm's progress bar.
        # The list() wrapper collects all the results.
        shift_values = np.linspace(0, 360, 60).astype(int)
        frames = pool.starmap(generate_frame, [(hdr_image, i, args.axis, face, args.Lmax) for i in shift_values])
    frames = (np.stack(frames).clip(0, 1) * 255).astype(int)
    torchvision.io.write_video(f"out_{pf}_{args.axis}_{nm}_Lmax{args.Lmax}.mp4", frames, fps=24)
