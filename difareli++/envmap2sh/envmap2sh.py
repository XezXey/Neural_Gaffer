# create a spherical harmonic from a chromeball image

import os
import numpy as np 
import skimage
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool 
from tqdm.auto import tqdm
from sh_utils import get_shcoeff, unfold_sh_coeff, flatten_sh_coeff

INPUT_DIR = "../diffusionlight_test/predicted_iter2/envmap"
OUTPUT_DIR = "../diffusionlight_test/predicted_iter2/shcoeffs"

def process_file(image_name):
    
    image = skimage.io.imread(os.path.join(INPUT_DIR, image_name))
    image = skimage.img_as_float(image)
    coeff = get_shcoeff(image, Lmax=2)
    shcoeff = flatten_sh_coeff(coeff, max_sh_level=2)
    np.save(os.path.join(OUTPUT_DIR, image_name.replace(".png", ".npy")), shcoeff)
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(os.listdir(INPUT_DIR))
    with Pool(16) as p:
        list(tqdm(p.imap(process_file, files), total=len(files)))

if __name__ == "__main__":
    main()