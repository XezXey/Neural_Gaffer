import numpy as np
import torch as th
import os, glob, argparse, tqdm
import torchvision
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', required=True, type=str, help='Path to the input result directory')
parser.add_argument('--out_dir', required=True, type=str, help='Path to the output directory')
args = parser.parse_args()

def proc(path, subject):
    out_dict = {}
    input_image = sorted(glob.glob(f'{path}/{subject}/input_image/*.png'))
    pred_image = sorted(glob.glob(f'{path}/{subject}/pred_image/*.png'))
    target_envmap_ldr = sorted(glob.glob(f'{path}/{subject}/target_envmap_ldr/*.png'))
    
    for i in range(len(input_image)):
        envmap_name = '_'.join(os.path.basename(input_image[i]).split('_')[:-1])
        if out_dict.get(envmap_name) is None:
            out_dict[envmap_name] = {
                'input_image_path': [input_image[i]],
                'pred_image_path': [pred_image[i]],
                'target_envmap_ldr_path': [target_envmap_ldr[i]],
                'input_image': [np.array(Image.open(input_image[i]))],
                'pred_image': [np.array(Image.open(pred_image[i]))],
                'target_envmap_ldr': [np.array(Image.open(target_envmap_ldr[i]))],
            }
        else:
            out_dict[envmap_name]['input_image_path'].append(input_image[i])
            out_dict[envmap_name]['pred_image_path'].append(pred_image[i])
            out_dict[envmap_name]['target_envmap_ldr_path'].append(target_envmap_ldr[i])
            out_dict[envmap_name]['input_image'].append(np.array(Image.open(input_image[i])))
            out_dict[envmap_name]['pred_image'].append(np.array(Image.open(pred_image[i])))
            out_dict[envmap_name]['target_envmap_ldr'].append(np.array(Image.open(target_envmap_ldr[i])))

    # print("[#] Copying...", end='')
    for k in out_dict.keys():
        save_dir = f'{args.out_dir}/{subject}/{k}/'
        os.makedirs(save_dir, exist_ok=True)
        length = len(out_dict[k]['input_image'])
        for i in range(length):
            input_path = out_dict[k]['input_image_path'][i]
            pred_path = out_dict[k]['pred_image_path'][i]
            target_path = out_dict[k]['target_envmap_ldr_path'][i]
            os.system(f'cp {input_path} {save_dir}/input_{i:04d}.png')
            os.system(f'cp {pred_path} {save_dir}/pred_{i:04d}.png')
            os.system(f'cp {target_path} {save_dir}/target_{i:04d}.png')
    # print("Done")
    
    t = tqdm.tqdm(list(out_dict.keys()), leave=False)

    for k in t:
        t.set_description(f"[#] Environment map: {k}")
        save_dir = f'{args.out_dir}/{subject}/{k}/'
        out_dict[k]['input_image'] = np.stack(out_dict[k]['input_image'])
        out_dict[k]['pred_image'] = np.stack(out_dict[k]['pred_image'])
        out_dict[k]['target_envmap_ldr'] = np.stack(out_dict[k]['target_envmap_ldr'])
        
        if out_dict[k]['input_image'].shape[0] != out_dict[k]['pred_image'].shape[0] or out_dict[k]['input_image'].shape[0] != out_dict[k]['target_envmap_ldr'].shape[0]:
            print("===> Length mismatch detected.")
            continue

        torchvision.io.write_video(filename=f'{save_dir}/{subject}_{k}_input.mp4', video_array=out_dict[k]['input_image'], fps=24, options = {"crf": "17"})
        torchvision.io.write_video(filename=f'{save_dir}/{subject}_{k}_input_rt.mp4', video_array=np.concatenate((out_dict[k]['input_image'], out_dict[k]['input_image'][::-1, ...])), fps=24, options = {"crf": "17"})
        torchvision.io.write_video(filename=f'{save_dir}/{subject}_{k}_pred.mp4', video_array=out_dict[k]['pred_image'], fps=24, options = {"crf": "17"})
        torchvision.io.write_video(filename=f'{save_dir}/{subject}_{k}_pred_rt.mp4', video_array=np.concatenate((out_dict[k]['pred_image'], out_dict[k]['pred_image'][::-1, ...])), fps=24, options = {"crf": "17"})
        torchvision.io.write_video(filename=f'{save_dir}/{subject}_{k}_target.mp4', video_array=out_dict[k]['target_envmap_ldr'], fps=24, options = {"crf": "17"})
        torchvision.io.write_video(filename=f'{save_dir}/{subject}_{k}_target_rt.mp4', video_array=np.concatenate((out_dict[k]['target_envmap_ldr'], out_dict[k]['target_envmap_ldr'][::-1, ...])), fps=24, options = {"crf": "17"})

        all_out = np.concatenate([out_dict[k]['input_image'], out_dict[k]['pred_image'], out_dict[k]['target_envmap_ldr']], axis=2)
        torchvision.io.write_video(f'{args.out_dir}/{subject}_{k}_out.mp4', all_out, fps=24, options = {"crf": "17"})
        torchvision.io.write_video(f'{args.out_dir}/{subject}_{k}_out_rt.mp4', np.concatenate((all_out, all_out[::-1]), axis=0), fps=24, options = {"crf": "17"})

if __name__ == '__main__':
    path = args.res_dir
    os.makedirs(args.out_dir, exist_ok=True)
    subject_folder = os.listdir(path)
    t = tqdm.tqdm(subject_folder)
    for subject in t:
        t.set_description(f"[#] Subject: {subject}")
        if subject == 'video': continue
        proc(path=path, subject=subject)
    