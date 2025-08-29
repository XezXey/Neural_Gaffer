import os
import sys
import mint_logging
import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_frames', default=60, type=int)
parser.add_argument('--lighting_per_view', default=5, type=int)
parser.add_argument('--image_dir', required=True, type=str)
parser.add_argument('--lighting_dir', nargs='+', required=True, type=str)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--sample_pair_json', nargs='+', type=str)
parser.add_argument('--idx', nargs='+', default=[-1], type=int, help='Index of the subject to process. Use -1 to process all subjects.')
parser.add_argument('--main_process_port', default='25539', type=str)
parser.add_argument('--save_dir', required=True, type=str)
args = parser.parse_args()

logger = mint_logging.createLogger()

"""
CUDA_VISIBLE_DEVICES=0 accelerate launch 
    --main_process_port 25539 
    --config_file configs/1_16fp.yaml neural_gaffer_inference_real_data.py 
    --output_dir logs/neural_gaffer_res256 
    --mixed_precision fp16 
    --resume_from_checkpoint latest 
    --total_view 60 
    --lighting_per_view 5 
    --val_img_dir "./difareli++/input_subject/preprocessed/img" 
    --val_lighting_dir "./difareli++_rotate_lighting/azimuth/" 
    --save_dir ./difareli++/results/azimuth/
"""

if __name__ == '__main__':
    for json_file in args.sample_pair_json:
        with open(json_file, 'r') as f:
            sample_pairs = json.load(f)['pair']
            sample_pairs_k = [k for k in sample_pairs.keys()]
            sample_pairs_v = [v for v in sample_pairs.values()]
            
        if len(args.idx) > 2:
            # Filter idx to be within 0 < idx < len(sample_pairs)
            to_run_idx = [i for i in args.idx if 0 <= i < len(sample_pairs)]
        elif args.idx == [-1]:
            s = 0
            e = len(sample_pairs)
            to_run_idx = list(range(s, e))
        elif len(args.idx) == 2:
            s, e = args.idx
            s = max(0, s)
            e = min(e, len(sample_pairs))
            to_run_idx = list(range(s, e))
        else:
            raise ValueError("Invalid index range provided. Please provide a valid range or -1 for all indices.")

        for ii, idx in enumerate(to_run_idx):
            pair = sample_pairs_v[idx]
            pair_id = sample_pairs_k[idx]
            # fn = f'{pair_id}_src={pair["src"]}_dst={pair["dst"]}'

            subject = pair["src"]
            for lighting in args.lighting_dir:
                img_path = f"{args.image_dir}/{subject.split('.')[0]}/img/"
                cmd = f"""
                    CUDA_VISIBLE_DEVICES={args.gpu_id} python \
                        neural_gaffer_inference_real_data.py \
                        --output_dir logs/neural_gaffer_res256 \
                        --mixed_precision fp16 \
                        --resume_from_checkpoint latest \
                        --total_view {args.n_frames} \
                        --lighting_per_view {args.lighting_per_view} \
                        --val_img_dir {img_path} \
                        --val_lighting_dir {lighting} \
                        --save_dir {args.save_dir}
                """
                logger.error(f"="*111)
                logger.warning(f"[#] Running on json: {json_file} with {to_run_idx}")
                logger.info(f"[#] Progress: {ii}/{len(to_run_idx)}")
                logger.info(f"[#] Running on subject: {subject}.")
                logger.info(f"[#] Lightning folder: {lighting}.")
                logger.info(f"[#] Image path: {img_path}")
                logger.info(f"[#] Command: {cmd.strip()}")
                logger.error(f"="*111)
                os.system(cmd)