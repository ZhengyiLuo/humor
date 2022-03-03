import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
from subprocess import Popen
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--s", type=int, default=0)
    parser.add_argument("--e", type=int, default=5)
    args = parser.parse_args()

    num_gpus = args.num_gpus
    gpu_start = args.gpu
    split_idxes = list(range(args.s, args.e))
    for split_idx in split_idxes:
        gpu_idx = split_idx % num_gpus  + gpu_start
        cmd = f"python humor/fitting/run_fitting.py @./configs/fit_proxd_next.cfg --start {split_idx} --end {split_idx + 1}"
        p = Popen(cmd.split(" "), env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_idx)))
    p.wait()


    