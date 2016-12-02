import numpy as np
import h5py
import os
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--illum", help="turn on illum correction",
                    action="store_true")
parser.add_argument("-e", "--excludecontrol", help="exclude control",
                    action="store_true")
parser.add_argument("-n", "--num", help="num samples", type=int,
                    default=15)
parser.add_argument("-s", "--seed", help="seed", type=int,
                    default=42)
parser.add_argument("-w", "--size", help="sample size", type=int,
                    default=299)
args = parser.parse_args()

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
proj_root = os.path.join(script_dir, os.pardir, os.pardir)

exclude_control = True if args.excludecontrol else False
excludecontrol_app = "_wocontrol" if exclude_control else ""

processed_path = os.path.join(proj_root, 'input/processed/')
info_csv = pd.read_csv(os.path.join(processed_path,
                                    'full_images%s.csv' % excludecontrol_app),
                       index_col=0)

use_illumcorr = True if args.illum else False
illumcor_app = "_illumcorr" if use_illumcorr else ""
sample_size = args.size

seed = args.seed
# source images are 1280x1024, the image samples are 299 x 299
# balanced sampling uses 14 - 15 samples
# <14 samples is undersampled, >15 samples is oversampled
# for 224 x 224 optimum is 26
num_samples = args.num

print('generating samples with size %d num %d seed %d %s'
      % (sample_size, num_samples, seed,
         "and illum corr" if use_illumcorr else ""))


rng = np.random.RandomState(seed)
images = h5py.File(os.path.join(processed_path,
                                "full_images%s%s.hdf5"
                                % (excludecontrol_app, illumcor_app)),
                   "r")['images']

filename = ('rand_sampled%s%s_seed%d_size%d_num%d'
            % (excludecontrol_app, illumcor_app, seed, sample_size, num_samples))

f = h5py.File(os.path.join(processed_path, "%s.hdf5" % filename), "w")
sampled_images = f.create_dataset("images", (images.shape[0] * num_samples,
                                             sample_size,
                                             sample_size,
                                             3))

outCSV = pd.DataFrame(columns=info_csv.columns)
pbar = tqdm(total=len(images))
for i in xrange(images.shape[0]):
    for j in xrange(num_samples):
        outCSV = outCSV.append(info_csv.loc[i])
        r_x = rng.randint(images.shape[1] - sample_size)
        r_y = rng.randint(images.shape[2] - sample_size)
        sampled_images[i * num_samples + j] = images[i,
                                                     r_x:r_x+sample_size,
                                                     r_y:r_y+sample_size,
                                                     :]
    pbar.update(1)

outCSV.reset_index(drop=True, inplace=True)
outCSV.to_csv(os.path.join(processed_path, "%s.csv" % filename))

f.close()
pbar.close()
