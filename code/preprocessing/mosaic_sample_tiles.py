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
parser.add_argument("-o", "--offset", help="offset", type=int,
                    default=0)
parser.add_argument("-s", "--stride", help="stride", type=int,
                    default=299)
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

offset = args.offset
stride = args.stride

images = h5py.File(os.path.join(processed_path,
                                "full_images%s%s.hdf5"
                                % (excludecontrol_app, illumcor_app)),
                   "r")['images']
print('generating samples with size %d stride %d offset %d %s'
      % (sample_size, stride, offset,
         "and illum corr" if use_illumcorr else ""))
filename = ('mosaic_sampled%s%s_size%d_off%d_stride%d'
            % (excludecontrol_app, illumcor_app, sample_size, offset, stride))

num_x = ((images.shape[1] - offset - sample_size) / stride) + 1
num_y = ((images.shape[2] - offset - sample_size) / stride) + 1

f = h5py.File(os.path.join(processed_path, "%s.hdf5" % filename), "w")
sampled_images = f.create_dataset("images", (images.shape[0] * num_x * num_y,
                                             sample_size,
                                             sample_size,
                                             3))

outCSV = pd.DataFrame(columns=info_csv.columns)
pbar = tqdm(total=len(images))
sample_idx = 0
for i in xrange(images.shape[0]):
    for n_x in xrange(num_x):
        for n_y in xrange(num_y):
            outCSV = outCSV.append(info_csv.iloc[i])
            pos_x = offset + n_x * stride
            pos_y = offset + n_y * stride
            sampled_images[sample_idx] = images[i,
                                                pos_x:pos_x+sample_size,
                                                pos_y:pos_y+sample_size,
                                                :]
            sample_idx += 1
    pbar.update(1)

outCSV.reset_index(drop=True, inplace=True)
outCSV.to_csv(os.path.join(processed_path, "%s.csv" % filename))

f.close()
pbar.close()
