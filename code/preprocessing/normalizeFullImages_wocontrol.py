import numpy as np
import h5py
import os
import pandas as pd
import scipy.misc as misc
from tqdm import tqdm

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
proj_root = os.path.join(script_dir, os.pardir, os.pardir)

illum_path = os.path.join(proj_root, 'input/illum/')
processed_path = os.path.join(proj_root, 'input/processed/')
info_csv = pd.read_csv(os.path.join(processed_path, 'full_images_wocontrol.csv'))

images = h5py.File(os.path.join(processed_path,
                                "full_images_wocontrol.hdf5"), "r")['images']

f = h5py.File(os.path.join(processed_path, "full_images_wocontrol_illumcorr.hdf5"), "w")
corr_images = f.create_dataset("images", images.shape)

pbar = tqdm(total=len(images))
for i, plate in enumerate(info_csv['plate']):
    normalizer = np.zeros(images.shape[1:])
    dapi_file = os.path.join(illum_path, '%s_IllumDAPI.png' % plate)
    normalizer[:, :, 0] = misc.imread(os.path.join(illum_path, dapi_file))

    tubulin_file = os.path.join(illum_path, '%s_IllumTubulin.png' % plate)
    normalizer[:, :, 1] = misc.imread(os.path.join(illum_path, tubulin_file))

    actin_file = os.path.join(illum_path, '%s_IllumActin.png' % plate)
    normalizer[:, :, 2] = misc.imread(os.path.join(illum_path, tubulin_file))

    corr_images[i, :] = images[i, :] / normalizer
    maxima = np.max(corr_images[i, :])
    minima = np.min(corr_images[i, :])
    pbar.update(1)

max_perc = np.percentile(maxima, 95)
min_perc = np.percentile(minima, 5)
resc = (max_perc - min_perc)

for i in xrange(len(corr_images)):
    tmp_rescaled = (corr_images[i, :] - min_perc) / resc
    corr_images[i, :] = np.minimum(np.maximum(tmp_rescaled, 0.0), 1.0)

f.close()
pbar.close()
