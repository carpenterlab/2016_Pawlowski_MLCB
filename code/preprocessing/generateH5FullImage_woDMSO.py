import scipy.misc as misc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import h5py

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
proj_root = os.path.join(script_dir, os.pardir, os.pardir)


image_csv = pd.read_csv(os.path.join(proj_root,
                                     'input/raw/BBBC021_v1_image.csv'))
moa_csv = pd.read_csv(os.path.join(proj_root, 'input/raw/BBBC021_v1_moa.csv'))

combined = pd.merge(image_csv, moa_csv,
                    how='inner',
                    left_on=('Image_Metadata_Compound',
                             'Image_Metadata_Concentration'),
                    right_on=('compound', 'concentration'))

combined = combined[~(combined['Image_Metadata_Compound'] == 'DMSO')]


outCSV = pd.DataFrame(columns=('compound', 'concentration', 'moa', 'plate',
                               'well', 'replicate'))

pbar = tqdm(total=len(combined))

indizes = None
images = None
img_shape = (1024, 1280, 3)
f = h5py.File(os.path.join(proj_root, "input/processed/full_images_dmso.hdf5"), "w")
images = f.create_dataset("images", (len(combined),) + img_shape)
curFile = 0

for row in zip(combined['Image_FileName_DAPI'],
               combined['Image_FileName_Tubulin'],
               combined['Image_FileName_Actin'],
               combined['Image_PathName_DAPI'],
               combined['compound'], combined['concentration'],
               combined['moa'], combined['Image_Metadata_Plate_DAPI'],
               combined['Image_Metadata_Well_DAPI'], combined['Replicate']):

    dapi_file = row[1]
    tubulin_file = row[0]
    actin_file = row[2]
    directory = row[3]

    compound = row[4]
    concentration = str(row[5])
    moa = row[6]
    plate = row[7]
    well = row[8]
    replicate = row[9]

    image_directory = os.path.join(proj_root, 'input/raw/', directory)
    c1 = misc.imread(os.path.join(image_directory, dapi_file))
    c2 = misc.imread(os.path.join(image_directory, tubulin_file))
    c3 = misc.imread(os.path.join(image_directory, actin_file))

    img = np.zeros(c1.shape + (3,))
    img[:, :, 0] = c1
    img[:, :, 1] = c2
    img[:, :, 2] = c3

    img = img / (2.**16)
    images[curFile, :] = img

    curFile += 1

    outCSV.loc[len(outCSV)] = [compound, concentration, moa,
                               plate, well, replicate]

    pbar.update(1)

f.close()

outCSV.to_csv(os.path.join(proj_root, 'input/processed/full_images_dmso.csv'))
pbar.close()
