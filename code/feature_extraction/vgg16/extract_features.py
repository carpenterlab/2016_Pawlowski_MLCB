import numpy as np
import h5py
import os
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from itertools import chain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--channelasimage", help="use_channel_as_image",
                    action="store_true")
parser.add_argument("-f", "--filename", help="base file name",
                    default="mosaic_sampled_size224_off0_stride224")
parser.add_argument("-b", "--batchsize", help="batch size", type=int,
                    default=128)
args = parser.parse_args()

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
proj_root = os.path.join(script_dir, os.pardir, os.pardir, os.pardir)

processed_path = os.path.join(proj_root, 'input/processed/')
features_path = os.path.join(proj_root, 'input/features/')

base_file_name = args.filename

info_csv = pd.read_csv(os.path.join(processed_path, '%s.csv' % base_file_name),
                       index_col=0)

images = h5py.File(os.path.join(processed_path,
                                "%s.hdf5" % base_file_name),
                   "r")['images']

use_channel_as_image = True if args.channelasimage else False
channel_app = "_channelasimg" if use_channel_as_image else ""
batch_size = args.batchsize

input_placeholder = tf.placeholder("float", [None, 224, 224, 3])
with open("vgg16.tfmodel", mode='rb') as f:
    print('Loading VGG16 model')
    fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    tf.import_graph_def(graph_def, input_map={"images": input_placeholder})

    graph = tf.get_default_graph()

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    graph = tf.get_default_graph()
    features = graph.get_tensor_by_name("import/fc8/Reshape:0")

    feature_columns = ['feature%d' % i for i in xrange(features.get_shape()[1])]

    print('Processing images')
    pbar = tqdm(total=len(images))
    feature_columns = ['feature%d' % i for i in xrange(features.get_shape()[1])]
    if use_channel_as_image:
        feature_columns = (['chan0_feature%d' % i for i in xrange(features.get_shape()[1])]
                           + ['chan1_feature%d' % i for i in xrange(features.get_shape()[1])]
                           + ['chan2_feature%d' % i for i in xrange(features.get_shape()[1])])
    tmp_feat_df = np.zeros((len(images), len(feature_columns)))
    if not(images.shape[1] == 224 and images.shape[2] == 224):
        res_place = tf.placeholder(tf.float32, (None,) + images.shape[1:])
        res_imgs = tf.image.resize_images(res_place, (224, 224))
    for idx in xrange(0, images.shape[0], batch_size):
        tmp_imgs = np.array(images[idx:idx+batch_size, :])
        if tmp_imgs.shape[1] != 224 or tmp_imgs.shape[2] != 224:
            tmp_imgs = sess.run(res_imgs, feed_dict={res_place: tmp_imgs})
        if use_channel_as_image:
            for ch_idx in xrange(3):
                input_data = tmp_imgs[:, :, :, ch_idx][:, :, :, np.newaxis].repeat(3, axis=-1)
                extracted_features = \
                    sess.run(features,
                             feed_dict={input_placeholder: input_data})
                start_feature_idx = ch_idx * extracted_features.shape[1]
                end_feature_idx = (ch_idx + 1) * extracted_features.shape[1]
                tmp_feat_df[idx:idx+batch_size,
                            start_feature_idx:end_feature_idx] = extracted_features

        else:
            input_data = tmp_imgs
            extracted_features = \
                sess.run(features,
                         feed_dict={input_placeholder: input_data})
            tmp_feat_df[idx:idx+batch_size] = extracted_features
        pbar.update(batch_size)

    if idx + 1 < images.shape[0]:
        tmp_imgs = np.array(images[idx:, :])
        if tmp_imgs.shape[1] != 224 or tmp_imgs.shape[2] != 224:
            tmp_imgs = sess.run(res_imgs, feed_dict={res_place: tmp_imgs})
        if use_channel_as_image:
            for ch_idx in xrange(3):
                input_data = tmp_imgs[:, :, :, ch_idx][:, :, :, np.newaxis].repeat(3, axis=-1)
                extracted_features = \
                    sess.run(features,
                             feed_dict={input_placeholder: input_data})
                start_feature_idx = ch_idx * extracted_features.shape[1]
                end_feature_idx = (ch_idx + 1) * extracted_features.shape[1]
                tmp_feat_df[idx:,
                            start_feature_idx:end_feature_idx] = extracted_features
        else:
            input_data = tmp_imgs
            extracted_features = \
                sess.run(features,
                         feed_dict={input_placeholder: input_data})
            tmp_feat_df[idx:idx+batch_size] = extracted_features
    tmp_df = pd.DataFrame(tmp_feat_df, columns=feature_columns)
    feature_csv = pd.concat([info_csv, tmp_df], axis=1)
    pbar.close()

print('Processing extracted features')
feature_csv.to_csv(os.path.join(features_path,
                                'vgg16_%s%s.csv' % (base_file_name, channel_app)))

tmp_features = feature_csv[~(feature_csv['compound'] == 'DMSO')]
tmp_features = tmp_features[~((tmp_features['compound'] == 'taxol')
                              & (tmp_features['compound'] == 0.3))]

compounds = tmp_features['compound'].drop_duplicates()
compounds.sort_values(inplace=True)
unique_moas = tmp_features['moa'].drop_duplicates()
unique_moas.sort_values(inplace=True)

labels = np.array([list(unique_moas).index(elem) for elem
                   in tmp_features['moa']], dtype=int)

tmp_features['labels'] = labels

tmp_groups = tmp_features[['compound', 'concentration', 'labels']
                          + feature_columns].groupby(['compound',
                                                      'concentration'])
lat_reps = tmp_groups.mean().iloc[:, :]
lat_reps['labels'] = tmp_groups['labels'].apply(lambda x: np.unique(x)[0])
lat_reps.to_csv(os.path.join(features_path,
                'vgg16_reps_%s%s.csv' % (base_file_name, channel_app)))


trueLabels = []

kNNpredLabels = []
svcpredLabels = []
naivepredLabels = []
np.set_printoptions(formatter={'float': '{: 0.3f}'.format},
                    linewidth=150)
kNN = KNeighborsClassifier(n_neighbors=1)
for numC, compound in enumerate(compounds):
    print 'leaving out %s , %d of %d' % (compound, numC,
                                         len(compounds))
    leaveout = [compound]

    mask = lat_reps.index.isin(leaveout, level=0)

    tmp = lat_reps.drop(['labels'], axis=1)

    trainFeatures = tmp[~mask]
    trainLabels = lat_reps[~mask]['labels']

    testFeatures = lat_reps.drop('labels', axis=1)[mask]
    testLabels = lat_reps[mask]['labels']

    trueLabels.append(testLabels)

    kNN.fit(trainFeatures, trainLabels)
    preds = kNN.predict(testFeatures)
    kNNpredLabels.append(preds)
    knnAcc = kNN.score(testFeatures, testLabels)
    print '1NN acc is %.2f %%' % (knnAcc * 100)

print '1NN:'
nn_confMatrix = \
    confusion_matrix(
        np.fromiter(chain.from_iterable(trueLabels),
                    dtype=int),
        np.fromiter(chain.from_iterable(kNNpredLabels),
                    dtype=int))
print nn_confMatrix
nn_acc = np.sum(nn_confMatrix.diagonal()) / np.sum(nn_confMatrix,
                                                   dtype=np.float)
print nn_acc
with open(os.path.join(features_path,
                       'vgg16_reps_%s%s.log' % (base_file_name, channel_app)),
          'w') as logfile:
    print >>logfile, '1NN:'
    print >>logfile, nn_confMatrix
    print >>logfile, nn_acc
