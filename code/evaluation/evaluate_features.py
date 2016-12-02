import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from itertools import chain
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--metric", help="metric to use",
                    choices=['cosine', 'euclidean', 'manhattan', 'chebyshev',
                             'minkowski', 'wminkowski', 'seuclidean',
                             'mahalanobis'],
                    default='euclidean')
parser.add_argument("-f", "--filename", help="base file name",
                    default="mosaic_sampled_size224_off0_stride224")
args = parser.parse_args()

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
proj_root = os.path.join(script_dir, os.pardir, os.pardir)

eval_path = os.path.join(proj_root, 'input/eval/')
features_path = os.path.join(proj_root, 'input/features/')

base_file_name = args.filename

feature_csv = pd.read_csv(os.path.join(features_path,
                                       '%s.csv' % base_file_name),
                          index_col=0)
print feature_csv.shape

feature_columns = [col for col in feature_csv.columns.values
                   if 'feature' in col]

print 'found %d features' % len(feature_columns)

tmp_features = feature_csv[~(feature_csv['compound'] == 'DMSO')]
# tmp_features = tmp_features[~((tmp_features['compound'] == 'taxol')
#                               & (tmp_features['compound'] == 0.3))]

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

trueLabels = []

kNNpredLabels = []
svcpredLabels = []
naivepredLabels = []
np.set_printoptions(formatter={'float': '{: 0.3f}'.format},
                    linewidth=150)

if args.metric != 'cosine':
    kNN = KNeighborsClassifier(n_neighbors=1, metric=args.metric)
else:
    raise Warning('cosine is not implemented but euclidean distance provides the same ordering: see http://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance or http://stackoverflow.com/questions/34144632/using-cosine-distance-with-scikit-learn-kneighborsclassifier')
    kNN = KNeighborsClassifier(n_neighbors=1,
                               metric='euclidean')
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
with open(os.path.join(eval_path,
                       'eval_%s.log' % (base_file_name)),
          'w') as logfile:
    print >>logfile, '1NN:'
    print >>logfile, nn_confMatrix
    print >>logfile, nn_acc

unq_moas_list = list(unique_moas)
with open(os.path.join(eval_path,
                       'eval_%s.conf' % (base_file_name)),
          'w') as conf_file:
    for i in xrange(len(unique_moas)):
        for j in xrange(len(unique_moas)):
            print >>conf_file, ('%s\t%s\t%d'
                                % (unq_moas_list[i], unq_moas_list[j],
                                   nn_confMatrix[i, j]))
