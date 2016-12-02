# Automating Morphological Profiling with Generic Deep Convolutional Networks

## Abstract
Morphological profiling aims to create signatures of genes, chemicals and diseases from microscopy images. Current approaches use classical computer vision-based segmentation and feature extraction. Deep learning models achieve state-of-the-art performance in many computer vision tasks such as classification and segmentation. We propose to transfer activation features of generic deep convolutional networks to extract features for morphological profiling. Our approach surpasses currently used methods in terms of accuracy and processing speed. Furthermore, it enables fully automated processing of microscopy images without need for single cell identification.

## Reproduction of results
### Setup of environment

This code uses input from the `./input` directory. It also uses code from `https://github.com/tensorflow/models/tree/master/inception` , `https://github.com/ry/tensorflow-vgg16` and `https://github.com/ry/tensorflow-resnet`. And the pipeline from Singh S, Bray MA, Jones TR, Carpenter AE (2014). Pipeline for illumination correction of images for high-throughput microscopy. Journal of Microscopy 256(3):231-6 / doi. PMID: 25228240 `http://d1zymp9ayga15t.cloudfront.net/PublishedPipelines/JMicroscopy_Singh_2014.zip`
