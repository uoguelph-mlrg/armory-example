"""
VGG-11 BN CNN model for 244x244x3 image classification

Model contributed by: AngusG, adapted from armory pytorch ResNet50 example and
keras densenet121_resisc45 example model.
"""
import logging

from art.classifiers import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# AngusG for loading weights
import os
import os.path as osp
from armory import paths
from tensorflow.keras.preprocessing import image

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESISC_MEANS = [0.36386173189316956, 0.38118692953271804, 0.33867067558870334]
RESISC_STDEV = [0.20350874, 0.18531173, 0.18472934]

num_classes = 45


def mean_std():
    resisc_mean = np.array(
        [0.36386173189316956, 0.38118692953271804, 0.33867067558870334,]
    )

    resisc_std = np.array([0.20350874, 0.18531173, 0.18472934])

    return resisc_mean, resisc_std


def preprocess_input_resisc(img):
    # Model was trained with Caffe preprocessing on the images
    # load the mean and std of the [0,1] normalized dataset
    # Normalize images: divide by 255 for [0,1] range
    mean, std = mean_std()
    img_norm = img / 255.0
    # Standardize the dataset on a per-channel basis
    output_img = (img_norm - mean) / std
    return output_img


def preprocessing_fn(x: np.ndarray) -> np.ndarray:
    shape = (224, 224)  # Expected input shape of model
    output = []
    for i in range(x.shape[0]):
        im_raw = image.array_to_img(x[i])
        im = image.img_to_array(im_raw.resize(shape))
        output.append(im)
    output = preprocess_input_resisc(
        np.array(output)).transpose(0, 3, 1, 2).astype('float32')  # from NHWC to NCHW
    return output


def resisc_vgg11bn(model_kwargs):

    model = models.vgg11_bn(**model_kwargs)

    # manually implement Keras 'include_top=False' option
    return nn.Sequential(
        model.features,
        nn.AvgPool2d(kernel_size=7),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):

    # set TORCH_HOME to prevent permission error accessing ~/.cache when pretrained=True
    os.environ['TORCH_HOME'] = paths.runtime_paths().saved_model_dir

    model_pre_softmax = resisc_vgg11bn(model_kwargs)
    model = nn.Sequential(model_pre_softmax, nn.Softmax(dim=1)).to(DEVICE)

    if weights_file:
        checkpoint = torch.load(
            osp.join(paths.runtime_paths().saved_model_dir, weights_file), map_location=DEVICE)
        model_pre_softmax.load_state_dict(checkpoint)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=False),
        input_shape=(3, 224, 224),
        nb_classes=num_classes,
        **wrapper_kwargs,
        clip_values=(
            np.array(
                [
                    0.0 - RESISC_MEANS[0] / RESISC_STDEV[0],
                    0.0 - RESISC_MEANS[1] / RESISC_STDEV[1],
                    0.0 - RESISC_MEANS[2] / RESISC_STDEV[2],
                ]
            ),
            np.array(
                [
                    1.0 - RESISC_MEANS[0] / RESISC_STDEV[0],
                    1.0 - RESISC_MEANS[1] / RESISC_STDEV[1],
                    1.0 - RESISC_MEANS[2] / RESISC_STDEV[2],
                ]
            ),
        ),
    )
    return wrapped_model
