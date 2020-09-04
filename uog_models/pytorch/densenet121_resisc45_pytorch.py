"""
DenseNet121 CNN model for 244x244x3 image classification

Model contributed by: AngusG, adapted from armory pytorch ResNet50 example and
keras densenet121_resisc45 example model.
"""
import logging

from art.classifiers import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from tensorflow.keras.preprocessing import image

from armory.data.utils import maybe_download_weights_from_s3


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#RESISC_MEANS = [0.485, 0.456, 0.406]
#RESISC_STDEV = [0.229, 0.224, 0.225]

RESISC_MEANS = [0.36386173189316956, 0.38118692953271804, 0.33867067558870334]
RESISC_STDEV = [0.20350874, 0.18531173, 0.18472934]

num_classes = 45

'''
def preprocessing_fn(img):
    """
    Standardize, then normalize RESISC images
    """
    # Standardize images to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Normalize images RESISC means
    for i, (mean, std) in enumerate(zip(RESISC_MEANS, RESISC_STDEV)):
        img[i] -= mean
        img[i] /= std

    return img
'''

def mean_std():
    resisc_mean = np.array(
        [0.36386173189316956, 0.38118692953271804, 0.33867067558870334,]
    )

    resisc_std = np.array([0.20350874, 0.18531173, 0.18472934])

    return resisc_mean, resisc_std


def preprocess_input_densenet121_resisc(img):
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
    output = preprocess_input_densenet121_resisc(np.array(output)).transpose(0, 3, 1, 2).astype('float32')  # from NHWC to NCHW
    return output


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    #model = models.resnet50(**model_kwargs)

    # throwing permission error accessing ~/.cache when pretrained=True/False
    #model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
    model = models.densenet121(pretrained=False)

    # manually implement Keras 'include_top=False' option
    model_newtop = nn.Sequential(
        model.features,
        nn.AvgPool2d(kernel_size=7),
        nn.Flatten(),
        nn.Linear(1024, num_classes),
        nn.Softmax()
    ).to(DEVICE)

    if weights_file:
        pass
        #filepath = maybe_download_weights_from_s3(weights_file)
        #checkpoint = torch.load(filepath, map_location=DEVICE)
        #model.load_state_dict(checkpoint)

    #mean, std = mean_std()
    wrapped_model = PyTorchClassifier(
        model_newtop,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.003),
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
