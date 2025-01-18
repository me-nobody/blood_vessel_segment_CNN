import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms as T

import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import segmentation_models_pytorch as smp

model_path = "/users/ad394h/Documents/segment_blood_vessels/models/models_2_test/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_drp2 = smp.Unet(
    encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",            # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,
    activation = None,                     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    encoder_depth =5,
    decoder_channels = [256,128,64,32,16],
    classes=2,                             # model output channels (number of classes in your dataset)
    aux_params=dict(
        pooling = 'avg',                   # one of 'avg', 'max'
        dropout= 0.2,                      # dropout ratio, default is None
        activation = 'sigmoid',            # activation function, default is None
        classes=2                          # define number
                )
)

torch.save(model_drp2, model_path+"Unet_efficientnet_2_class_dropout_2.pt")

model_drp3 = smp.Unet(
    encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",            # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,
    activation = None,                     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    encoder_depth =5,
    decoder_channels = [256,128,64,32,16],
    classes=2,                             # model output channels (number of classes in your dataset)
    aux_params=dict(
        pooling = 'avg',                   # one of 'avg', 'max'
        dropout= 0.3,                      # dropout ratio, default is None
        activation = 'sigmoid',            # activation function, default is None
        classes=2                          # define number
                )
)

torch.save(model_drp3, model_path+"Unet_efficientnet_2_class_dropout_3.pt")

model_drp5 = smp.Unet(
    encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",            # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,
    activation = None,                     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    encoder_depth =5,
    decoder_channels = [256,128,64,32,16],
    classes=2,                             # model output channels (number of classes in your dataset)
    aux_params=dict(
        pooling = 'avg',                   # one of 'avg', 'max'
        dropout= 0.5,                      # dropout ratio, default is None
        activation = 'sigmoid',            # activation function, default is None
        classes=2                          # define number
                )
)

torch.save(model_drp5, model_path+"Unet_efficientnet_2_class_dropout_5.pt")
