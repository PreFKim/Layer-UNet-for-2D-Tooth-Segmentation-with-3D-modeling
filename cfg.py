import model
from losses import Focal_IoU

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400

num_classes = 1

unet_level = 5

initial_features = 32

model = model.layerUNET(
    n_levels=unet_level,
    DSV=True,
    initial_features=initial_features,
    IMAGE_HEIGHT=IMAGE_HEIGHT,
    IMAGE_WIDTH=IMAGE_WIDTH,
    out_channels=num_classes
    )

loss = Focal_IoU
