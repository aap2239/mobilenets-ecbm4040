import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    DepthwiseConv2D,
    Conv2D,
    BatchNormalization,
    ReLU,
    AvgPool2D,
    Flatten,
    Dense,
)
from tensorflow.keras import Model, Sequential


class MobileNet(Model):
    def __init__(
        self,
        num_classes,
        alpha=1,
        input_shape_=(224, 224, 3),
    ) -> None:
        super(
            MobileNet,
            self,
        ).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.input_shape_ = input_shape_

        self.model = Sequential()

        ## Input Layers
        self.model.add(Input(shape=self.input_shape_))
        self.model.add(
            Conv2D(
                filters=32,
                kernel_size=3,
                strides=2,
                padding="same",
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(ReLU())

        self._mobilenet_block(
            filters=64,
            strides=1,
        )
        self._mobilenet_block(
            filters=128,
            strides=2,
        )
        self._mobilenet_block(
            filters=128,
            strides=1,
        )
        self._mobilenet_block(
            filters=256,
            strides=2,
        )
        self._mobilenet_block(
            filters=256,
            strides=1,
        )
        self._mobilenet_block(
            filters=512,
            strides=2,
        )

        for _ in range(5):
            self._mobilenet_block(
                filters=512,
                strides=1,
            )

        self._mobilenet_block(
            filters=1024,
            strides=2,
        )

        self._mobilenet_block(
            filters=1024,
            strides=1,
        )

        self.model.add(
            AvgPool2D(
                pool_size=7,
                strides=1,
                data_format="channels_first",
            )
        )
        # self.model.add(Flatten())

        if self.num_classes == 2:
            self.model.add(
                Dense(
                    units=1,
                    activation="sigmoid",
                )
            )
        else:
            self.model.add(
                Dense(
                    units=self.num_classes,
                    activation="softmax",
                )
            )

    def _mobilenet_block(
        self,
        filters,
        strides,
    ):
        self.model.add(
            DepthwiseConv2D(
                kernel_size=3,
                strides=strides,
                depth_multiplier=self.alpha,
                padding="same",
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(
            Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(ReLU())

    def call(
        self,
        inputs=(None, 224, 224, 3),
    ):
        return self.model
