import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    DepthwiseConv2D,
    Conv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
)
from tensorflow.keras import Model, Sequential

class MobileNet(Model):
    def __init__(
        self,
        num_classes,
        alpha = 1,
        input_shape_=(224, 224, 3),
        **kwargs,
    ) -> None:
        super(
            MobileNet,
            self,
        ).__init__(**kwargs)
        self.alpha = alpha
        self.num_classes = num_classes
        self.input_shape_ = input_shape_

        self.conv2d_1 = Conv2D(
            filters = 32,
            kernel_size = 3,
            strides = 2,
            padding = "same",
        )
        self.batch_norm_1 = BatchNormalization()
        self.relu_1 = ReLU()

        ## MobileNet Block 1
        self.depthwise_conv2d_1 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, 
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_1_1 = BatchNormalization()
        self.dw_relu_1_1 = ReLU()
        self.dw_conv2d_1 = Conv2D(
            filters = 64,
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_1_2 = BatchNormalization()
        self.dw_relu_1_2 = ReLU()

        ## MobileNet Block 2
        self.depthwise_conv2d_2 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 2, 
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_2_1 = BatchNormalization()
        self.dw_relu_2_1 = ReLU()
        self.dw_conv2d_2 = Conv2D(
            filters = 128,
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_2_2 = BatchNormalization()
        self.dw_relu_2_2 = ReLU()

        ## MobileNet Block 3
        self.depthwise_conv2d_3 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_3_1 = BatchNormalization()
        self.dw_relu_3_1 = ReLU()
        self.dw_conv2d_3 = Conv2D(
            filters = 128, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_3_2 = BatchNormalization()
        self.dw_relu_3_2 = ReLU()

        ## MobileNet Block 4
        self.depthwise_conv2d_4 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 2, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_4_1 = BatchNormalization()
        self.dw_relu_4_1 = ReLU()
        self.dw_conv2d_4 = Conv2D(
            filters = 256, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_4_2 = BatchNormalization()
        self.dw_relu_4_2 = ReLU()

        ## MobileNet Block 5
        self.depthwise_conv2d_5 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_5_1 = BatchNormalization()
        self.dw_relu_5_1 = ReLU()
        self.dw_conv2d_5 = Conv2D(
            filters = 256, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_5_2 = BatchNormalization()
        self.dw_relu_5_2 = ReLU()

        ## MobileNet Block 6
        self.depthwise_conv2d_6 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 2, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_6_1 = BatchNormalization()
        self.dw_relu_6_1 = ReLU()
        self.dw_conv2d_6 = Conv2D(
            filters = 512, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_6_2 = BatchNormalization()
        self.dw_relu_6_2 = ReLU()

        ## MobileNet Block 7
        self.depthwise_conv2d_7 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_7_1 = BatchNormalization()
        self.dw_relu_7_1 = ReLU()
        self.dw_conv2d_7 = Conv2D(
            filters = 512, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_7_2 = BatchNormalization()
        self.dw_relu_7_2 = ReLU()

        ## MobileNet Block 8
        self.depthwise_conv2d_8 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_8_1 = BatchNormalization()
        self.dw_relu_8_1 = ReLU()
        self.dw_conv2d_8 = Conv2D(
            filters = 512, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_8_2 = BatchNormalization()
        self.dw_relu_8_2 = ReLU()

        ## MobileNet Block 9
        self.depthwise_conv2d_9 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_9_1 = BatchNormalization()
        self.dw_relu_9_1 = ReLU()
        self.dw_conv2d_9 = Conv2D(
            filters = 512, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_9_2 = BatchNormalization()
        self.dw_relu_9_2 = ReLU()

        ## MobileNet Block 10
        self.depthwise_conv2d_10 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_10_1 = BatchNormalization()
        self.dw_relu_10_1 = ReLU()
        self.dw_conv2d_10 = Conv2D(
            filters = 512, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_10_2 = BatchNormalization()
        self.dw_relu_10_2 = ReLU()

        ## MobileNet Block 11
        self.depthwise_conv2d_11 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_11_1 = BatchNormalization()
        self.dw_relu_11_1 = ReLU()
        self.dw_conv2d_11 = Conv2D(
            filters = 512, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_11_2 = BatchNormalization()
        self.dw_relu_11_2 = ReLU()

        ## MobileNet Block 12
        self.depthwise_conv2d_12 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 2, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_12_1 = BatchNormalization()
        self.dw_relu_12_1 = ReLU()
        self.dw_conv2d_12 = Conv2D(
            filters = 1024, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_12_2 = BatchNormalization()
        self.dw_relu_12_2 = ReLU()

        ## MobileNet Block 13
        self.depthwise_conv2d_13 = DepthwiseConv2D(
            kernel_size = 3,
            strides = 1, ## strides
            depth_multiplier = self.alpha,
            padding = "same",
        )
        self.dw_batch_norm_13_1 = BatchNormalization()
        self.dw_relu_13_1 = ReLU()
        self.dw_conv2d_13 = Conv2D(
            filters = 1024, ##filters
            kernel_size = 1,
            strides = 1,
        )
        self.dw_batch_norm_13_2 = BatchNormalization()
        self.dw_relu_13_2 = ReLU()

        ## Average Pool 
        self.avg_pool = GlobalAveragePooling2D(
#             pool_size = 7,
#             strides = 1,
            data_format = "channels_first",
        )

        ##Output 
        if self.num_classes < 2: 
            raise ValueError(
                f"Invalid Number of classes provided {self.num_classes}. Should be greater than 2."
            )
        elif self.num_classes == 2:
            self.dense_output = Dense(
                units = 1,
                activation = "sigmoid",
            )
        else:
            self.dense_output = Dense(
                units=self.num_classes,
                activation="softmax",
            )

    def call(
        self,
        x,
    ):
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x) 

        x = self.depthwise_conv2d_1(x)
        x = self.dw_batch_norm_1_1(x)
        x = self.dw_relu_1_1(x)
        x = self.dw_conv2d_1(x)
        x = self.dw_batch_norm_1_2(x)
        x = self.dw_relu_1_2(x)

        x = self.depthwise_conv2d_2(x)
        x = self.dw_batch_norm_2_1(x)
        x = self.dw_relu_2_1(x)
        x = self.dw_conv2d_2(x)
        x = self.dw_batch_norm_2_2(x)
        x = self.dw_relu_2_2(x)

        x = self.depthwise_conv2d_3(x)
        x = self.dw_batch_norm_3_1(x)
        x = self.dw_relu_3_1(x)
        x = self.dw_conv2d_3(x)
        x = self.dw_batch_norm_3_2(x)
        x = self.dw_relu_3_2(x)
        
        x = self.depthwise_conv2d_4(x)
        x = self.dw_batch_norm_4_1(x)
        x = self.dw_relu_4_1(x)
        x = self.dw_conv2d_4(x)
        x = self.dw_batch_norm_4_2(x)
        x = self.dw_relu_4_2(x)

        x = self.depthwise_conv2d_5(x)
        x = self.dw_batch_norm_5_1(x)
        x = self.dw_relu_5_1(x)
        x = self.dw_conv2d_5(x)
        x = self.dw_batch_norm_5_2(x)
        x = self.dw_relu_5_2(x)

        x = self.depthwise_conv2d_6(x)
        x = self.dw_batch_norm_6_1(x)
        x = self.dw_relu_6_1(x)
        x = self.dw_conv2d_6(x)
        x = self.dw_batch_norm_6_2(x)
        x = self.dw_relu_6_2(x)

        x = self.depthwise_conv2d_7(x)
        x = self.dw_batch_norm_7_1(x)
        x = self.dw_relu_7_1(x)
        x = self.dw_conv2d_7(x)
        x = self.dw_batch_norm_7_2(x)
        x = self.dw_relu_7_2(x)

        x = self.depthwise_conv2d_8(x)
        x = self.dw_batch_norm_8_1(x)
        x = self.dw_relu_8_1(x)
        x = self.dw_conv2d_8(x)
        x = self.dw_batch_norm_8_2(x)
        x = self.dw_relu_8_2(x)

        x = self.depthwise_conv2d_9(x)
        x = self.dw_batch_norm_9_1(x)
        x = self.dw_relu_9_1(x)
        x = self.dw_conv2d_9(x)
        x = self.dw_batch_norm_9_2(x)
        x = self.dw_relu_9_2(x)

        x = self.depthwise_conv2d_10(x)
        x = self.dw_batch_norm_10_1(x)
        x = self.dw_relu_10_1(x)
        x = self.dw_conv2d_10(x)
        x = self.dw_batch_norm_10_2(x)
        x = self.dw_relu_10_2(x)

        x = self.depthwise_conv2d_11(x)
        x = self.dw_batch_norm_11_1(x)
        x = self.dw_relu_11_1(x)
        x = self.dw_conv2d_11(x)
        x = self.dw_batch_norm_11_2(x)
        x = self.dw_relu_11_2(x)

        x = self.depthwise_conv2d_12(x)
        x = self.dw_batch_norm_12_1(x)
        x = self.dw_relu_12_1(x)
        x = self.dw_conv2d_12(x)
        x = self.dw_batch_norm_12_2(x)
        x = self.dw_relu_12_2(x)

        x = self.depthwise_conv2d_13(x)
        x = self.dw_batch_norm_13_1(x)
        x = self.dw_relu_13_1(x)
        x = self.dw_conv2d_13(x)
        x = self.dw_batch_norm_13_2(x)
        x = self.dw_relu_13_2(x)

        x = self.avg_pool(x)
        x = self.dense_output(x)

        return x
    
    def make(self, input_shape=(224, 224, 3)):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
        print(model.summary())
        return model
