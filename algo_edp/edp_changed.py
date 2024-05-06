from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras import Model, layers, models
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.utils import array_to_img


import tensorflow as tf
import tensorflow
import matplotlib.pyplot as plt
import numpy as np  
import scipy.misc
from matplotlib.pyplot import imshow




from numpy.random import seed
seed(1988)
tensorflow.random.set_seed(1988)


base_model=tf.keras.applications.DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)



kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.0)


WEIGHT_DECAY = 2e-4



def conv2d(kernel_size, stride, filters, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), padding="same", use_bias=False,
           kernel_initializer="he_normal", **kwargs):
    return layers.Conv2D(kernel_size=kernel_size, strides=stride, filters=filters, kernel_regularizer=kernel_regularizer, padding=padding,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, **kwargs)

def Inception(x,nb_filter):  
#     branch1x1 = Conv2D(nb_filter,(1,1),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(x)  
    branch1x1 = CondConv2D(kernel_size=(1,1), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch1x1")(x)

#     branch3x3 = Conv2D(nb_filter,(1,1),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(x) 
    branch3x3 = CondConv2D(kernel_size=(1,1), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch3x3")(x)

#     branch3x31 = Conv2D(nb_filter,(3,1),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(branch3x3)  
    branch3x31 = CondConv2D(kernel_size=(3,1), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch3x31")(branch3x3)

    
#     branch3x32 = Conv2D(nb_filter,(1,3),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(branch3x3) 
    branch3x32 = CondConv2D(kernel_size=(1,3), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch3x32")(branch3x3)

    out1= layers.Add()([branch3x31, branch3x32])
    
#     branch5x5 = Conv2D(nb_filter,(1,1),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(x) 
    branch5x5 = CondConv2D(kernel_size=(1,1), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch5x5")(x)

#     branch5x5_1 = Conv2D(nb_filter,(3,1),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(branch5x5) 
    branch5x5_1 = CondConv2D(kernel_size=(3,1), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch5x5_1")(branch5x5)

#     branch5x5_2 = Conv2D(nb_filter,(1,3),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(branch5x5) 
    branch5x5_2 = CondConv2D(kernel_size=(1,3), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch5x5_2")(branch5x5)

    out2= layers.Add()([branch5x5_1, branch5x5_2])

#     branch5x51 = Conv2D(nb_filter,(3,1),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(out2) 
    branch5x51 = CondConv2D(kernel_size=(3,1), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch5x51")(out2)

#     branch5x52 = Conv2D(nb_filter,(1,3),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(out2) 
    branch5x52 = CondConv2D(kernel_size=(1,3), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branch5x52")(out2)

    out3= layers.Add()([branch5x51, branch5x52])
    
    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)  
#     branchpool = Conv2D(nb_filter,(1,1),padding='same',strides=(1,1),activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(branchpool) 
    branchpool = CondConv2D(kernel_size=(1,1), filters=nb_filter, stride=1, padding='same', use_bias=True,  num_experts=3, name="branchpool")(branchpool)

    
    x = concatenate([branch1x1,out1,out3,branchpool],axis=3)    
    return x  

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def sse_block(input_feature, ratio=4):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    se_feature = Dense(channel // ratio,
                        activation='relu',
                        kernel_initializer='he_normal',
                        use_bias=True,
                        bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel//ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1,1,channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])

    mean = tf.reduce_mean(input_feature, 3)
    mean = tf.expand_dims(mean, -1)
    std =  tf.math.reduce_std(input_feature, 3) 
    std = tf.expand_dims(std, -1)
    max = tf.reduce_max(input_feature, 3)
    max = tf.expand_dims(max, -1)

    se_feature = layers.Concatenate()([se_feature, mean, std, max])

    return se_feature


class Patches(layers.Layer): 
    def __init__(self, patch_size,**kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        return {'patch_size': self.patch_size}


    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,**kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=self.num_patches, output_dim=self.projection_dim)
    
    def get_config(self):
        return {'num_patches': self.num_patches,
               'projection_dim':self.projection_dim} 


    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class Routing(layers.Layer):
    def __init__(self, out_channels, dropout_rate, temperature=30, **kwargs):
        super(Routing, self).__init__(**kwargs)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.fc = layers.Dense(units=out_channels)
        self.softmax = layers.Softmax()
        self.temperature = temperature

    def call(self, inputs, **kwargs):
        """
        :param inputs: (b, c, h, w)
        :return: (b, out_features)
        """
        out = self.avgpool(inputs)
        out = self.dropout(out)

        # refer to paper: https://arxiv.org/pdf/1912.03458.pdf
        out = self.softmax(self.fc(out) * 1.0 / self.temperature)
        return out

class CondConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, stride=1, use_bias=True, num_experts=3, padding="same", **kwargs):
        super(CondConv2D, self).__init__(**kwargs)

        self.routing = Routing(out_channels=num_experts, dropout_rate=0.2, name="routing_layer")
        self.convs = []
        for _ in range(num_experts):
            self.convs.append(conv2d(filters=filters, stride=stride, kernel_size=kernel_size, use_bias=use_bias, padding=padding))

    def call(self, inputs, **kwargs):
        """
        :param inputs: (b, h, w, c)
        :return: (b, h_out, w_out, filters)
        """
        routing_weights = self.routing(inputs)
        feature = routing_weights[:, 0] * tf.transpose(self.convs[0](inputs), perm=[1, 2, 3, 0])
        for i in range(1, len(self.convs)):
            feature += routing_weights[:, i] * tf.transpose(self.convs[i](inputs), perm=[1, 2, 3, 0])
        feature = tf.transpose(feature, perm=[3, 0, 1, 2])
        return feature

image_size =56
patch_size = 5  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 32
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4



x1 = base_model.get_layer('conv1/relu').output
# x1 = sse_block(x1, ratio=4)
x1 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv1")(x1)
x1 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv2")(x1)
x1 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv3")(x1)
# x1 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv13")(x1)
x1 = Conv2D(kernel_size=3, filters=29, strides=1, padding='valid', use_bias=True)(x1)
x_conv1 = sse_block(x1, ratio=4)

x2 = base_model.get_layer('conv2_block3_concat').output
# x2 = sse_block(x2, ratio=4)
x2 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv4")(x2)
x2 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv25")(x2)
x2 = Conv2D(kernel_size=3, filters=29, strides=1, padding='valid', use_bias=True)(x2)
x_conv2 = sse_block(x2, ratio=4)

x_inc = base_model.get_layer('conv2_block6_concat').output
x_inc = Inception(x_inc,32)
# x_inc = sse_block(x_inc, ratio=4)
x3 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv6")(x_inc)
x3 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv7")(x3)
x3 = Conv2D(kernel_size=3, filters=29, strides=1, padding='valid', use_bias=True)(x3)
x_conv3 = sse_block(x3, ratio=4)

patches = Patches(patch_size)(x_inc)
    # Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
# # Create multiple layers of the Transformer block.
for _ in range(transformer_layers):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])

# Create a [batch_size, projection_dim] tensor.
x = layers.LayerNormalization(epsilon=1e-6, name='cam_layer')(encoded_patches)

x_t = layers.Reshape((11,11,32))(x)

x = layers.Add()([x_conv1, x_conv2, x_conv3, x_t])
x = sse_block(x, ratio=4)

x = layers.GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# Load the saved model

model.load_weights('augmented_LCCViT_Maize_custom_best_weights_maize.h5')
# Load and preprocess the input image
img_path = 'download.jpg'
# Assuming your model has 4 classes, adjust this number based on your actual number of classes
num_classes = 4
print("Initialisations done...")

import cv2
import io


def analysis(frame):
    
    
    image_size =56
    patch_size = 5  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 32
    num_heads = 4
    transformer_units = [
    projection_dim * 2,
    projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 4



    x1 = base_model.get_layer('conv1/relu').output
    # x1 = sse_block(x1, ratio=4)
    x1 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv1")(x1)
    x1 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv2")(x1)
    x1 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv3")(x1)
    # x1 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv13")(x1)
    x1 = Conv2D(kernel_size=3, filters=29, strides=1, padding='valid', use_bias=True)(x1)
    x_conv1 = sse_block(x1, ratio=4)

    x2 = base_model.get_layer('conv2_block3_concat').output
    # x2 = sse_block(x2, ratio=4)
    x2 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv4")(x2)
    x2 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv25")(x2)
    x2 = Conv2D(kernel_size=3, filters=29, strides=1, padding='valid', use_bias=True)(x2)
    x_conv2 = sse_block(x2, ratio=4)

    x_inc = base_model.get_layer('conv2_block6_concat').output
    x_inc = Inception(x_inc,32)
    # x_inc = sse_block(x_inc, ratio=4)
    x3 = CondConv2D(kernel_size=3, filters=16, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv6")(x_inc)
    x3 = CondConv2D(kernel_size=3, filters=32, stride=2, padding='valid', use_bias=True,  num_experts=3, name="condi_conv7")(x3)
    x3 = Conv2D(kernel_size=3, filters=29, strides=1, padding='valid', use_bias=True)(x3)
    x_conv3 = sse_block(x3, ratio=4)

    patches = Patches(patch_size)(x_inc)
        # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    x = layers.LayerNormalization(epsilon=1e-6, name='cam_layer')(encoded_patches)

    x_t = layers.Reshape((11,11,32))(x)

    x = layers.Add()([x_conv1, x_conv2, x_conv3, x_t])
    x = sse_block(x, ratio=4)

    x = layers.GlobalAveragePooling2D()(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)


    # Load the saved model

    model.load_weights('augmented_LCCViT_Maize_custom_best_weights_maize.h5')
    # Load and preprocess the input image
    img_path = 'download.jpg'
    # Assuming your model has 4 classes, adjust this number based on your actual number of classes
    num_classes = 4
    print("Initialisations done...")
    print(frame)
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print(img_array)
    predictions = model.predict(img_array)
    row=predictions[0]
    labels = []

    print(row)
    for val in row:
        if val == 0.0:
            labels.append('common rust')
        elif val == 1.0:  
            labels.append('grey leaf spot')
        elif val == 2.0:
            labels.append('healthy')
        elif val == 3.0:
            labels.append('northern leaf blight')
        else:
            labels.append('unknown')
    print(predictions)
    print("hello")
    print(labels)
    return row






# ----------------------------------------------VIDEO FEED-------------------------------------------------------------






import cv2
import io
cap = cv2.VideoCapture(0)

# print("Starting video feed...")
# c=0
# while c!=1:
#     _, frame = cap.read()
    
    

        
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(frame_rgb, (224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     predictions = model.predict(img_array)
#     row=predictions[0]
#     labels = []

#     print(row)
#     for val in row:
#         if val == 0.0:
#             labels.append('common rust')
#         elif val == 1.0:  
#             labels.append('grey leaf spot')
#         elif val == 2.0:
#             labels.append('healthy')
#         elif val == 3.0:
#             labels.append('northern leaf blight')
#         else:
#             labels.append('unknown')

#     print(labels)
#     c+=1


#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break

# cap.release()
# cv2.destroyAllWindows()