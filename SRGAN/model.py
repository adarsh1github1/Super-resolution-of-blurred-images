import tensorflow as tf
import numpy
#modules
#conv block -- upsampling block --


def normalize(x):
    return x / 255.0 # normalize to 0-1
def denormalize(x):
    return x * 255.0 #return back to original shape
def pixelshuffler(scale):
    return lambda x: tf.nn.depth_to_space(x,scale)

def conv_block(x,kernel_size=3,filters=64,use_act=True):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=1,use_bias=False,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x

def upsample(x_in,scale):
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=1,padding='same')(x_in)
    x = tf.keras.layers.Lambda(pixelshuffler(scale=2))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x

def residual_block(x_in, kernel_size=3,filters=64,stride=1,**kwargs):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,use_bias=False,padding='same')(x_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,use_bias=False,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x_in,x])

    return x

def generator_network(input_shape = (100,100,3),kernel_size = 3 , filters = 64, residual_blocks = 16,stride=1,**kwargs):
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=9,strides=1,use_bias=True,padding='same')(input)
    x = x_1 = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    #print(x.shape)

    for _ in range(residual_blocks):
        x = residual_block(x, kernel_size=3,filters=64,stride=1)

    x = tf.keras.layers.Conv2D(filters,kernel_size=kernel_size,strides=stride,use_bias=False,padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x,x_1])

    x = upsample(x,scale=2)
    x = upsample(x,scale=2)

    output = tf.keras.layers.Conv2D(filters=3,kernel_size=9,strides=1,padding='same')(x)

    # return output
    return tf.keras.Model(input,output)




gen_model = generator_network(kernel_size = 3 , filters = 64, residual_blocks = 16,stride=1)

#testing with random tensor
#random_tensor = numpy.random.rand(1,60,60,3)
#output_tensor = generator_network(input_shape = (60,60,3),kernel_size = 3 , filters = 64, residual_blocks = 16,stride=1)
#print(output_tensor.shape)


#discriminator network

def discriminator_block(x_in, filters=64,kernel_size=3,stride=1):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,use_bias=False)(x_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(x)

def discriminator_network():
    input = tf.keras.layers.Input(input_shape = (100,100,3))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1)(input)
    x = tf.keras.layers.LeakyReLU()(x)

    x = discriminator_block(x,stride=2)
    x = discriminator_block(x,filters=128,stride=1)

    x = discriminator_block(x, filters=128, stride=2)
    x = discriminator_block(x, filters=256, stride=1)

    x = discriminator_block(x, filters=256, stride=2)
    x = discriminator_block(x, filters=512, stride=1)

    x = discriminator_block(x, filters=512, stride=2)

    x = tf.keras.layers.Dense(units=1024)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    output = tf.keras.layers.Dense(units=1,activation='sigmoid')(x)

    return tf.keras.Model(input,output)

disc_model = discriminator_network()