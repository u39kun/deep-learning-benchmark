from time import time

import numpy as np

from frameworks.tensorflow.tf_models import resnet_model, vgg_model
from frameworks.tensorflow.tf_models import convnet_builder
import tensorflow as tf

class tensorflow_base:

    def __init__(self, model_creator, precision, image_shape, batch_size):
        phase_train = False
        data_format = 'NCHW'
        data_type = tf.float32 if precision == 'fp32' else tf.float16
        image_shape = [16, 3, image_shape[0], image_shape[1]]
        nclass = 1000
        use_tf_layers = False

        tf.reset_default_graph()

        images = tf.constant(np.random.rand(*image_shape), dtype=data_type)

        network = convnet_builder.ConvNetBuilder(
            images, 3, phase_train, use_tf_layers,
            data_format, data_type, data_type)

        model = model_creator()
        model.add_inference(network)
        self.logits = network.affine(nclass, activation='linear')
        # bogus loss to force backprop
        self.loss = tf.reduce_sum(self.logits)
        self.grad = tf.gradients(self.loss, tf.trainable_variables())
        self.initializer = tf.global_variables_initializer()

    def eval(self, num_iterations, num_warmups):
        durations = []
        with tf.Session() as sess:
            sess.run(self.initializer)
            for i in range(num_iterations + num_warmups):
                t1 = time()
                sess.run(self.logits)
                t2 = time()
                print(t2 - t1)
                if i >= num_warmups:
                    durations.append(t2 - t1)
        return durations

    def train(self, num_iterations, num_warmups):
        durations = []
        with tf.Session() as sess:
            sess.run(self.initializer)
            for i in range(num_iterations + num_warmups):
                t1 = time()
                sess.run(self.grad)
                t2 = time()
                print(t2 - t1)
                if i >= num_warmups:
                    durations.append(t2 - t1)
        return durations

class vgg16(tensorflow_base):

  def __init__(self, precision, image_shape, batch_size):
    tensorflow_base.__init__(self, vgg_model.Vgg16Model, precision, image_shape, batch_size)


class resnet152(tensorflow_base):

  def __init__(self, precision, image_shape, batch_size):
      tensorflow_base.__init__(self, resnet_model.create_resnet152_model, precision, image_shape, batch_size)


