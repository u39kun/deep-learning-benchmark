import numpy as np
from time import time
from caffe2.proto import caffe2_pb2
from caffe2.python.modeling.initializers import Initializer, pFP16Initializer
from caffe2.python import core, model_helper, workspace, brew

#core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

class caffe2_base:

    def __init__(self, model_name, precision, image_shape, batch_size):
        self.float_type = np.float32 if precision == 'fp32' else np.float16
        self.input = np.random.rand(batch_size, 3, image_shape[0], image_shape[1]).astype(self.float_type)

    def eval(self):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            t1 = time()
            workspace.FeedBlob("data", self.input)
            workspace.RunNet(self.forward_net.Proto().name)
            out = workspace.FetchBlob("softmax")
            t2 = time()
            return t2 - t1

    def train(self):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            t1 = time()
            workspace.FeedBlob("data", self.input)
            workspace.RunNet(self.model.net.Proto().name)
            out = workspace.FetchBlob("softmax")
            t2 = time()
            return t2 - t1

class vgg16(caffe2_base):

  def __init__(self, precision, image_shape, batch_size):
    caffe2_base.__init__(self, 'vgg16', precision, image_shape, batch_size)
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
      model = model_helper.ModelHelper(name="vgg16", init_params=True)
      softmax = self.vgg16_model(model, "data", precision)
      if precision == 'fp16':
        softmax = model.net.HalfToFloat(softmax, softmax + "_fp32")
      forward_net = core.Net(model.net.Proto())
      # bogus loss function to force backprop
      loss = model.net.Sum(softmax, 'loss')
      model.AddGradientOperators([loss])
      workspace.RunNetOnce(model.param_init_net)

      data = np.zeros((16, 3, 224, 224), dtype=self.float_type)
      workspace.FeedBlob("data", data)
      workspace.CreateNet(model.net)
      workspace.CreateNet(forward_net)
      self.model = model
      self.forward_net = forward_net

  def vgg16_model(self, model, data, precision):
    initializer = Initializer if precision == 'fp32' else pFP16Initializer
    with brew.arg_scope([brew.conv, brew.fc],
                        WeightInitializer=initializer,
                        BiasInitializer=initializer,
                        enable_tensor_core=True):
      conv1_1 = brew.conv(model, data, 'conv1_1', dim_in=3, dim_out=64, kernel=3, pad=1)
      conv1_1 = brew.relu(model, conv1_1, conv1_1)
      conv1_2 = brew.conv(model, conv1_1, 'conv1_2', dim_in=64, dim_out=64, kernel=3, pad=1)
      conv1_2 = brew.relu(model, conv1_2, conv1_2)
      pool1 = brew.max_pool(model, conv1_2, 'pool1', kernel=2, stride=2)
    
      conv2_1 = brew.conv(model, pool1, 'conv2_1', dim_in=64, dim_out=128, kernel=3, pad=1)
      conv2_1 = brew.relu(model, conv2_1, conv2_1)
      conv2_2 = brew.conv(model, conv2_1, 'conv2_2', dim_in=128, dim_out=128, kernel=3, pad=1)
      conv2_2 = brew.relu(model, conv2_2, conv2_2)
      pool2 = brew.max_pool(model, conv2_2, 'pool2', kernel=2, stride=2)
    
      conv3_1 = brew.conv(model, pool2, 'conv3_1', dim_in=128, dim_out=256, kernel=3, pad=1)
      conv3_1 = brew.relu(model, conv3_1, conv3_1)
      conv3_2 = brew.conv(model, conv3_1, 'conv3_2', dim_in=256, dim_out=256, kernel=3, pad=1)
      conv3_2 = brew.relu(model, conv3_2, conv3_2)
      conv3_3 = brew.conv(model, conv3_2, 'conv3_3', dim_in=256, dim_out=256, kernel=3, pad=1)
      conv3_3 = brew.relu(model, conv3_3, conv3_3)
      pool3 = brew.max_pool(model, conv3_3, 'pool3', kernel=2, stride=2)
   
      conv4_1 = brew.conv(model, pool3, 'conv4_1', dim_in=256, dim_out=512, kernel=3, pad=1)
      conv4_1 = brew.relu(model, conv4_1, conv4_1)
      conv4_2 = brew.conv(model, conv4_1, 'conv4_2', dim_in=512, dim_out=512, kernel=3, pad=1)
      conv4_2 = brew.relu(model, conv4_2, conv4_2)
      conv4_3 = brew.conv(model, conv4_2, 'conv4_3', dim_in=512, dim_out=512, kernel=3, pad=1)
      conv4_3 = brew.relu(model, conv4_3, conv4_3)
      pool4 = brew.max_pool(model, conv4_3, 'pool4', kernel=2, stride=2)
 
      conv5_1 = brew.conv(model, pool4, 'conv5_1', dim_in=512, dim_out=512, kernel=3, pad=1)
      conv5_1 = brew.relu(model, conv5_1, conv5_1)
      conv5_2 = brew.conv(model, conv5_1, 'conv5_2', dim_in=512, dim_out=512, kernel=3, pad=1)
      conv5_2 = brew.relu(model, conv5_2, conv5_2)
      conv5_3 = brew.conv(model, conv5_2, 'conv5_3', dim_in=512, dim_out=512, kernel=3, pad=1)
      conv5_3 = brew.relu(model, conv5_3, conv5_3)
      pool5 = brew.max_pool(model, conv5_3, 'pool5', kernel=2, stride=2)
 
      fc6 = brew.fc(model, pool5, 'fc6', dim_in=25088, dim_out=4096)
      fc6 = brew.relu(model, fc6, fc6)
      fc7 = brew.fc(model, fc6, 'fc7', dim_in=4096, dim_out=4096)
      fc7 = brew.relu(model, fc7, fc7)
      pred = brew.fc(model, fc7, 'pred', 4096, 1000)
      softmax = brew.softmax(model, pred, 'softmax')
      return softmax

