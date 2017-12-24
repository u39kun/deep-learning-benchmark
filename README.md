# Benchmark on Deep Learning Frameworks and GPUs

Performance of popular deep learning frameworks and GPUs are compared, including the effect of adjusting the floating point precision (the new Volta architecture allows performance boost by utilizing half/mixed-precision calculations.)

# Deep Learning Frameworks

Note: Docker images available from NVIDIA GPU Cloud were used so as to make benchmarking controlled and repeatable by anyone.

* PyTorch 0.3.0
  * `docker pull nvcr.io/nvidia/pytorch:17.12`

* Caffe2 0.8.1
  * `docker pull nvcr.io/nvidia/caffe2:17.12`

* Tensorflow 1.4.0 (coming next)
  * `docker pull nvcr.io/nvidia/tensorflow:17.12`

* MXNet 1.0.0 (anyone interested?)
  * `docker pull nvcr.io/nvidia/mxnet:17.12`

* CNTK (anyone interested?)
  * `docker pull nvcr.io/nvidia/cntk:17.12`


# GPUs

|Model     |Architecture|Memory    |CUDA Cores|Tensor Cores|F32 TFLOPS|F16 TFLOPS|Retail|Cloud  |
|----------|------------|----------|----------|------------|----------|----------|------|-----|
|Tesla V100|Volta       |16GB HBM2 |5120      |640         |15.7      |125       |      |$3.06/hr (p3.2xlarge)|
|Titan V   |Volta       |12GB HBM2 |5120      |640         |15        |110*      |$2999 |N/A  |
|1080 Ti   |Pascal      |11GB GDDR5|3584      |0           |11        |N/A       |$699  |N/A  |        


# CUDA / CuDNN
* CUDA 9.0.176
* CuDNN 7.0.0.5
Except where noted.


# Networks
* VGG16
* Resnet152
* Densenet161
* Any others you might be interested in?

# Benchmark Results

## PyTorch 0.3.0

The results are based on running the models with images of size 224 x 224 x 3
with a batch size of 16.
"Eval" shows the duration for a single forward pass averaged over 20 passes.
"Train" shows the duration for a pair of forward and backward passes averaged over 20 runs.
In both scenarios, 20 runs of warm up is performed and those are not counted towards the measured numbers.

Titan V gets a significant speed up when going to half precision by utilizing its Tensor cores, while
1080 Ti gets a small speed up with half precision computation.
Similarly, the numbers from V100 on an Amazon p3 instance is shown.  It is faster than Titan V and the speed up when going to half-precision is similar to that of Titan V.

### Titan V
| Precision   | vgg16 eval   | vgg16 train   | resnet152 eval   | resnet152 train   | densenet161 eval   | densenet161 train   |
|:------------|:-------------|:--------------|:-----------------|:------------------|:-------------------|:--------------------|
| 32-bit      | 31.3ms       | 108.8ms       | 48.9ms           | 180.2ms           | 52.4ms             | 174.1ms             |
| 16-bit      | 14.7ms       | 74.1ms        | 26.1ms           | 115.9ms           | 32.2ms             | 118.9ms             |

### 1080 Ti
| Precision   | vgg16 eval   | vgg16 train   | resnet152 eval   | resnet152 train   | densenet161 eval   | densenet161 train   |
|:------------|:-------------|:--------------|:-----------------|:------------------|:-------------------|:--------------------|
| 32-bit      | 39.3ms       | 131.9ms       | 57.8ms           | 206.4ms           | 62.9ms             | 211.9ms             |
| 16-bit      | 33.5ms       | 117.6ms       | 46.9ms           | 193.5ms           | 50.1ms             | 191.0ms             |

### V100 (Amazon p3, CUDA 9.0.176, CuDNN 7.0.0.3)
|Precision   |VGG16 eval|VGG16 train|Resnet152 eval   |Resnet152 train|Densenet161 eval   |Densenet161 train|
|------------|----------|-----------|-----------------|---------------|-------------------|-----------------|
|32-bit      |26.2ms    |83.5ms     |38.7ms           |136.5ms        |48.3ms             |142.5ms          |
|16-bit      |12.6ms    |58.8ms     |21.7ms           |92.9ms         |35.7ms             |102.3ms          |


## TensorFlow 1.4.0

### Titan V
| Precision   | vgg16 eval   | vgg16 train   | resnet152 eval   | resnet152 train   | densenet161 eval   | densenet161 train   |
|:------------|:-------------|:--------------|:-----------------|:------------------|:-------------------|:--------------------|
| 32-bit      | 32.0ms       | 195.3ms       | 50.6ms           | 274.4ms           |                    |                     |
| 16-bit      | 16.3ms       | 86.0ms        | 28.8ms           | 197.9ms           |                    |                     |

### 1080 Ti
| Precision   | vgg16 eval   | vgg16 train   | resnet152 eval   | resnet152 train   | densenet161 eval   | densenet161 train   |
|:------------|:-------------|:--------------|:-----------------|:------------------|:-------------------|:--------------------|
| 32-bit      | 43.4ms       | 131.3ms       | 69.6ms           | 300.6ms           |                    |                     |
| 16-bit      | 38.6ms       | 121.1ms       | 53.9ms           | 257.0ms           |                    |                     |


## Caffe2 0.8.1

### Titan V
| Precision   | vgg16 eval   | vgg16 train   | resnet152 eval   | resnet152 train   | densenet161 eval   | densenet161 train   |
|:------------|:-------------|:--------------|:-----------------|:------------------|:-------------------|:--------------------|
| 32-bit      | 57.5ms       | 185.4ms       | 74.4ms           | 214.1ms           |                    |                     |
| 16-bit      | 41.6ms       | 156.1ms       | 56.9ms           | 172.7ms           |                    |                     |                    

### 1080 Ti
|Precision   |VGG16 eval|VGG16 train|Resnet152 eval   |Resnet152 train|Densenet161 eval|Densenet161 train|
|------------|----------|-----------|-----------------|---------------|----------------|-----------------|
| 32-bit     | 47.0ms   | 158.9ms   | 77.9ms          | 223.9ms       |                |                 |
| 16-bit     | 40.1ms   | 137.8ms   | 61.7ms          | 184.1ms       |                |                 |


# Contributors

* Yusaku Sako
* Bartosz Ludwiczuk (thank you for supplying the V100 numbers)
