# Benchmark on Deep Learning Frameworks and GPUs

Performance of popular deep learning frameworks and GPUs are compared, including the effect of adjusting the floating point precision (the new Volta architecture allows performance boost by utilizing half/mixed-precision calculations.)

# Deep Learning Frameworks

Note: Docker images available from NVIDIA GPU Cloud were used so as to make benchmarking controlled and repeatable by anyone.

* PyTorch 0.3.0
  * `docker pull nvcr.io/nvidia/pytorch:17.12`


* Tensorflow 1.4.0 (coming soon)
  * `docker pull nvcr.io/nvidia/tensorflow:17.12`


* MXNet 1.0.0 (coming soon)
  * `docker pull nvcr.io/nvidia/mxnet:17.12`


* Caffe2 (coming soon)
  * `docker pull nvcr.io/nvidia/caffe2:17.12`


* CNTK (coming soon)
  * `docker pull nvcr.io/nvidia/cntk:17.12`


# GPUs

|Manufacturer|Model     |Architecture|Memory    |CUDA Cores|Tensor Cores|F32 TFLOPS|F16 TFLOPS|Retail|EC2  |
|------------|----------|------------|----------|----------|------------|----------|----------|------|-----|
|NVIDIA      |V100      |Volta       |16GB HBM2 |5120      |640         |15.7      |125       |      |$3.06/hr|
|NVIDIA      |Titan V   |Volta       |12GB HBM2 |5120      |640         |15        |110*      |$2999 |N/A  |
|NVIDIA      |1080 Ti   |Pascal      |11GB GDDR5|3584      |0           |11        |N/A       |$699  |N/A  |        


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

The results are based on running the models with images of size 224 x 224 x 3
with a batch size of 16.
"Eval" shows the duration for a single forward pass averaged over 10 passes.
"Train" shows the duration for a pair of forward and backward passes averaged over 10 runs.

Titan V gets a significant speed up when going to half precision by utilizing its Tensor cores, while
1080 Ti gets a small speed up with half precision computation.
Similarly, the numbers from V100 on an Amazon p3 instance is shown.  It is faster than Titan V and the speed up when going to half-precision is similar to that of Titan V.

## Titan V
|Framework    |Precision   |VGG16 eval   |VGG16 train|Resnet152 eval   |Resnet152 train|Densenet161 eval   |Densenet161 train|
|-------------|------------|-------------|-----------|-----------------|---------------|-------------------|-----------------|
|PyTorch 0.3.0|32-bit      |31.2ms       |110.7ms    |49.6ms           |178.6ms        |56.4ms             |181.1ms          |
|PyTorch 0.3.0|16-bit      |15.4ms       |75.5ms     |26.6ms           |117.1ms        |37.6ms             |123.7ms          |

## 1080 Ti
|Framework    |Precision   |VGG16 eval   |VGG16 train|Resnet152 eval   |Resnet152 train|Densenet161 eval   |Densenet161 train|
|-------------|------------|-------------|-----------|-----------------|---------------|-------------------|-----------------|
|PyTorch 0.3.0|32-bit      |38.9ms       |133.3ms    |58.5ms           |206.7ms        |63.7ms             |209.0ms          |    
|PyTorch 0.3.0|16-bit      |33.9ms       |119.4ms    |46.9ms           |194.6ms        |50.0ms             |188.7ms          |    

## V100 (Amazon p3, CUDA 9.0.176, CuDNN 7.0.0.3)
|Framework    |Precision   |VGG16 eval   |VGG16 train|Resnet152 eval   |Resnet152 train|Densenet161 eval   |Densenet161 train|
|-------------|------------|-------------|-----------|-----------------|---------------|-------------------|-----------------|
|PyTorch 0.3.0|32-bit      |26.2ms       |83.5ms    |38.7ms           |136.5ms        |48.3ms             |142.5ms          |
|PyTorch 0.3.0|16-bit      |12.6ms       |58.8ms     |21.7ms           |92.9ms        |35.7ms             |102.3ms          |
