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

|Manufacturer|Model     |Architecture|Memory    |CUDA Cores|Tensor Cores|F32 TFLOPS|F16 TFLOPS|Retail|
|------------|----------|------------|----------|----------|------------|----------|----------|------|
|NVIDIA      |Titan V   |Volta       |12GB HBM2 |5120      |640         |15        |110*      |$2999 |
|NVIDIA      |1080 Ti   |Pascal      |11GB GDDR5|3584      |0           |11        |N/A       |$699  |        


# CUDA / CuDNN
* CUDA 9.0.176
* CuDNN 7.0.0.5


# Networks
* VGG16
* Resnet152
* Densenet161
* ...


# Benchmark Results

Results are based on running the models with images of size 224 x 224 x 3,
with a batch size of 16.
"Eval" shows the duration for a single forward pass averaged over 10 passes.
"Train" shows the duration for a pair of forward and backward passes averaged over 10 runs.
Titan V gets a significant speed up when going to half precision by utilizing its Tensor cores, while
1080 Ti gets a small speed up with half precision computation.

|Model  |Framework   |Precision   |VGG16 eval   |VGG16 train|Resnet152 eval   |Resnet152 train|Densenet161 eval   |Densenet161 train|
|-------|-------------|------------|-------------|-----------|-----------------|---------------|-------------------|-----------------|
|Titan V|PyTorch 0.3.0|32-bit      |31.2ms       |110.7ms    |49.6ms           |178.6ms        |48.9ms             |178.2ms          |
|Titan V|PyTorch 0.3.0|16-bit      |15.4ms       |75.5ms     |26.6ms           |117.1ms        |26.1ms             |117.4ms          |
|1080 Ti|PyTorch 0.3.0|32-bit      |38.9ms       |133.3ms    |58.5ms           |206.7ms        |57.7ms             |207.9ms          |    
|1080 Ti|PyTorch 0.3.0|16-bit      |33.9ms       |119.4ms    |46.9ms           |194.6ms        |48.8ms             |194.9ms          |    