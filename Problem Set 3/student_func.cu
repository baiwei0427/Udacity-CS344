/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

#include <stdio.h>
#include <math.h>
#include <float.h>

// by default, we are always using the first device for GPU execution
#define DEVICE_ID 0

__global__ void reduce_min_kernel(const float *d_in, unsigned int in_size, float *d_out)
{
  extern __shared__ float s_data[];
    
  // thread ID inside the block
  unsigned int tid = threadIdx.x;
  // global ID across all blocks
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // # of elements in this block
  unsigned int block_size = blockDim.x;

  // if it is the last block
  if (blockIdx.x == gridDim.x - 1) {
    block_size -= gridDim.x * blockDim.x - in_size;
  }

  // copy elements from global memoery into per-block shared memory
  if (gid < in_size) {
    s_data[tid] = d_in[gid];
  } 
  __syncthreads();

  unsigned int s = (block_size + 1) << 1;
    
  while (s > 0) {
    if (tid < s && tid + s < block_size) {
      s_data[tid] = fminf(s_data[tid], s_data[tid + s]);
    }
    //ensure s keeps decreasing
    s = min((s + 1) << 1, s - 1);
    __syncthreads();
  }

  // write output from shared memory to global memory
  if (tid == 0) {
    d_out[blockIdx.x] = s_data[0];
  }
}

void reduce_min(const float *d_in, unsigned int in_size, float &h_out)
{
  float *d_inter, *d_out;
  // # of blocks, # of threads per block
  unsigned int blocks, threads_per_block;
  // # of elements in intermediate output
  unsigned int inter_size;

  cudaSetDevice(DEVICE_ID);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, DEVICE_ID);

  threads_per_block = prop.maxThreadsPerBlock;
  // ceil(in_size / threads_per_block) 
  blocks = (in_size + threads_per_block - 1) / threads_per_block;
  inter_size = blocks;

  // allocate GPU memory
	checkCudaErrors(cudaMalloc((void **)&d_inter, inter_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(float)));

  // phase 1 reduction
  reduce_min_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_in, in_size, d_inter);

  // phase 2 reduction
  reduce_min_kernel<<<1, blocks, blocks * sizeof(float)>>>(d_inter, inter_size, d_out);

  checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_inter));
  checkCudaErrors(cudaFree(d_out));
}

__global__ void reduce_max_kernel(const float *d_in, unsigned int in_size, float *d_out)
{
  extern __shared__ float s_data[];
    
  // thread ID inside the block
  unsigned int tid = threadIdx.x;
  // global ID across all blocks
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  // # of elements in this block
  unsigned int block_size = blockDim.x;

  // if it is the last block
  if (blockIdx.x == gridDim.x - 1) {
    block_size -= gridDim.x * blockDim.x - in_size;
  }

  // copy elements from global memoery into per-block shared memory
  if (gid < in_size) {
    s_data[tid] = d_in[gid];
  } 
  __syncthreads();

  unsigned int s = (block_size + 1) << 1;
    
  while (s > 0) {
    if (tid < s && tid + s < block_size) {
      s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
    }
    // ensure that s keeps decreasing
    s = min((s + 1) << 1, s - 1);
    __syncthreads();
  }

  // write output from shared memory to global memory
  if (tid == 0) {
    d_out[blockIdx.x] = s_data[0];
  }
}

void reduce_max(const float *d_in, unsigned int in_size, float &h_out)
{
  float *d_inter, *d_out;
  // # of blocks, # of threads per block
  unsigned int blocks, threads_per_block;
  // # of elements in intermediate output
  unsigned int inter_size;

  cudaSetDevice(DEVICE_ID);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, DEVICE_ID);

  threads_per_block = prop.maxThreadsPerBlock;
  // ceil(in_size / threads_per_block) 
  blocks = (in_size + threads_per_block - 1) / threads_per_block;
  inter_size = blocks;

  // allocate GPU memory
	checkCudaErrors(cudaMalloc((void **)&d_inter, inter_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(float)));

  // phase 1 reduction
  reduce_max_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_in, in_size, d_inter);

  // phase 2 reduction
  reduce_max_kernel<<<1, blocks, blocks * sizeof(float)>>>(d_inter, inter_size, d_out);

  checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_inter));
  checkCudaErrors(cudaFree(d_out));
}

__global__ void histogram_kernel(const float *d_in,
                                 unsigned int num_in, 
                                 float min_val, 
                                 float max_val,
                                 unsigned int *bins,
                                 unsigned int num_bins)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = blockDim.x * gridDim.x;
  unsigned int bin;

  while (i < num_in) {
    // calculate bin index
    bin = (d_in[i] - min_val) / (max_val - min_val) * num_bins;
    atomicAdd(&bins[bin], 1);
    i += stride;     
  }   
}

__global__ void scan_kernel(unsigned int *d_in, unsigned int size)
{
  extern __shared__ unsigned int sdata[];

  // ID inside the block
  unsigned int tid = threadIdx.x;
  // global ID across blocks
  unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;

  // copy input from global memory to shared memory
  sdata[tid] = d_in[gid];
  __syncthreads();

  for (int offset = 1; offset < size; offset *= 2) {
    unsigned int tmp = sdata[tid];
    if (tid >= offset) {
      tmp += sdata[tid - offset];
    }
    
    __syncthreads();
    sdata[tid] = tmp;
    __syncthreads(); 
  }

  // copy output from shared memory to global memory
  d_in[gid] = sdata[tid];  
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    
  // 1) find the minimum and maximum value in the input logLuminance channel store in min_logLum and max_logLum
  unsigned int size = numRows * numCols;
    
  reduce_max(d_logLuminance, size, max_logLum);
  reduce_min(d_logLuminance, size, min_logLum);

  printf("Max: %f\n", max_logLum);
  printf("Min: %f\n", min_logLum);
  

  unsigned int threads = 1024;
  // ceil(size / threads)
  unsigned int blocks = (size + threads - 1) / threads;
  histogram_kernel<<<blocks, threads>>>(d_logLuminance, size, min_logLum, max_logLum, d_cdf, numBins);

  scan_kernel<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_cdf, numBins);  
}