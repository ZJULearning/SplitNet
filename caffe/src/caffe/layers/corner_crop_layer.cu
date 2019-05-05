#include <vector>

#include "caffe/layers/corner_crop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void crop(const int num, const int channel, 
        const int height, const int width, 
        const int x, const int y, const int crop_h_, const int crop_w_,
        const Dtype* in_data, Dtype* out_data, const bool forward) {
  const int count = num*channel*height*width;
  int top_index;
  CUDA_KERNEL_LOOP(index, count) {
    const int w = index%width;
    const int h = (index/width)%height;
    const int c = (index/width/height)%channel;
    const int n = index/width/height/channel;
    if (y<=h && h<y+crop_h_ && x<=w && w<x+crop_w_) {
      top_index = ((n*channel+c)*crop_h_+h-y)*crop_w_+w-x;
      if (forward) {
         out_data[top_index] = in_data[index];
      } else {
        out_data[index] += in_data[top_index];
      }
    }
  }
}

template <typename Dtype>
__global__ void average(const int num, const int channel,
        const int height, const int width,
        Dtype* out_data, const int* count_crop) {
  const int count = num*channel*height*width;
  CUDA_KERNEL_LOOP(index, count) {
    const int w = index%width;
    const int h = (index/width)%height;
    if (count_crop[h*width+w]) {
      out_data[index] /= count_crop[h*width+w];
    }
  }
}

template <typename Dtype>
void CornerCropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int channel = bottom[0]->channels();
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  int x, y;
  const int* points = points_.cpu_data();
  for (int i = 0; i < crop_num_; ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    x = points[i*2+0];
    y = points[i*2+1];
    crop<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        num, channel, height, width, x, y, crop_h_, crop_w_,
        bottom_data, top_data, true);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void CornerCropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int channel = bottom[0]->channels();
  const int num = bottom[0]->num();
  const int count = bottom[0]->count();
  int x, y;
  const int* points = points_.cpu_data();
  for (int i = 0; i < crop_num_; ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    x = points[i*2+0];
    y = points[i*2+1];
    crop<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        num, channel, height, width, x, y, crop_h_, crop_w_,
        top_diff, bottom_diff, false);
    CUDA_POST_KERNEL_CHECK;
  }
  average<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          num, channel, height, width, bottom_diff, counts_.gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(CornerCropLayer);

}  // namespace caffe
