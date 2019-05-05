#include <algorithm>
#include <vector>

#include "caffe/layers/corner_crop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CornerCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CornerCropParameter crop_param = this->layer_param().corner_crop_param();
    if (crop_param.has_crop_size()) {
        crop_h_ = crop_param.crop_size();
        crop_w_ = crop_param.crop_size();
    } else {
        crop_h_ = crop_param.crop_h();
        crop_w_ = crop_param.crop_w();
    }
    h_offset_ = (bottom[0]->height() - crop_h_) / 2;
    w_offset_ = (bottom[0]->width() - crop_w_) / 2;
    CHECK_GT(bottom[0]->height(), crop_h_);
    CHECK_GT(bottom[0]->width(), crop_w_);
    CHECK_EQ((bottom[0]->height() - crop_h_)%2, 0);
    CHECK_EQ((bottom[0]->width() - crop_w_)%2, 0);

    crop_num_ = crop_param.position_size();
    CHECK(top.size() == crop_num_ || (top.size() == 5 && crop_num_ == 0));
    crop_num_ = top.size();
    vector<int> point_shape = bottom[0]->shape();
    point_shape[0] = 1;
    point_shape[1] = crop_num_;
    point_shape[2] = 2;
    point_shape[3] = 1;
    points_.Reshape(point_shape);
    int x = 0, y = 0;
    int* points = points_.mutable_cpu_data();
    for (int i = 0; i < crop_num_; ++i) {
        std::string string_position = crop_param.position(i);
        if (string_position == "center") {
            x = w_offset_;
            y = h_offset_;
        } else if (string_position == "upperleft") {
            x = 0;
            y = 0;
        } else if (string_position == "upperright") {
            x = 2*w_offset_;
            y = 0;
        } else if (string_position == "lowerleft") {
            x = 0;
            y = 2*h_offset_;
        } else if (string_position == "lowerright") {
            x = 2*w_offset_;
            y = 2*h_offset_;
        } else {
            LOG(FATAL) << "Unknown position: " << string_position << ".\n"
                << "Please select from upperleft/upperright/lowerleft/lowerright.";
        }
        points[i*2+0] = x;
        points[i*2+1] = y;
    }
    vector<int> count_shape = bottom[0]->shape();
    count_shape[0] = 1;
    count_shape[1] = 1;
    counts_.Reshape(count_shape);
    caffe_set(counts_.count(), 0, counts_.mutable_cpu_data());
    for (int h = 0; h < bottom[0]->height(); ++h) {
        for (int w = 0; w < bottom[0]->width(); ++w) {
            for (int k = 0; k < crop_num_; ++k) {
                x = points[k*2+0];
                y = points[k*2+1];
                if (y<=h && h<y+crop_h_ && x<=w && w<x+crop_w_) {
                    counts_.mutable_cpu_data()[counts_.offset(1,1,h,w)] += 1;
                }
            }
        }
    }
}

template <typename Dtype>
void CornerCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[2] = crop_h_;
  top_shape[3] = crop_w_;
  for (int i = 0; i < crop_num_; ++i) {
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void CornerCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  int bottom_index, top_index;
  int x, y;
  const int* points = points_.cpu_data();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          bottom_index = bottom[0]->offset(i,c,h,w);
          for (int k = 0; k < crop_num_; ++k) {
            Dtype* top_data = top[k]->mutable_cpu_data();
            x = points[k*2+0];
            y = points[k*2+1];
            if (y<=h && h<y+crop_h_ && x<=w && w<x+crop_w_) {
              top_index = top[k]->offset(i,c,h-y,w-x);
              top_data[top_index] = bottom_data[bottom_index];
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void CornerCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  int bottom_index, top_index;
  int x, y;
  const int* points = points_.cpu_data();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          bottom_index = bottom[0]->offset(i,c,h,w);
          for (int k = 0; k < crop_num_; ++k) {
            const Dtype* top_diff = top[k]->cpu_diff();
            x = points[k*2+0];
            y = points[k*2+1];
            if (y<=h && h<y+crop_h_ && x<=w && w<x+crop_w_) {
              top_index = top[k]->offset(i,c,h-y,w-x);
              bottom_diff[bottom_index] += top_diff[top_index];
            }
          }
        }
      }
    }
  }
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          bottom_index = bottom[0]->offset(i,c,h,w);
          int count_crop = counts_.cpu_data()[counts_.offset(1,1,h,w)];
          if (count_crop) {
            bottom_diff[bottom_index] /= count_crop;
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CornerCropLayer);
#endif

INSTANTIATE_CLASS(CornerCropLayer);
REGISTER_LAYER_CLASS(CornerCrop);

}  // namespace caffe
