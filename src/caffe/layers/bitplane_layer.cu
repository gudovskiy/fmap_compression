#include <algorithm>
#include <vector>

#include "caffe/layers/bitplane_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void forward_i2b(const int n, const Dtype* in, Dtype* out, const int b, const int fl) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype otmp = in[index] * powf(2, fl);
    unsigned utmp = __float2uint_rz(otmp);
    unsigned btmp = (utmp >> b) & 0x00000001;
    out[index] = __uint2float_rz(btmp);
  }
}

/*template <typename Dtype>
__global__ void forward_b2i(const int n, const Dtype* in, Dtype* out, const Dtype scale) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] += in[index] * scale;
  }
}*/

template <typename Dtype>
void BitplaneLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  //
  const int num      = bottom[0]->num(); // batches
  const int channels = bottom[0]->channels();
  const int height   = bottom[0]->height();
  const int width    = bottom[0]->width();
  const int spatial  = height*width;
  const int fmap     = channels*spatial;
  const int count    = num*fmap;
  //
  const bool dir = this->layer_param_.bitplane_param().direction();
  const int bw   = this->layer_param_.bitplane_param().bw_layer();
  const int fl   = this->layer_param_.bitplane_param().fl_layer();
  //
  const int fmapI    = fmap/bw;
  const int countI   = count/bw;
  // set to zero
  if (dir != true) {
    caffe_gpu_set(countI, Dtype(0), top_data);
  }
  //
  for (int n = 0; n < num; ++n) {
    for (int b = 0; b < bw; ++b) {
      if (dir == true) { // int to bits
        forward_i2b<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(fmap), CAFFE_CUDA_NUM_THREADS>>>(
            fmap, bottom_data + n*fmap, top_data + fmap*(b + n*bw), b, fl);
      } else { // bits to int
        const Dtype scale = powf(2, b-fl); // forward scaler
        caffe_gpu_axpy(fmapI, scale, bottom_data + fmapI*(b + n*bw), top_data + n*fmapI);
        /*forward_b2i<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
            <<<CAFFE_GET_BLOCKS(fmapI), CAFFE_CUDA_NUM_THREADS>>>(
            fmapI, bottom_data + fmapI*(b + n*bw), top_data + n*fmapI, scale);*/
      }
    }
  }
  //
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void backward_i2b(const int n, const Dtype* in, Dtype* out, const Dtype* sw) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (sw[index] > Dtype(0));
  }
}

/*template <typename Dtype>
__global__ void backward_b2i(const int n, const Dtype* in, Dtype* out, const Dtype scale) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] += in[index] * scale;
  }
}*/

template <typename Dtype>
void BitplaneLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    //
    const int num      = bottom[0]->num(); // batches
    const int channels = bottom[0]->channels();
    const int height   = bottom[0]->height();
    const int width    = bottom[0]->width();
    const int spatial  = height*width;
    const int fmap     = channels*spatial;
    const int count    = num*fmap;
    //
    const bool dir = this->layer_param_.bitplane_param().direction();
    const int bw   = this->layer_param_.bitplane_param().bw_layer();
    const int fl   = this->layer_param_.bitplane_param().fl_layer();
    //
    const int fmapI    = fmap/bw;
    //const int countI   = count/bw;
    const Dtype scale = 2.0 * powf(bw, -1); // simplified gradient scaler
    //
    if (dir == true) { // set to zero
      caffe_gpu_set(count, Dtype(0), bottom_diff);
    }
    //
    for (int n = 0; n < num; ++n) {
      for (int b = 0; b < bw; ++b) {
        if (dir != true) { // int to bits
          backward_i2b<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
              <<<CAFFE_GET_BLOCKS(fmapI), CAFFE_CUDA_NUM_THREADS>>>(
              fmapI, top_diff + n*fmapI, bottom_diff + fmapI*(b + n*bw), bottom_data + fmapI*(b + n*bw));
        } else { // bits to int
          caffe_gpu_axpy(fmap, scale, top_diff + fmap*(b + n*bw), bottom_diff + n*fmap);
          /*backward_b2i<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
              <<<CAFFE_GET_BLOCKS(fmap), CAFFE_CUDA_NUM_THREADS>>>(
              fmap, top_diff + fmap*(b + n*bw), bottom_diff + n*fmap, scale);*/
        }
      }
    }
    //
    CUDA_POST_KERNEL_CHECK;
    // << " count: " << count << " bottom_data: "
    //     << (unsigned long)bottom_data
    //     << " top_data: " << (unsigned long)top_data
    //     << " blocks: " << CAFFE_GET_BLOCKS(count)
    //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BitplaneLayer);

}  // namespace caffe
