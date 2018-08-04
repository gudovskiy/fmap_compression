#include <algorithm>
#include <vector>

#include "caffe/layers/bitplane_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BitplaneLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const BitplaneParameter& bitplane_param = this->layer_param_.bitplane_param();
  CHECK_EQ(bottom.size(), 1) << "Box Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1)    << "Box Layer takes exactly one blob as output.";
  //CHECK(!(bitplane_param.has_direction() && bitplane_param.has_bw_layer() && bitplane_param.has_fl_layer()))
  //    << "Bitplane parameters are missing.";
}

template <typename Dtype>
void BitplaneLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const bool dir = this->layer_param_.bitplane_param().direction();
  const int bw   = this->layer_param_.bitplane_param().bw_layer();
  //
  int input_dim = bottom[0]->num_axes();
  vector<int> new_shape(bottom[0]->shape());
  if (dir) {
    new_shape[1] *= bw; // increase by bw
  } else {
    new_shape[1] /= bw; // decrease by bw
  }
  //
  top[0]->Reshape(new_shape);
  int top_count = bottom[0]->shape(0);
  for (int i = 1; i < input_dim; ++i) {
    top_count *= bottom[0]->shape(i);
  }
  //
  if (dir) {
    top_count *= bw;
  } else {
    top_count /= bw;
  }
  CHECK_EQ(top_count, top[0]->count());
}

template <typename Dtype>
void i2b(const int n, const Dtype* in, Dtype* out, const int b, const int fl) {
  for (int index = 0; index < n; ++index) {
    Dtype otmp = in[index] * powf(2, fl);
    unsigned utmp = (unsigned)(otmp);
    unsigned btmp = (utmp >> b) & 0x00000001;
    out[index] = (float)(btmp);
  }
}

template <typename Dtype>
void b2i(const int n, const Dtype* in, Dtype* out, const int b, const int fl) {
  for (int index = 0; index < n; ++index) {
    out[index] += in[index] * powf(2, b-fl);
  }
}

template <typename Dtype>
void BitplaneLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
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
    caffe_set(countI, Dtype(0), top_data);
  }
  //
  for (int n = 0; n < num; ++n) {
    for (int b = 0; b < bw; ++b) {
      if (dir == true) { // int to bits
        i2b(fmap, bottom_data + n*fmap, top_data + fmap*(b + n*bw), b, fl);
      } else { // bits to int
        b2i(fmapI, bottom_data + fmapI*(b + n*bw), top_data + n*fmapI, b, fl);
      }
    }
  }
  //
}

template <typename Dtype>
void BitplaneLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
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
    //
    if (dir == true) { // set to zero
      caffe_set(count, Dtype(0), bottom_diff);
    }
    //
    for (int n = 0; n < num; ++n) {
      for (int b = 0; b < bw; ++b) {
        if (dir != true) { // int to bits
          i2b(fmapI, top_diff + n*fmapI, bottom_diff + fmapI*(b + n*bw), b, fl);
        } else { // bits to int
          b2i(fmap, top_diff + fmap*(b + n*bw), bottom_diff + n*fmap, b, fl);
        }
      }
    }
    //
  }
}

#ifdef CPU_ONLY
STUB_GPU(BitplaneLayer);
#endif

INSTANTIATE_CLASS(BitplaneLayer);

}  // namespace caffe
