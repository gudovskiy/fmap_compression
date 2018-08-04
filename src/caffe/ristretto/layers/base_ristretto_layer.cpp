#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include "ristretto/base_ristretto_layer.hpp"

namespace caffe {

template <typename Dtype>
BaseRistrettoLayer<Dtype>::BaseRistrettoLayer() {
  // Initialize random number generator
  srand(time(NULL));
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeWeights_cpu(
      vector<shared_ptr<Blob<Dtype> > > weights_quantized, const int rounding,
      const bool bias_term) {
  Dtype* weight = weights_quantized[0]->mutable_cpu_data();
  const int cnt_weight = weights_quantized[0]->count();
  switch (precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2FixedPoint_cpu(weight, cnt_weight, bw_params_, rounding, fl_params_);
    if (bias_term) {
      Trim2FixedPoint_cpu(weights_quantized[1]->mutable_cpu_data(),
          weights_quantized[1]->count(), bw_params_ + bw_layer_out_, rounding, bw_params_ + fl_layer_out_);
    }
    break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << precision_;
    break;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerInputs_cpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_cpu(
      Dtype* data, const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <>
void BaseRistrettoLayer<float>::Trim2FixedPoint_cpu(float* data, const int cnt,
      const int bit_width, const int rounding, const int fl) {
  for (int index = 0; index < cnt; ++index) {
    // round data
    data[index] *= powf(2, fl);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = roundf(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floorf(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    // saturate data
    float max_data = (powf(2, bit_width - 1) - 1.0);
    float min_data = -powf(2, bit_width - 1);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // back to float
    data[index] *= powf(2, -fl);
    }
}
template <>
void BaseRistrettoLayer<double>::Trim2FixedPoint_cpu(double* data, const int cnt,
      const int bit_width, const int rounding, const int fl) {
  for (int index = 0; index < cnt; ++index) {
    // round data
    data[index] *= pow(2, fl);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    // saturate data
    double max_data = (pow(2, bit_width - 1) - 1.0);
    double min_data = -pow(2, bit_width - 1);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // back to float
    data[index] *= pow(2, -fl);
    }
}

template <typename Dtype>
double BaseRistrettoLayer<Dtype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}

template BaseRistrettoLayer<double>::BaseRistrettoLayer();
template BaseRistrettoLayer<float>::BaseRistrettoLayer();
template void BaseRistrettoLayer<double>::QuantizeWeights_cpu(
    vector<shared_ptr<Blob<double> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseRistrettoLayer<float>::QuantizeWeights_cpu(
    vector<shared_ptr<Blob<float> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseRistrettoLayer<double>::QuantizeLayerInputs_cpu(double* data,
    const int count);
template void BaseRistrettoLayer<float>::QuantizeLayerInputs_cpu(float* data,
    const int count);
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_cpu(double* data,
    const int count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_cpu(float* data,
    const int count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_cpu(double* data,
    const int cnt, const int bit_width, const int rounding, const int fl);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_cpu(float* data,
    const int cnt, const int bit_width, const int rounding, const int fl);
//
template double BaseRistrettoLayer<float>::RandUniform_cpu();

}  // namespace caffe
