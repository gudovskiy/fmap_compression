#include "ristretto/base_ristretto_layer.hpp"
#include "ristretto/base_ristretto_layer.cuh"

namespace caffe {

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeWeights_gpu(
      vector<shared_ptr<Blob<Dtype> > > weights_quantized, const int rounding,
      const bool bias_term) {
  Dtype* weight = weights_quantized[0]->mutable_gpu_data();
  const int cnt_weight = weights_quantized[0]->count();
  switch (precision_) {
  case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
    Trim2FixedPoint_gpu(weight, cnt_weight, bw_params_, rounding, fl_params_);
    if (bias_term) {
      Trim2FixedPoint_gpu(weights_quantized[1]->mutable_gpu_data(),
          weights_quantized[1]->count(), bw_params_ + bw_layer_out_, rounding, bw_params_ + fl_layer_out_);
    }
    break;
  default:
    LOG(FATAL) << "Unknown trimming mode: " << precision_;
    break;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerInputs_gpu(
    Dtype* data, const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::QuantizeLayerOutputs_gpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
__global__ void Trim2FixedPoint_kernel(Dtype* data, const int cnt,
      const int bit_width, const int rounding, const int fl) {
	CUDA_KERNEL_LOOP(index, cnt) {
    // round data
    data[index] *= powf(2, fl);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = rintf(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      //data[index] += RandUniform_device(index);
      data[index] = __float2int_rd(data[index] + RandUniform_device(index)); // ??? somehow this is working ???
      break;
    default:
      break;
    }
    // saturate data
    Dtype max_data = (powf(2, bit_width - 1) - 1.0);
    Dtype min_data = -powf(2, bit_width - 1);
    data[index] = fmaxf(fminf(data[index], max_data), min_data);
    // back to float
    data[index] *= powf(2, -fl);
	}
}

template <typename Dtype>
void BaseRistrettoLayer<Dtype>::Trim2FixedPoint_gpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, const int fl) {
  Trim2FixedPoint_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bit_width, rounding, fl);
}

// Explicit instantiations
template void BaseRistrettoLayer<double>::QuantizeWeights_gpu(
    vector<shared_ptr<Blob<double> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseRistrettoLayer<float>::QuantizeWeights_gpu(
    vector<shared_ptr<Blob<float> > > weights_quantized, const int rounding,
    const bool bias_term);
template void BaseRistrettoLayer<double>::QuantizeLayerInputs_gpu(double* data,
    const int count);
template void BaseRistrettoLayer<float>::QuantizeLayerInputs_gpu(float* data,
    const int count);
template void BaseRistrettoLayer<double>::QuantizeLayerOutputs_gpu(
    double* top_data, const int top_count);
template void BaseRistrettoLayer<float>::QuantizeLayerOutputs_gpu(
    float* top_data, const int top_count);
template void BaseRistrettoLayer<double>::Trim2FixedPoint_gpu(double* data,
    const int cnt, const int bit_width, const int rounding, const int fl);
template void BaseRistrettoLayer<float>::Trim2FixedPoint_gpu(float* data,
    const int cnt, const int bit_width, const int rounding, const int fl);

}  // namespace caffe
