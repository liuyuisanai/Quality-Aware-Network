#include <vector>
#include <cmath>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalization_layer.hpp"

namespace caffe {

	// template <typename Dtype>
	// void NormalizationLayer<Dtype>::LayerSetUp(
	//   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//   Layer<Dtype>::LayerSetUp(bottom, top);
	// }

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Layer<Dtype>::Reshape(bottom, top);
		top[0]->ReshapeLike(*bottom[0]);
		squared_.ReshapeLike(*bottom[0]);
		// top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		//     bottom[0]->height(), bottom[0]->width());
		// squared_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
		//   bottom[0]->height(), bottom[0]->width());
	}

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* squared_data = squared_.mutable_cpu_data();
		int n = bottom[0]->num();
		int d = bottom[0]->count() / n;
		Dtype normabs = caffe_cpu_asum<Dtype>(n, bottom_data);
		switch (this->layer_param_.normalization_param().norm()) {
		case NormalizationParameter_Norm_L2:
			//L2
			//const Dtype* bottom_data = bottom[0]->cpu_data();
			//Dtype* top_data = top[0]->mutable_cpu_data();
			//	Dtype* squared_data = squared_.mutable_cpu_data();
			//int n = bottom[0]->num();
			//int d = bottom[0]->count() / n;
			caffe_sqr<Dtype>(n*d, bottom_data, squared_data);
			for (int i = 0; i<n; ++i) {
				Dtype normsqr = caffe_cpu_asum<Dtype>(d, squared_data + i*d);
				caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data + i*d, top_data + i*d);
			}
			break;
		case NormalizationParameter_Norm_L1:
			//modify L1
			//const Dtype* bottom_data = bottom[0]->cpu_data();
			//Dtype* top_data = top[0]->mutable_cpu_data();
			//int n = bottom[0]->num();
			//int d = bottom[0]->count() / n;
			/*	for (int i = 0; i<n; ++i) {
			Dtype normabs = caffe_cpu_asum<Dtype>(d, bottom_data + i*d);
			LOG(INFO) << "debug info here---:num " << n<< ",d"
			<< d<< ",normabs" << normabs<< ","
			<< bottom[0]->count();
			caffe_cpu_scale<Dtype>(d, pow(normabs, -1), bottom_data + i*d, top_data + i*d);
			}*/
			//Dtype normabs = caffe_cpu_asum<Dtype>(n, bottom_data);
			caffe_cpu_scale<Dtype>(n, pow(normabs, -1), bottom_data, top_data);
			break;
		case NormalizationParameter_Norm_L3:
			for (int i = 0; i<n; ++i) {
				Dtype abs = caffe_cpu_asum<Dtype>(d, bottom_data+i*d);
				caffe_cpu_scale<Dtype>(d, pow(abs, -1), bottom_data + i*d, top_data + i*d);
			}
			break;
		}
	}

	template <typename Dtype>
	void NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* top_data = top[0]->cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* bottom_data_1 = bottom[0]->cpu_data();
		const int count = bottom[0]->count();
		int channels_ = bottom[0]->channels();
		int height_ = bottom[0]->height();
		int width_ = bottom[0]->width();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int n = top[0]->num();
		int d = top[0]->count() / n;
		Dtype normabs = caffe_cpu_asum<Dtype>(n, bottom_data);
		Dtype a=0;
		//vector<Dtype> data_sum_L2;
		//Dtype data_sum_L1 = 0;
		//for (int j = 0; j < n; j++){
		//	data_sum_L2[j] = 0;
		//	for (int t = 0; t < channels_*height_*width_; t++){
		//		
		//		data_sum_L2[j] += bottom_data_1[t];
		//		data_sum_L1 += bottom_data_1[t];
		//	}
		//	LOG(INFO) << "debug info here---:sum_L2 " << data_sum_L2[j] << ",sum L1" << data_sum_L1;
		//	bottom_data_1 += bottom[0]->offset(1);
		//}
		//	const Dtype* bottom_data = bottom[0]->cpu_data();
		switch (this->layer_param_.normalization_param().norm()) {
			//L2
		case NormalizationParameter_Norm_L2:
			//modify by songgl
			for (int i = 0; i<n; ++i) {

				//Dtype a = caffe_cpu_dot(d, bottom_data + i*d, bottom_data + i*d);
				//LOG(INFO) << "debug info here---:Dtype a: " << a;
				//LOG(INFO) << "debug info here---:top_diff[0]: " << top_diff[0];
				////caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d);
				//caffe_cpu_scale(d, Dtype(pow(a, -0.5)), top_data + i*d, bottom_diff + i*d);
				//caffe_cpu_scale(d, data_sum_L2[i], top_data + i*d, bottom_diff + i*d);
				//caffe_sub(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
				//caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff + i*d, bottom_diff + i*d);
				a = caffe_cpu_dot(d, top_data + i*d, top_diff + i*d);
				caffe_cpu_scale(d, a, top_data + i*d, bottom_diff + i*d);
				caffe_sub(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
				a = caffe_cpu_dot(d, bottom_data + i*d, bottom_data + i*d);
				caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff + i*d, bottom_diff + i*d);
			}
			break;
		case NormalizationParameter_Norm_L1:
			//L1 by songgl think bottom_data>0
			//caffe_cpu_scale(n, data_sum_L1, top_diff, bottom_diff);
			//caffe_cpu_scale(n, Dtype(pow(normabs, -1)), bottom_diff, bottom_diff);
			//caffe_sub(n, top_diff, bottom_diff, bottom_diff);
			//caffe_cpu_scale(n, Dtype(pow(normabs, -1)), bottom_diff, bottom_diff);
			a = caffe_cpu_dot(n, top_data, top_diff);
		//	LOG_IF(INFO, Caffe::root_solver())
		//		<< "Creating training net from net file: " << param_.net();
			caffe_set(n, a, bottom_diff);
			caffe_sub(n, top_diff, bottom_diff, bottom_diff);
			caffe_cpu_scale(n, Dtype(pow(normabs, -1)), bottom_diff, bottom_diff);
			break;

		case NormalizationParameter_Norm_L3:
			for (int i = 0; i < n; ++i) {
				Dtype ab = caffe_cpu_asum<Dtype>(d, bottom_data + i*d);
				a = caffe_cpu_dot(d, top_data+i*d, top_diff+i*d);
				caffe_set(d, a, bottom_diff+i*d);
				caffe_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);
				caffe_cpu_scale(d, Dtype(pow(ab, -1)), bottom_diff+i*d, bottom_diff+i*d);
			}
			break;
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(NormalizationLayer);
#endif

	INSTANTIATE_CLASS(NormalizationLayer);
	REGISTER_LAYER_CLASS(Normalization);

}  // namespace caffe