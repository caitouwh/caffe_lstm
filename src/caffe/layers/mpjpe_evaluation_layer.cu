#include <vector>
#include "caffe/util/math_functions.hpp"
#include "caffe/d3_pose_layers.hpp"
#include <math.h>

namespace caffe {

/*
 *  Recalculate point's location according to their parent node
*/
template <typename Dtype>
void MPJPEEvaluationLayer<Dtype>::Project_to_origin(Blob<Dtype> *predict_blob){
  const int num = predict_blob -> num();
  const int num_skel = predict_blob -> channels() / 3; // for 3d pose
  int parent_node[] = {0, 0,1,2, 0,4,5, 0,7,8,9, 8,11,12, 8,14,15 }; 
  Dtype *blob_data = predict_blob -> mutable_cpu_data();
  
  for(int n = 0; n < num; ++n) {
    // except the root node
    for(int c = 1; c < num_skel; ++c){
      blob_data[predict_blob->offset(n,3*c)] += blob_data[predict_blob->offset(n,3*parent_node[c])];
      blob_data[predict_blob->offset(n,3*c+1)] += blob_data[predict_blob->offset(n,3*parent_node[c] + 1)];
      blob_data[predict_blob->offset(n,3*c+2)] += blob_data[predict_blob->offset(n,3*parent_node[c] + 2)];
       
    }
  }
} 

template <typename Dtype>
void MPJPEEvaluationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  sample_ind_ += bottom[0] -> num();
  int count = bottom[0]->count();
  

  Project_to_origin(bottom[0]);
  caffe_gpu_scal<Dtype>(count, Dtype(100.0), bottom[0]->mutable_gpu_data());
  caffe_gpu_scal<Dtype>(count, Dtype(100.0), bottom[1]->mutable_gpu_data());


  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  caffe_gpu_powx<Dtype>(count, diff_.mutable_gpu_data(), 2, diff_.mutable_gpu_data());

  int num_skeleton = bottom[0] ->channels() / 3; // 3 for x,y,z
  const Dtype *cpu_diff_data = diff_.cpu_data();
  
  Dtype current_batch_error = 0.0;
  for(int i = 0; i < bottom[0] -> num(); ++i) {
    Dtype sqrt_error = 0.0;
    for(int j = 0; j < num_skeleton; ++j){
       sqrt_error += std::sqrt(cpu_diff_data[diff_.offset(i,3*j)] + 
                            cpu_diff_data[diff_.offset(i,3*j + 1)] + 
                            cpu_diff_data[diff_.offset(i,3*j + 2)]  
                            );
    }
    sqrt_error /= Dtype(num_skeleton);
    
    current_batch_error += sqrt_error;
  }

  top[0]->mutable_cpu_data()[0] = current_batch_error / bottom[0]->num();  
  error_ += current_batch_error;
  
  int offset = sample_ind_ - bottom[0] ->num();

  const Dtype *bottom_cpu_data = bottom[0] -> cpu_data();
  for(int ind = 0; ind < bottom[0] -> num(); ++ind) {
    save_file_ << ind + offset + 1;
    for(int c = 0; c < bottom[0] -> channels(); ++c){
      save_file_ << "," << bottom_cpu_data[bottom[0] -> offset(ind,c)];
    }
    save_file_ << std::endl;
  }

  
  if (sample_ind_ >= sample_num_) {
    LOG(INFO) << "Final MPJPE is : " << error_ / sample_ind_ 
              << " ("<< error_ << "/" << sample_ind_ << ").";
     sample_ind_ = 0;
     error_ = 0;
  }

}


INSTANTIATE_LAYER_GPU_FUNCS(MPJPEEvaluationLayer);

}  // namespace caffe
