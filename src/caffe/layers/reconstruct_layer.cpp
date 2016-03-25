#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/d3_pose_layers.hpp"

#include "boost/algorithm/string.hpp"
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>

namespace caffe {

template <typename Dtype>
void ReconstructLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  dictionary_dim_ = this->layer_param_.reconstruct_param().dictionary_dim();
  atom_dim_ = this->layer_param_.reconstruct_param().atom_dim();

  update_dictionary_ = this->layer_param_.reconstruct_param().update_dictionary();

  M_ = bottom[0] -> num();
  N_ = atom_dim_;
  K_ = dictionary_dim_;
  // dictionary is a K_* N_ dictionary, K_ is the dictionary dim;

  const string dictionary_src_  = this->layer_param_.reconstruct_param().dictionary_src();

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the dictionary 
    vector<int> weight_shape(2);
    weight_shape[0] = atom_dim_;
    weight_shape[1] = dictionary_dim_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    if(dictionary_src_.size() > 0){
      LOG(INFO) << "Initializing dictionary weight from " << dictionary_src_;
      // init from learned dictionary
      Dtype *dictionary_data =  this->blobs_[0] -> mutable_cpu_data();
      std::ifstream infile(dictionary_src_.c_str());
      string line;
      // for example from 0 to 96
      int atom_elem_ind = 0;
      while (getline(infile,line))
      {
          std::vector<float> weight_vec;

          std::vector<std::string> weight_info;
          boost::trim(line);                                                                                                                             
          boost::split(weight_info, line, boost::is_any_of(","));
          
          CHECK_EQ(weight_info.size() , dictionary_dim_) << "dictionary dim doesnot match";

          for(int dict_elem_ind = 0; dict_elem_ind < weight_info.size(); ++ dict_elem_ind){
            // for example from 0 to 1000
            dictionary_data[this->blobs_[0] -> offset(atom_elem_ind,dict_elem_ind)] 
                              = Dtype(std::atof(weight_info[dict_elem_ind].c_str()));
            
          }
          atom_elem_ind += 1;
      }
      CHECK_EQ(atom_elem_ind, atom_dim_) << "atom size doesnot match";


    }else{
      // random init
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.reconstruct_param().weight_filler()));
      weight_filler->Fill(this->blobs_[0].get());  
    }
    
  }  // parameter initialization
  
  // Must set this or will not propagate down
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ReconstructLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  vector<int> top_shape = vector<int>(2,0);
  top_shape[0] = bottom[0] -> num();
  top_shape[1] = atom_dim_;
  top[0]->Reshape(top_shape);
  
}

template <typename Dtype>
void ReconstructLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  // if (bias_term_) {
  //   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
  //       bias_multiplier_.cpu_data(),
  //       this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  // }

}

template <typename Dtype>
void ReconstructLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (this->param_propagate_down_[0] && update_dictionary_ ) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }

  // if (bias_term_ && this->param_propagate_down_[1]) {
  //   const Dtype* top_diff = top[0]->cpu_diff();
  //   // Gradient with respect to bias
  //   caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
  //       bias_multiplier_.cpu_data(), (Dtype)1.,
  //       this->blobs_[1]->mutable_cpu_diff());
  // }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReconstructLayer);
#endif

INSTANTIATE_CLASS(ReconstructLayer);
REGISTER_LAYER_CLASS(Reconstruct);

}  // namespace caffe
