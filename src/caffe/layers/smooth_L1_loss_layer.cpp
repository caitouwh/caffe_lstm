// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

#include "caffe/proto/caffe.pb.h"

#include "caffe/d3_pose_layers.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() == 3);
  l1_norm_ = (bottom.size() == 1);
  LossLayer<Dtype>::LayerSetUp(bottom,top);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  if ( l1_norm_ != true){
    LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  }else{
      vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
      top[0]->Reshape(loss_shape);
  }
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe