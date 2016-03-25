// Author: Mude Lin

#ifndef D3_POSE_LAYER_HPP
#define D3_POSE_LAYER_HPP

#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MultiLabelImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultiLabelImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiLabelImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiLabelImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  
  // top[0] data
  // top[1] label
  // top[2] mask [optional]
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, vector<float> > > lines_;
  int lines_id_;
  int label_num_;
  bool need_output_mask_;
  Dtype scale_; // inorder to avoid too big or too small label
};


template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param), diff_() {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  
  // if bottom blos == 1 will be a L1 norm term

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  /**
  * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
  * to both inputs -- override to return true and always allow force_backward.
  */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  bool has_weights_;
  bool l1_norm_;
};


/*
 * Reconstruct giving Dictionary and predicted weights
 */
template <typename Dtype>
class ReconstructLayer : public Layer<Dtype> {
 public:
  explicit ReconstructLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Reconstruct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  // bottom[0] the weight
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int dictionary_dim_;
  int atom_dim_;
  int M_, N_, K_;
  bool update_dictionary_;
};



/*
* Mean Per Joint Position Error
* With respect to Human3.6m: 
*/
template <typename Dtype>
class MPJPEEvaluationLayer : public LossLayer<Dtype> {
public:
  explicit MPJPEEvaluationLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param), diff_() {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MPJPEEvaluation"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  
  // if bottom blos == 1 will be a L1 norm term

  // virtual inline int MinBottomBlobs() const { return 1; }
  // virtual inline int MaxBottomBlobs() const { return 3; }

  /**
  * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
  * to both inputs -- override to return true and always allow force_backward.
  */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return false;
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    NOT_IMPLEMENTED;
  };
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  };
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  };
  void Project_to_origin(Blob<Dtype> *predict_blob);

  int sample_num_;
  int sample_ind_;
  Dtype error_;
  Blob<Dtype> diff_;
  Blob<Dtype> one_mulplier_;
  Blob<Dtype> frame_error_;

  Dtype scale_;
  std::ofstream save_file_;

};


}  // namespace caffe

#endif  // D3_POSE_LAYER_HPP
