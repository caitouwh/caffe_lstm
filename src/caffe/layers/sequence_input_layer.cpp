#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

// #include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/sequence_layers.hpp"
#include <boost/algorithm/string/join.hpp>
#include "boost/algorithm/string.hpp"

namespace caffe {

template <typename Dtype>
SequenceInputLayer<Dtype>::~SequenceInputLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void SequenceInputLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  // const int min_height = this->layer_param_.image_data_param().min_height();
  // const int min_width  = this->layer_param_.image_data_param().min_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // CHECK((min_height == 0 && min_width == 0) ||
  //     (min_height > 0 && min_width > 0)) << "Current implementation requires "
  //     "min_height and min_width to be set at the same time.";
  // Read the file with filenames and labels 
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string label_str;
  std::string previous_prefix;
  int current_offset = -1;
  while (infile >> filename >> label_str) {
    ++current_offset;

    vector<float> labels;
    std::vector<std::string> label_info;
    boost::trim(label_str);                                                                                                                             
    boost::split(label_info, label_str, boost::is_any_of(","));
    
    num_label_ = label_info.size();

    for(int label_ind = 0; label_ind < label_info.size(); ++ label_ind){
      Dtype label = Dtype(std::atof(label_info[label_ind].c_str()));
      labels.push_back(label);   
    }
    lines_.push_back(std::make_pair(filename, labels));

    // split sequence according to the filename
    std::vector<std::string> path_info;
    boost::trim(filename);                                                                                                                             
    boost::split(path_info, filename, boost::is_any_of("/"));
    path_info = std::vector<std::string>(path_info.begin(), path_info.end() - 1);
    std::string path_prefix = boost::algorithm::join(path_info, "/");
    if(path_prefix != previous_prefix){
      previous_prefix = path_prefix;
      sequence_offset_.push_back(current_offset);
    }
  }

  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  // const int channels = cv_img.channels();
  // const int height = cv_img.rows;
  // const int width = cv_img.cols;
  // image
  // const int crop_size = this->layer_param_.transform_param().crop_size();
  
  num_buffer_ = this->layer_param_.sequence_input_param().num_buffer();
  num_frame_ = this->layer_param_.sequence_input_param().num_frame();
  frame_step_ = this->layer_param_.sequence_input_param().frame_step();
  multiply_scale_ = this->layer_param_.sequence_input_param().multiply_scale();


  const int batch_size = num_frame_ * num_buffer_;

  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  // if (crop_size > 0) {
  //   top[0]->Reshape(batch_size, channels, crop_size, crop_size);
  //   this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
  //   this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  // } else {
  //   top[0]->Reshape(batch_size, channels, height, width);
  //   this->prefetch_data_.Reshape(batch_size, channels, height, width);
  //   this->transformed_data_.Reshape(1, channels, height, width);
  // }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
      
  // label
  vector<int> label_shape(2, batch_size);
  
  label_shape[1] = num_label_ + 1;  // one for clip marker

  top[1]->Reshape(label_shape);

    
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void SequenceInputLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  
  // const int batch_size = num_frame_ * num_buffer_;

  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  // const int min_height = image_data_param.min_height();
  // const int min_width  = image_data_param.min_width();
  // const int crop_size = this->layer_param_.transform_param().crop_size();
  
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();



  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
 // datum scales
  const int lines_size = lines_.size();

  vector<int> random_offset(this->num_buffer_,0);
  // random on sequence
  for(int i =0; i < this->num_buffer_; ++i) {
    caffe::rng_t* prefetch_rng =
    static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    const int seletected_index = (*prefetch_rng)() % sequence_offset_.size();
    random_offset[i] = sequence_offset_[seletected_index];

    // random on frames
    const int frames_count = seletected_index == sequence_offset_.size() - 1 ? 
                                  lines_size - sequence_offset_[seletected_index]
                                  : sequence_offset_[seletected_index + 1]  - sequence_offset_[seletected_index];
    CHECK_GT(frames_count, num_frame_ * frame_step_);
    const int frame_offset =(*prefetch_rng)() % (frames_count - num_frame_ * frame_step_);
    random_offset[i] = frame_offset;

  }
    
  for(int frame_id =0; frame_id < this->num_frame_; ++frame_id){

    for(int buffer_id = 0; buffer_id < this->num_buffer_; ++ buffer_id){
      const int item_id = frame_id * buffer_id + buffer_id;
      DLOG(INFO) << "Item id :" << item_id ;
      
      lines_id_ = random_offset[buffer_id];
      if(frame_id == 0){
        prefetch_label[batch->label_.offset(item_id, this->num_label_)] = Dtype(0);
      }else{
        prefetch_label[batch->label_.offset(item_id, this->num_label_)] = Dtype(1);
      }

      // get a blob
      timer.Start();
      CHECK_GT(lines_size, lines_id_);
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();

      for(int i =0;i < this->num_label_ ; ++i){
        prefetch_label[batch->label_.offset(item_id, i)] = 
                    Dtype(lines_[lines_id_].second[i]) * this->multiply_scale_;
      }
      // go to the next iter
      
      // if (lines_id_ >= lines_size) {
      //   // We have reached the end. Restart from the first.
      //   DLOG(INFO) << "Restarting data prefetching from start.";
      //   lines_id_ = 0;
      //   if (this->layer_param_.image_data_param().shuffle()) {
      //     ShuffleImages();
      //   }
      // }
      random_offset[buffer_id] +=  this->frame_step_;
    }
    
    
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(SequenceInputLayer);
REGISTER_LAYER_CLASS(SequenceInput);

}  // namespace caffe
