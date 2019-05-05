#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if(argc < 7) {
      LOG(ERROR) << "Usage: ./extract model prototxt blob fea.txt label.txt batchnum [gpu] \n";
      return 1;
  }
  
  int gpu_num = 0;
  if(argc == 8) {
      gpu_num = atoi(argv[7]);
  }
  LOG(ERROR) << "Using Device_id=" << gpu_num;
  Caffe::SetDevice(gpu_num);
  Caffe::set_mode(Caffe::GPU);
  
  std::string pretrained_binary_proto(argv[1]); 
  std::string deploy_proto(argv[2]);
  LOG(ERROR) << "Init net with: " << deploy_proto;
  boost::shared_ptr<Net<Dtype> > net(new Net<Dtype>(deploy_proto, caffe::TEST));
  LOG(ERROR) << "Init weight with: " << pretrained_binary_proto;
  net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string blob_name(argv[3]);

  int num_mini_batches = atoi(argv[6]);
  
  std::ofstream f_fea(argv[4]);
  std::ofstream f_label;
  bool label_flag = false;
  if(strcmp(argv[5], "no")) {
    label_flag = true;
    f_label.open(argv[5]);
  }

  LOG(ERROR) << "Writing feature to: " << argv[4];

  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
      net->Forward();
      const boost::shared_ptr<Blob<Dtype> > fea = net->blob_by_name(blob_name);
      int batch_size = fea->num();
      int dim_features = fea->count() / batch_size;
      const Dtype *fea_data;
      for(int n = 0; n < batch_size; n++) {
        fea_data = fea->cpu_data() + fea->offset(n);
        for (int d = 0; d < dim_features; ++d) {
            f_fea << " " << fea_data[d];
        }
        f_fea << std::endl;
      }

      if (label_flag) {
        //read labels
        const boost::shared_ptr<Blob<Dtype> > label = net->blob_by_name("label");
        const Dtype* label_data = label->cpu_data();
        int num = label->num();
        for (int i = 0; i < num; ++i){
            unsigned int label = static_cast<unsigned int>(label_data[i]);
            f_label << label << "\n";
         }
      }
      if (batch_index % 100 == 0) {
          LOG(ERROR) << batch_index << "/" << num_mini_batches << " done.";
      }
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

  f_fea.close();
  f_label.close();

  LOG(ERROR)<< "Successfully extracted the feature: " << argv[4];
  return 0;
}
