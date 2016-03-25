#ifndef GLOBAL_VALUE_HPP_
#define GLOBAL_VALUE_HPP_

#include <string>
#include <caffe.hpp>

namespace caffe {

class GlobalValue {
public:
	float GetScale(){
		return scale_;
	}
protected:
	float scale_; // scale datalayer and for predict layer

};

} // end of caffe
#endif