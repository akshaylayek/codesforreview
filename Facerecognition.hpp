#ifndef FACERECOGNITION_HPP
#define FACERECOGNITION_HPP
#define DLIB_JPEG_SUPPORT
#define DLIB_PNG_SUPPORT
#define DLIB_GIF_SUPPORT
#define DLIB_USE_BLAS
#define DLIB_USE_LAPACK
#define DLIB_THREAD_POOL_TIMEOUT
#include <iostream>
#include <chrono>
#include <cmath>
#include <regex>
#include <exception>
#include <thread>
#include <vector>
#include <algorithm>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/signals2.hpp>
#include <opencv2/opencv.hpp>
#include "cxxopts.hpp"
enum SourceType {DEVICE,VIDEO_FILE };
class Facerecognition{
      	public:
      		 /// Constructor
		Facerecognition();
		/// Destructor
		~Facerecognition();
		void initializeStream();
		void	setSource(std::string camera_id);
		std::string	getSource() const;
		void	setSourceType(int camera_id);
		int	getSourceType() const;

		void setLandmarkFile(std::string) ;
		std::string getLandmarkFile() const;
         	void setSkipRate(int) ;
                void setforsave(int);
                int getforsave() const;
		int getSkipRate() const;
		void setnewflag(int);
 		int getnewflag() const;
		void setnewdouble(double);
		double getnewdouble()const;
		void setnewstring(std::string);
  		std::string getnewstring()const ;


	private:
       		static std::string source_;
		static int source_type_;
		static std::string source_path_;
		static std::string source_stem_;
		static std::string source_ext_;
		static std::string landmark_file_;
		int skip_rate_ = {5};
                int for_flag_={6};
		int new_flag_={4};
		double new_double_={4.5};
	        static std::string new_string_;
		int frame_counter_= {0};
     		std::unique_ptr<cv::VideoCapture> cap;
 };
std::string Facerecognition::source_ = {"0"};
std::string Facerecognition::landmark_file_ = {"shape_predictor_68_face_landmarks.dat"};
std::string Facerecognition::new_string_={"0"};
int Facerecognition::source_type_ = {SourceType::DEVICE};
Facerecognition::Facerecognition(){
  	std::cerr << "opening default ctor" << std::endl;
	try{
		cap.reset(new cv::VideoCapture());
	}catch(std::exception& e){
 		std::cerr << "caught exception" << e.what() << std::endl;
     	}
}
Facerecognition::~Facerecognition(){
	std::cerr << "calling default dtor" << std::endl;
}
void	Facerecognition::setSourceType(int source_type){
	source_type_ = source_type;
}
int	Facerecognition::getSourceType() const{
	return source_type_;
}

void	Facerecognition::setSource(std::string source){
	source_ = source;
}
std::string	Facerecognition::getSource() const{
	return source_;
}
void	Facerecognition::setLandmarkFile(std::string landmark_file) {
	landmark_file_ = landmark_file;
}
std::string	Facerecognition::getLandmarkFile() const{
	return landmark_file_;
}

void Facerecognition::initializeStream(){
 	if(source_type_ == SourceType::DEVICE ){
		cap->open( boost::lexical_cast<int>(source_));
	}else if(source_type_ == SourceType::VIDEO_FILE ){
 		std::cerr<< "I got a  video file or url" << std::endl;
                cap->open( source_);
  	 }

}

void Facerecognition::setSkipRate(int skip_rate) {
	skip_rate_ = skip_rate;
}
int Facerecognition::getSkipRate() const{
	return skip_rate_;
}
void Facerecognition::setnewflag(int new_flag) {
	new_flag_ = new_flag;
}
int Facerecognition::getnewflag() const{
        return new_flag_;
}
void Facerecognition::setnewdouble(double new_double) {
         new_double_ = new_double;
 }
double Facerecognition::getnewdouble() const{
         return new_double_;
}
void Facerecognition::setnewstring(std::string new_string) {
         new_string_ = new_string;
 }
std::string Facerecognition::getnewstring() const{
         return new_string_;
 }
 void Facerecognition::setforsave(int new_flag) {
          for_flag_ = new_flag;
  }
  int Facerecognition::getforsave() const{
          return for_flag_;
  }

#endif
