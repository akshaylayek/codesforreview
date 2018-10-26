// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
   This example program shows how to find frontal human faces in an image and
   estimate their pose.  The pose takes the form of 68 landmarks.  These are
   points on the face such as the corners of the mouth, along the eyebrows, on
   the eyes, and so forth.
   This face detector is made using the classic Histogram of Oriented
   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
   and sliding window detection scheme.  The pose estimator was created by
   using dlib's implementation of the paper:
   One Millisecond Face Alignment with an Ensemble of Regression Trees by
   Vahid Kazemi and Josephine Sullivan, CVPR 2014
   and was trained on the iBUG 300-W face landmark dataset.
   Also, note that you can train your own models using dlib's machine learning
   tools.  See train_shape_predictor_ex.cpp to see an example.
   Finally, note that the face detector is fastest when compiled with at least
   SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
   chip then you should enable at least SSE2 instructions.  If you are using
   cmake to compile this program you can enable them by using one of the
   following commands when you create the build project:
   cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
   cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
   cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
   This will set the appropriate compiler options for GCC, clang, Visual
   Studio, or the Intel compiler.  If you are using another compiler then you
   need to consult your compiler's manual to determine how to enable these
   instructions.  Note that AVX is the fastest but requires a CPU from at least
   2011.  SSE4 is the next fastest and is supported by most current machines.
   */
//#include <bits/stdc++.h>
#include<dlib/opencv.h>
#include<opencv2/opencv.hpp>
#include<sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <string.h>
#include "Facerecognition.hpp"
#include "boost/filesystem.hpp"
//#include "serialise_1.hpp"
using namespace boost::filesystem;
//using namespace cv;
//using namespace dlib;
//using namespace std::chrono_literals;
using namespace std;
void parse_command_line(int argc, char **argv,Facerecognition&);
// ----------------------------------------------------------------------------------------
int main(int argc, char** argv) {
   	int skip_counter = 5;
	int counter = 0;
	Facerecognition face_recognition;
	//First you parse the command line
	parse_command_line(argc,argv,face_recognition);
        std::cout<<"the new flag is:"<<face_recognition.getnewflag()<<std::endl;
	std::cout<<"the new double is is:"<<face_recognition.getnewdouble()<<std::endl;
	std::cout<<"the new string is:"<<face_recognition.getnewstring()<<std::endl;
	std::string dnn_metric_file = "shape_predictor_68_face_landmarks.dat";
	// as a command line argument.
	dlib::shape_predictor sp;
	std::cout<< "opening shape predictor landmarks!" << std::endl;
	dlib::deserialize(dnn_metric_file) >> sp;
	std::cout<< "done!" << std::endl;
	bool continue_flag = true;
	unsigned camera_id = 0;
	std::unique_ptr<cv::VideoCapture> cap(new cv::VideoCapture());

	if(face_recognition.getSourceType() == SourceType::DEVICE ){
         	   	 cap->open( boost::lexical_cast<int>(face_recognition.getSource()));

			 //cap.reset(  new cv::VideoCapture(boost::lexical_cast<int>(face_recognition.getSource())));
   	}else if(face_recognition.getSourceType() == SourceType::VIDEO_FILE ){
		std::cerr<< "I got a video file or url" << std::endl;
 cap.reset(  new cv::VideoCapture(face_recognition.getSource()));
             	 }
        if(!cap->isOpened())  // check if we succeeded
		return -1;
	try
	{
                cv ::namedWindow("head counting",1);
		cv::Mat frame;
		dlib::array2d<dlib::rgb_pixel> img;

		auto detector = dlib::get_frontal_face_detector();
		std::vector<dlib::full_object_detection> shapes;

		dlib::image_window win(img, "Original Image"), win_faces(img, "overlayed Image");
 		win_faces.set_title("s");
		int m=0;
		while(continue_flag)
		{
    			*cap >> frame; // get a new frame from camera


			if(++counter%face_recognition.getSkipRate() == 0){
   				//cvtColor(frame, edges, COLOR_BGR2GRAY);
				//GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
				// each face in an image.
				// And we also need a shape_predictor.  This is the tool that will predict face
				// landmark positions given an image and face bounding box.  Here we are just
				// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
				//std::cout << "deserialization  done" << std::endl;
                                m++;

				dlib::array2d<dlib::rgb_pixel> img;

				//dlib::load_image(img, argv[i]);
				// Make the image larger so we can detect small faces.
				dlib::assign_image(img, dlib::cv_image<dlib::rgb_pixel>(frame));

				dlib::pyramid_up(img);
				// Now tell the face detector to give us a list of bounding boxes
				// around all the faces in the image.
				std::vector<dlib::rectangle> dets = detector(img);
				int num_faces =  dets.size();
				cout << "Number of faces detected: " << dets.size() << endl;
				// Now we will go ask the shape_predictor to tell us the pose of
				// each face we detected.
#pragma omp parallel for
				for (unsigned long j = 0; j < num_faces; ++j)
				{
       	 	 		 	dlib::full_object_detection shape = sp(img, dets[j]);
					//cout << "number of parts: "<< shape.num_parts() << endl;
					//cout << "pixel position of first part:  " << shape.part(0) << endl;
					//cout << "pixel position of second part: " << shape.part(1) << endl;
					// You get the idea, you can get all the face part locations if
					// you want them.  Here we just store them in shapes so we can
					// put them on the screen.
      	  				shapes.push_back(shape);
				}
				// Now let's view our face poses on the screen.
				//win.clear_overlay();
				//win.set_image(img);
				//win.add_overlay(render_face_detections(shapes));
				// We can also extract copies of each face that are cropped, rotated upright,
				// and scaled to a standard size as shown here:
				if(dets.size()>0){
                                        dlib::array<dlib::array2d<dlib::rgb_pixel> > face_chips;
					extract_image_chips(img, get_face_chip_details(shapes), face_chips);
					win.clear_overlay();
					win.set_image(img);

             	     				win_faces.clear_overlay();
win_faces.set_title(std::to_string(boost::lexical_cast<int>(num_faces)));
             	  		 		win_faces.set_image(tile_images(face_chips));
 					shapes.clear();
                                        if(face_recognition.getforsave())
                                        {
                                         std::string s = std::to_string(m);
                            std::string name="frame";
                            std::string result = name + s;
                            const char * c = result.c_str();
                           // mkdir(c,0777);
                           std::stringstream ss;
                           ss<<"./"<<face_recognition.getnewstring()<<"/"<<c;
                           std::string path1=ss.str();
                           ss.str("");
                           boost::filesystem::path rootPath (path1);
                           boost::system::error_code returnedError;

                           boost::filesystem::create_directories( rootPath, returnedError );
                           if ( returnedError )
                         std::cout<<"directory creation not successful"<<std::endl;


                            char b='_';
                           // std::string s1=std::to_string(b);
                             std::string result1=result+b;
					for(unsigned long j=0;j<dets.size();++j)
					{
					// rename(face_chips(j),/Users/PINWHEEL/softwares/inhouse/image_processing/machine_learning/head_count/c);

					// stringstream ss;
                                        // string name = "cropped_";
                                        // string type = ".jpg"
				        // ss<<name<<(ct + 1)<<type;
					// string filename = ss.str();
					// ss.str("");
					// imwrite(filename, img_cropped);
                                                std::string result2=s+b+std::to_string(j)+".png";
					      //	std::string type=".jpg";
						ss<<"./"<<face_recognition.getnewstring()<<"/"<<c<<"/"<<result2;
						std:: string fullPath = ss.str();
                                                ss.str("");
						cv::Mat img1 = dlib::toMat(face_chips[j]);
						cv::imwrite(fullPath,img1);
					}
                                        }
 				}
 			}
 	 	}
               // std::ostream& out1;
               // seialize(images,out1);

	 }
	catch (std::exception& e)
	{
        	cout << "\nexception thrown!" << endl;
                cout << e.what() << endl;
	}
	getchar();
    	return 0;
}
// ----------------------------------------------------------------------------------------
void parse_command_line(int argc, char **argv,Facerecognition& face_recognition){
                	cxxopts::Options options(argv[0], " - example command line options");
	options.positional_help("[optional arguments]").show_positional_help();
	options.allow_unrecognised_options().add_options()
           	 	("s,source", "input source", cxxopts::value <std::string>())
		("k,skip", "skip frames", cxxopts::value<std::string>())
		("t,new_flag", "third parameter", cxxopts::value<std::string>())
                ("u,new_double", "fourth parameter", cxxopts::value<std::string>())
		("v,new_string", "fifth parameter", cxxopts::value<std::string>())
                 ("w,for_save", "sixth parameter", cxxopts::value<std::string>())
                ("i,input", "Input", cxxopts::value<std::string>())
		("o,output", "Output file", cxxopts::value<std::string>()
		 ->default_value("a.out")->implicit_value("b.def"), "BIN")
		("positional","Positional arguments: these are the arguments that are entered "
		 "without an option", cxxopts::value<std::vector<std::string>>())
      		 ;
	//options.add_options("Group") ("c,compile", "compile") ("d,drop", "drop",cxxopts::value<std::vector<std::string>>());
	options.parse_positional({"input", "output", "positional"});
	auto result = options.parse(argc, argv);
	if (result.count("help"))
	{
       		std::cerr << options.help({"", "Group"}) << std::endl;
      		exit(0);
	}
	if (result.count("s"))
	{
          	 std::cerr << result["source"].as<std::string>() << std::endl;
		std::string temp_str = result["source"].as<std::string>();
		std::cerr << "found type s" << std::endl;
		try{
         		    	std::cerr <<"Looks like device" << std::endl;
		        int source=boost::lexical_cast<int>(temp_str);
			int source_type = SourceType::DEVICE;
			face_recognition.setSource(boost::lexical_cast<std::string>(source));
    		  	face_recognition.setSourceType(source_type);
                }catch(std::exception& e){
           	 	 	std::cerr <<"failed to cast to int..trying string" << std::endl;
			std::cout << "Either a video file or url" << std::endl;
			int source_type = SourceType::VIDEO_FILE;
			//Add a validator here either web url type or local file type
			std::string source = temp_str;
			//Check if either exists
			face_recognition.setSource(source);
   		  	face_recognition.setSourceType(source_type);
     		}
	}
         if (result.count("k")){
       	 	std::cerr << result["skip"].as<std::string>() << std::endl;
		std::string temp_str = result["skip"].as<std::string>();
		face_recognition.setSkipRate(boost::lexical_cast<int>(temp_str));
                        std::cerr << "found type k" << std::endl;
 	}
	  if (result.count("t")){
                       std::cerr << result["new_flag"].as<std::string>() << std::endl;
                   std::string temp_str = result["new_flag"].as<std::string>();
                  face_recognition.setnewflag(boost::lexical_cast<int>(temp_str));
                     std::cerr << "found type t" << std::endl;
         }
	   if (result.count("u")){
                       std::cerr << result["new_double"].as<std::string>() << std::endl;
                    std::string temp_str = result["new_double"].as<std::string>();
		    face_recognition.setnewdouble(boost::lexical_cast<double>(temp_str));
                          std::cerr << "found type u" << std::endl;
	   }
	    if (result.count("v")){
                         std::cerr <<  result["new_string"].as<std::string>() << std::endl;
                    std::string temp_str = result["new_string"].as<std::string>();
                    std::string source1=temp_str;
		    face_recognition.setnewstring(source1);
                          std::cerr << "found type v" << std::endl;
                }
 if (result.count("w")){
                 std::cerr << result["for_save"].as<std::string>() << std::endl;
                  std::string temp_str = result["for_save"].as<std::string>();
                  face_recognition.setforsave(boost::lexical_cast<int>(temp_str));
                          std::cerr << "found type w" << std::endl;
          }
}


