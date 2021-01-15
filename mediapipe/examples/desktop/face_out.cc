// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <regex>
#include <vector>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

// for receiving the rect data
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"


// for receiving the rect data
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/calculators/util/rect_to_render_data_calculator.pb.h"


using namespace std;

constexpr char kInputStream[] = "input_video";
constexpr char kOutputImageStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

// get the rect,face_count, and face_landmark output stream.
// you can find this name (in " ") in the correspondense pbtxt, they are all output stream.
// for this code the pbtxt will be 'face_mesh_desktop_live.pbtxt'
constexpr char kLandmarksRectStream[] = "face_rects_from_landmarks";
constexpr char kOutputFaceCountStream[] = "face_count";
constexpr char kOutputLandmarksStream[] = "multi_face_landmarks";


DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  #if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
      capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
      capture.set(cv::CAP_PROP_FPS, 30);
  #endif
  }


  
  
  LOG(INFO) << "Start running the calculator graph.";

  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller image_poller,
                   graph.AddOutputStreamPoller(kOutputImageStream));
  
  // register face_count, landmark, rect stream 
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller face_count_poller,
                   graph.AddOutputStreamPoller(kOutputFaceCountStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
                   graph.AddOutputStreamPoller(kOutputLandmarksStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark_rect,
                   graph.AddOutputStreamPoller(kLandmarksRectStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));



  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  int prev_face_count = -1;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    mediapipe::Packet image_packet;
    if (!image_poller.Next(&image_packet)) break;
    auto& output_frame = image_packet.Get<mediapipe::ImageFrame>();

    // get the face_count packet, it will automatically count the packet that 
    // will go into landmark related subgraph.
    // if there is no face detected, then none of any stream will flow into landmark
    // module, (it will cause the frame stuck.).
    // the face_count calculator will count packet size, and then send it to landmark
    // module, with this calculator continuing sending data, the frame then won't stuck.
    mediapipe::Packet face_count_packet;
    if (!face_count_poller.Next(&face_count_packet)) break;
    auto& face_count = face_count_packet.Get<int>();

    std::cout << "waiting...............\n";
    if (face_count > 0) {
        
      // the landmarks you received here is a vector list, so it has some property
      // thay vetor will have:
      // eg:std::cout << landmarks[0].landmark_size();
      mediapipe::Packet landmarks_packet;
      if (!landmarks_poller.Next(&landmarks_packet)) break;
      auto& landmarks = landmarks_packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
      // print the landmark using this code
      for (const ::mediapipe::NormalizedLandmarkList& landmark : landmarks) {
        // std::cout << landmark.DebugString();
        // start to deal with string using regular expression.
        //string str = landmark.DebugString();
        string str = landmark.DebugString();
        // insert code here...
        //string str = "landmark {\n  x: 0.613949597\n  y: 0.512591064\n  z: -0.00256898953\n  visibility: 1\n  presence: 0.996413052\n}\nlandmark {\n  x: 0.611436605\n  y: 0.460630834\n  z: 0.00740333134\n  visibility: 1\n  presence: 0.993446052\n}\nlandmark {\n  x: 0.621986091\n  y: 0.460139275\n  z: -0.012594508\n  visibility: 0.999970436\n  presence: 0.992199838\n}\nlandmark {\n  x: 0.632396579\n  y: 0.45973289\n  z: -0.0055285017\n  visibility: 1\n  presence: 0.990741372\n}\nlandmark {\n  x: 0.572493136\n  y: 0.4619295\n  z: 0.000160436204\n  visibility: 1\n  presence: 0.996003449\n}\nlandmark {\n  x: 0.55606997\n  y: 0.462284327\n  z: 0.00086248375\n  visibility: 0.999983072\n  presence: 0.99661094\n}\nlandmark {\n  x: 0.539810061\n  y: 0.462845564\n  z: -0.00561079429\n  visibility: 1\n  presence: 0.99584806\n}\nlandmark {\n  x: 0.611480474\n  y: 0.477871656\n  z: -0.00129700301\n  visibility: 0.999875784\n  presence: 0.99706\n}\nlandmark {\n  x: 0.488170177\n  y: 0.486361384\n  z: 0.00296257483\n  visibility: 0.999917269\n  presence: 0.997121274\n}\nlandmark {\n  x: 0.619232595\n  y: 0.567374766\n  z: 0.00334401568\n  visibility: 0.9999\n  presence: 0.999299765\n}\nlandmark {\n  x: 0.569764555\n  y: 0.569226623\n  z: 0.00274809683\n  visibility: 0.999966383\n  presence: 0.999711215\n}\nlandmark {\n  x: 0.693515658\n  y: 0.71861738\n  z: -0.331908435\n  visibility: 0.999950409\n  presence: 0.999360144\n}\nlandmark {\n  x: 0.359382868\n  y: 0.745541513\n  z: -0.446730971\n  visibility: 0.999966621\n  presence: 0.999299049\n}\nlandmark {\n  x: 0.854609847\n  y: 0.986356258\n  z: -0.0496203899\n  visibility: 0.426453233\n  presence: 0.575895548\n}\nlandmark {\n  x: 0.294551075\n  y: 1.09573519\n  z: -0.253446579\n  visibility: 0.307693303\n  presence: 0.344096214\n}\nlandmark {\n  x: 0.995646119\n  y: 1.19363952\n  z: -0.231866375\n  visibility: 0.0917004868\n  presence: 0.0510586463\n}\nlandmark {\n  x: 0.290956229\n  y: 1.37292445\n  z: -0.347637683\n  visibility: 0.13372393\n  presence: 0.0619239025\n}\nlandmark {\n  x: 1.05973673\n  y: 1.25733399\n  z: -0.265884429\n  visibility: 0.0883765593\n  presence: 0.0317905508\n}\nlandmark {\n  x: 0.252514035\n  y: 1.46270847\n  z: -0.409361899\n  visibility: 0.10280589\n  presence: 0.03726184\n}\nlandmark {\n  x: 1.03619456\n  y: 1.27368474\n  z: -0.307165563\n  visibility: 0.137550548\n  presence: 0.0426366404\n}\nlandmark {\n  x: 0.299845159\n  y: 1.47575617\n  z: -0.446936309\n  visibility: 0.156356648\n  presence: 0.0543562509\n}\nlandmark {\n  x: 1.00429773\n  y: 1.25669742\n  z: -0.29283461\n  visibility: 0.149947524\n  presence: 0.0422711261\n}\nlandmark {\n  x: 0.320839852\n  y: 1.44397771\n  z: -0.432022244\n  visibility: 0.183153272\n  presence: 0.0603121817\n}\nlandmark {\n  x: 0.670869589\n  y: 1.34571433\n  z: 0.103311144\n  visibility: 0.0265163947\n  presence: 0.00194624288\n}\nlandmark {\n  x: 0.449693263\n  y: 1.36372232\n  z: -0.0709834844\n  visibility: 0.0179650653\n  presence: 0.00184070901\n}\nlandmark {\n  x: 0.677823\n  y: 1.82070255\n  z: 0.202397808\n  visibility: 2.81318371e-05\n  presence: 0.000949093199\n}\nlandmark {\n  x: 0.461592495\n  y: 1.82459021\n  z: 0.128505141\n  visibility: 4.49201543e-05\n  presence: 0.000335714954\n}\nlandmark {\n  x: 0.672832251\n  y: 2.23668718\n  z: 0.401643395\n  visibility: 4.92473919e-06\n  presence: 0.000119101358\n}\nlandmark {\n  x: 0.453322977\n  y: 2.24761581\n  z: 0.164003715\n  visibility: 2.65558756e-05\n  presence: 6.38726706e-05\n}\nlandmark {\n  x: 0.671541214\n  y: 2.30948019\n  z: 0.422435731\n  visibility: 7.93545769e-06\n  presence: 0.00013375617\n}\nlandmark {\n  x: 0.442752838\n  y: 2.32013273/n  z: 0.234550849\n  visibility: 2.83206391e-05\n  presence: 6.89925655e-05\n}\nlandmark {\n  x: 0.679653049\n  y: 2.36415768\n  z: 0.415985525\n  visibility: 6.93967195e-06\n  presence: 0.000137792522\n}\nlandmark {\n  x: 0.517023504\n  y: 2.37463856\n  z: 0.280007362\n  visibility: 2.20659531e-05\n  presence: 6.99142256e-05\n}\n\nstart to received rct data\nx_center: 0.560146093\ny_center: 1.35494471\nheight: 3.12043524\nwidth: 2.34032631\nrotation: -0.072670579\n";
        //cout << str << endl;
        
        // regular expression 取值    
        std::regex reg("[\\\n|\\s]");
        // output: hello, world!
        str = regex_replace(str, reg, "");
        //cout << str << endl;
        //cout << str << endl;
        smatch result;
        //regex pattern("\\{([^}]*)\\}");    //{中的内容}
        regex pattern("x([^z]*)");
        regex_search(str, result, pattern);    
        //遍历结果
        
        //迭代器声明
        string::const_iterator iterStart = str.begin();
        string::const_iterator iterEnd = str.end();
        regex pattern1("[\\d|.]+[^y]");
        regex pattern2("y:.+");
        smatch xcord;
        smatch ycord;
        double x;
        double y; 
        vector<vector<double>> cordinate_collection;
        // result 0 x:0.74297905y:0.690012038
        while (regex_search(iterStart, iterEnd, result, pattern))
        {
            vector<double> pair;
            str = result[0];
            // 0.742979获取
            //cout << xcord[0] << endl;
            regex_search(str, xcord, pattern1);
            regex_search(str, ycord, pattern2);
            //cout << xcord.str(0) << endl;
            //cout << ycord.str(0).substr(2,ycord.str(0).length()) << endl;
            x = stod(xcord.str(0));
            y = stod(ycord.str(0).substr(2,ycord.str(0).length()));
            pair.push_back(x);
            pair.push_back(y);
            cordinate_collection.push_back(pair);
            iterStart = result[0].second;
            
        }   
      }
      

      // start to get the rect data. almost using the same way...
      mediapipe::Packet landmark_rect_packet;
      if (!poller_landmark_rect.Next(&landmark_rect_packet)) break;
      auto& output_landmarks_rect = landmark_rect_packet.Get<std::vector<::mediapipe::NormalizedRect>>();
      for (const ::mediapipe::NormalizedRect& rect : output_landmarks_rect) {
        //std::cout << rect.DebugString();
      }

      // // you can use this to seperate the output result.
      // std::cout << "...............\n";
      // std::cout << "...............\n";
      // std::cout << "...............\n";
      // std::cout << "...............\n";
      // std::cout << "...............\n";
      // std::cout << "...............\n";
      // std::cout << "...............\n";
      // std::cout << "...............\n";
    
    }
    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(FLAGS_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
