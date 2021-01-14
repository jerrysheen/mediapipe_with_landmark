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

// for receiving the hand_landmark data
#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"


// for receiving the rect data
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/calculators/util/rect_to_render_data_calculator.pb.h"

// input and output streams to be used/retrieved by calculators
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

// get the rect,hand_count, and hand_landmark output stream.
// you can find this name (in " ") in the correspondense pbtxt, they are all output stream.
// for this code the pbtxt will be 'face_mesh_out_cpu.pbtxt'
constexpr char kOutputLandmarksStream[] = "landmarks";
constexpr char kOutputHandCountStream[] = "hand_count";
constexpr char kLandmarksRectStream[] = "multi_hand_rects";


// cli inputs
DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

::mediapipe::Status RunMPPGraph() {
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
  if (save_video) {
    LOG(INFO) << "Prepare video writer.";
    cv::Mat test_frame;
    capture.read(test_frame);                    // Consume first frame.
    capture.set(cv::CAP_PROP_POS_AVI_RATIO, 0);  // Rewind to beginning.
    writer.open(FLAGS_output_video_path,
                mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                capture.get(cv::CAP_PROP_FPS), test_frame.size());
    RET_CHECK(writer.isOpened());
  } else {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  }




  // output stream (i.e. rendered landmark frame)
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller image_poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  
  // register hand_count, landmark, rect stream 
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller hand_count_poller,
                   graph.AddOutputStreamPoller(kOutputHandCountStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarks_poller,
                   graph.AddOutputStreamPoller(kOutputLandmarksStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark_rect,
            graph.AddOutputStreamPoller(kLandmarksRectStream));




  LOG(INFO) << "Start running the calculator graph.";
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";

  bool grab_frames = true;
  int prev_face_count = -1;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    // get frame size by : 
    // rows = camera_frame_raw.rows;
    // cols = camera_frame_raw.cols;

    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;

    if (camera_frame_raw.empty()) break;  // End of video.
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

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet image_packet;
    if (!image_poller.Next(&image_packet)) break;
    auto& output_frame = image_packet.Get<mediapipe::ImageFrame>();

    mediapipe::Packet hand_count_packet;
    if (!hand_count_poller.Next(&hand_count_packet)) break;
    auto& hand_count = hand_count_packet.Get<int>();

    std::cout << "waiting...............\n";
    if (hand_count > 0) {
      
      std::cout << "start to received lmk data\n";
      
      // get the landmark data...
      mediapipe::Packet landmarks_packet;
      if (!landmarks_poller.Next(&landmarks_packet)) break;
      auto& landmarks = landmarks_packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
      std::cout << frame_timestamp_us << std::endl;
      for (const ::mediapipe::NormalizedLandmarkList& landmark : landmarks) {
        std::cout << landmark.DebugString();
      }

      // hand landmarks_rect  print
      mediapipe::Packet landmark_rect_packet;
      if (!poller_landmark_rect.Next(&landmark_rect_packet)) break;
      auto& output_landmarks_rect = landmark_rect_packet.Get<std::vector<::mediapipe::NormalizedRect>>();
      std::cout << "start to received rct data\n";
      for (const ::mediapipe::NormalizedRect& rect : output_landmarks_rect) {
        std::cout << rect.DebugString();
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
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
  } else {
    LOG(INFO) << "Success!";
  }
  return 0;
}