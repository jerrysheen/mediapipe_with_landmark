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
            std::cout << landmark.DebugString();
      }

      // start to get the rect data. almost using the same way...
      mediapipe::Packet landmark_rect_packet;
      if (!poller_landmark_rect.Next(&landmark_rect_packet)) break;
      auto& output_landmarks_rect = landmark_rect_packet.Get<std::vector<::mediapipe::NormalizedRect>>();
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
