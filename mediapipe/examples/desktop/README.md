## To run hand_tracking demo:
you can run hand_tracking demo, and get landmark and rect info without frame stuck using the following command:  
**build:**  
`bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_out_cpu`  
**run:**    
`bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_out_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt`  
## To run face_mesh demo:  
you can run face_mesh demo, and get landmark and rect info without frame stuck using the following command:  
**build:**  
`bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_mesh:face_mesh_out_cpu`  
**run:**  
`bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_out_cpu --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt`  
## To run pose_tracking demo:  
**build:**  
`bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/pose_tracking:pose_tracking_out_cpu`  
**run:**  
`bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_out_cpu --calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt`  

# Run it on windows： 
first you need to change your WORKSPACE File:
```
new_local_repository(
    name = "windows_opencv",
    build_file = "@//third_party:opencv_windows.BUILD",
    path = "E:\\Work\\opcv_3.4.10\\opencv\\build",
) 
```
add path to your opencv build path.

## To run hand_mark example
### build:
`bazel build -c opt  --action_env PYTHON_BIN_PATH="C://ProgramData//Anaconda3//python.exe" --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_out_cpu`
### run：
`bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_out_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt`  

## To run pose_tracking:
### build:
`bazel build -c opt  --action_env PYTHON_BIN_PATH="C://ProgramData//Anaconda3//python.exe" --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/pose_tracking:pose_tracking_out_cpu`
### run：
`bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_out_cpu --calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt`  

## To run face_mesh:  
### build:
`bazel build -c opt  --action_env PYTHON_BIN_PATH="C://ProgramData//Anaconda3//python.exe" --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_mesh:face_mesh_out_cpu`
### run：
`bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_out_cpu --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt`



