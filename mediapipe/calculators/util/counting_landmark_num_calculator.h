// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_UTIL_COUNTING_LANDMARK_NUM_CALCULATOR_H
#define MEDIAPIPE_CALCULATORS_UTIL_COUNTING_LANDMARK_NUM_CALCULATOR_H

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe {

// A calculator that check whether there is a rect input.
//
// Count input NormalizedRect (mediapipe::NormalizedRect) and return this 
// value to ouput_stream. Input IMAGE has no effect on calculation, but is used to
// ensure that the calculator works even when the rect is empty. And if the 
// input rect is empty, the number of faces found is zero.
//
// Example config:
// node {
//   calculator: "CountingRectNumCalculator"
//   input_stream: "CLOCK:input_image"
//   input_stream: "VECTOR:pose_rect_from_landmarks"
//   output_stream: "COUNT:rect_count"
// }

template <typename VectorT>
class CountingLandmarkNumCalculator : public CalculatorBase { 
  public:
    static ::mediapipe::Status GetContract(CalculatorContract* cc) {
        // Check tag.
        RET_CHECK(cc->Inputs().HasTag("CLOCK"));
        cc->Inputs().Tag("CLOCK").SetAny();
        RET_CHECK(cc->Inputs().HasTag("VECTOR"));
        cc->Inputs().Tag("VECTOR").Set<VectorT>();
        RET_CHECK(cc->Outputs().HasTag("COUNT"));
        cc->Outputs().Tag("COUNT").Set<int>();

        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Process(CalculatorContext* cc) {
        std::unique_ptr<int> landmark_count;
        if (!cc->Inputs().Tag("VECTOR").IsEmpty()) {
            //const auto& rects = cc->Inputs().Tag("VECTOR").Get<VectorT>();
            landmark_count = absl::make_unique<int>(1);
        } else {
            landmark_count = absl::make_unique<int>(0);
        }
        cc->Outputs().Tag("COUNT").Add(landmark_count.release(), cc->InputTimestamp());

        return ::mediapipe::OkStatus();
    };    
};

}

#endif  // MEDIAPIPE_CALCULATORS_UTIL_COUNTING_VECTOR_SIZE_CALCULATOR_H 