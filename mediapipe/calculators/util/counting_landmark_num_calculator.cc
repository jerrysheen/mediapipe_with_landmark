#include "mediapipe/calculators/util/counting_landmark_num_calculator.h"

#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe {

typedef CountingLandmarkNumCalculator<::mediapipe::NormalizedLandmarkList>
    CountingNormalizedLandmarkVectorSizeCalculator;

REGISTER_CALCULATOR(CountingNormalizedLandmarkVectorSizeCalculator);
}  // namespace mediapipe