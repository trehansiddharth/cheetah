// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

using namespace std;

int main()
{
    rs2::pipeline pipe;
    rs2::pipeline_profile selection = pipe.start();
    auto depth_stream = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto resolution = std::make_pair(depth_stream.width(), depth_stream.height());
    auto i = depth_stream.get_intrinsics();
    auto principal_point = std::make_pair(i.ppx, i.ppy);
    auto focal_length = std::make_pair(i.fx, i.fy);
    rs2_distortion model = i.model;

    printf("%i x %i: \n", i.height, i.width);
    printf("%f %f %f \n", i.fx, 0.0f, i.ppx);
    printf("%f %f %f \n", 0.0f, i.fy, i.ppy);
    printf("%f %f %f \n", 0.0f, 0.0f, 1.0f);

    return 0;
}
