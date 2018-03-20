%module run_realsense

%include <opencv.i>
%cv_instantiate_all_defaults

%{
    #include "run_realsense.hpp"
%}

%include "run_realsense.hpp"
