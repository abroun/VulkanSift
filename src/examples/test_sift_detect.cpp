#include "test_utils.h"
#include <vulkansift/vulkansift.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "Invalid command." << std::endl;
    std::cout << "Usage: ./test_sift_detect PATH_TO_IMAGE" << std::endl;
    return -1;
  }

  // Read image with OpenCV
  cv::Mat image = cv::imread(argv[1], 0);
  if (image.empty())
  {
    std::cout << "Failed to read image " << argv[1] << ". Stopping program." << std::endl;
    return -1;
  }

  vksift_setLogLevel(VKSIFT_LOG_INFO);

  // Load the Vulkan API (should never be called more than once per program)
  if (vksift_loadVulkan() != VKSIFT_SUCCESS)
  {
    std::cout << "Impossible to initialize the Vulkan API" << std::endl;
    return -1;
  }

  // Create a vksift instance using the default configuration
  vksift_Config config = vksift_getDefaultConfig();
  config.sift_buffer_count = 1; // only performing detection, a single GPU buffer is enough
  config.input_image_max_size = image.cols * image.rows;
  config.allow_cpu = true;

  vksift_Instance vksift_instance = NULL;
  if (vksift_createInstance(&vksift_instance, &config) != VKSIFT_SUCCESS)
  {
    std::cout << "Impossible to create the vksift_instance" << std::endl;
    vksift_unloadVulkan();
    return -1;
  }

  std::vector<vksift_Feature> feat_vec;
  bool draw_oriented_keypoints = true;

  std::chrono::steady_clock::duration avgDetectDuration;
  int numDetections = 0;

  int user_key = 0;
  while (user_key != 'x')
  {
    // Run SIFT feature detection and copy the results to the CPU
    auto detectStartTime = std::chrono::steady_clock::now();
    std::cout << "Starting to detect features" << std::endl;
    vksift_detectFeatures(vksift_instance, image.data, image.cols, image.rows, 0u);
    feat_vec.resize(vksift_getFeaturesNumber(vksift_instance, 0u));
    std::cout << "Starting to download features" << std::endl;
    vksift_downloadFeatures(vksift_instance, feat_vec.data(), 0u);
    auto detectEndTime = std::chrono::steady_clock::now();

    avgDetectDuration = (avgDetectDuration*numDetections + (detectEndTime - detectStartTime))/(numDetections + 1);
    numDetections++;

    std::cout << "Feature found: " << feat_vec.size() << " avg detect time = " 
      << std::chrono::duration_cast<std::chrono::milliseconds>(avgDetectDuration).count() << " ms" << std::endl;

    cv::Mat draw_frame;
    image.convertTo(draw_frame, CV_8UC3);
    cv::cvtColor(draw_frame, draw_frame, cv::COLOR_GRAY2BGR);
    if (draw_oriented_keypoints)
    {
      draw_frame = getOrientedKeypointsImage(image.data, feat_vec, image.cols, image.rows);
    }
    else
    {
      // Draw only points at the SIFT position
      for (int i = 0; i < (int)feat_vec.size(); i++)
      {
        cv::circle(draw_frame, cv::Point(feat_vec[i].x, feat_vec[i].y), 3, cv::Scalar(0, 0, 255), 1);
      }
    }

    cv::putText(draw_frame, "x: exit", cv::Size{10, draw_frame.rows - 20}, cv::FONT_HERSHEY_COMPLEX, 0.5f, cv::Scalar(0, 255, 0));
    cv::imshow("VulkanSIFT keypoints", draw_frame);
    user_key = cv::waitKey(1);
  }

  // Release vksift instance and API
  vksift_destroyInstance(&vksift_instance);
  vksift_unloadVulkan();

  return 0;
}