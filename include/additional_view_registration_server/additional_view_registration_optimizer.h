#pragma once

#include <pcl/common/common.h>
#include <opencv/cv.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/transforms.h>
#include <siftgpu/SiftGPU.h>


template<typename PointT>
class AdditionalViewRegistrationOptimizer
{
private:
    bool m_bVerbose;

public:
    AdditionalViewRegistrationOptimizer(bool verbose=false)
        :m_bVerbose (verbose)
    {}

    ~AdditionalViewRegistrationOptimizer()
    {}

    bool
    registerViews(const std::vector<typename pcl::PointCloud<PointT>::Ptr>& all_views, const std::vector<tf::StampedTransform>& all_initial_poses,
                  std::vector<int>& number_of_constraints, std::vector<tf::Transform>& registered_poses);

    std::vector<std::pair<PointT, PointT> >
    validateMatches(const std::vector<std::pair<SiftGPU::SiftKeypoint,SiftGPU::SiftKeypoint>>& matches,
                    const cv::Mat& image1, const cv::Mat& image2,
                    const  typename pcl::PointCloud<PointT>::ConstPtr & cloud1, const typename pcl::PointCloud<PointT>::Ptr & cloud2,
                    const double& depth_threshold,
                    std::vector<std::pair<cv::Point2d, cv::Point2d>>& remaining_image_matches);
};
