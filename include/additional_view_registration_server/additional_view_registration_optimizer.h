#pragma once

#include <pcl/common/common.h>
#include <opencv/cv.h>
#include <siftgpu/SiftGPU.h>


template<typename PointT>
class ViewRegister
{
private:
    bool m_bVerbose;

public:
    ViewRegister(bool verbose=false)
        :m_bVerbose (verbose)
    {}

    ~ViewRegister()
    {}

    bool
    registerViews(const std::vector<typename pcl::PointCloud<PointT>::Ptr>& all_views,
            const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &all_initial_poses,
            std::vector<int>& number_of_constraints,
            std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &registered_poses2);

    std::vector<std::pair<PointT, PointT> >
    validateMatches(const std::vector<std::pair<SiftGPU::SiftKeypoint,SiftGPU::SiftKeypoint>>& matches,
                    const cv::Mat& image1, const cv::Mat& image2,
                    const typename pcl::PointCloud<PointT>::ConstPtr & cloud1, const typename pcl::PointCloud<PointT>::ConstPtr &cloud2,
                    double depth_threshold,
                    std::vector<std::pair<cv::Point2d, cv::Point2d>>& remaining_image_matches);
};
