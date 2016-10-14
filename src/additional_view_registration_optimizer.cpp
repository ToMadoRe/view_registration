#include "additional_view_registration_server/additional_view_registration_optimizer.h"
#include "additional_view_registration_server/sift_wrapper.h"
#include "additional_view_registration_server/additional_view_registration_residual.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <ceres/ceres.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

void
Mat4f2RotTrans(const Eigen::Matrix4f &tf, Eigen::Quaternionf &q, Eigen::Vector3f &trans)
{
    Eigen::Matrix3f rotation = tf.block<3,3>(0,0);
    q = rotation;
    trans = tf.block<3,1>(0,3);
}

Eigen::Matrix4f
RotTrans2Mat4f(const Eigen::Quaternionf &q, const Eigen::Vector3f &trans)
{
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    tf.block<3,1>(0,3) = trans;
    return tf;
}

template<typename PointT>
std::pair<cv::Mat, cv::Mat>
createRGBandDepthFromCloud(const pcl::PointCloud<PointT> &cloud)
{
    std::pair<cv::Mat, cv::Mat> toRet;
    toRet.first = cv::Mat::zeros(cloud.height, cloud.width, CV_8UC3); // RGB image
    toRet.second = cv::Mat::zeros(cloud.height, cloud.width, CV_16UC1); // Depth image
    for (size_t y = 0; y < toRet.first.rows; y++)
    {
        for (size_t x = 0; x < toRet.first.cols; x++)
        {
            const PointT &point = cloud.points[y*toRet.first.cols + x];
            // RGB
            toRet.first.at<cv::Vec3b>(y, x)[0] = point.b;
            toRet.first.at<cv::Vec3b>(y, x)[1] = point.g;
            toRet.first.at<cv::Vec3b>(y, x)[2] = point.r;
            // Depth
            if (!pcl::isFinite(point))
                toRet.second.at<u_int16_t>(y, x) = 0.0; // convert to uint 16 from meters
            else
                toRet.second.at<u_int16_t>(y, x) = point.z*1000; // convert to uint 16 from meters
        }
    }
    return toRet;
}


template<typename PointT>
std::vector<std::pair<PointT, PointT> >
ViewRegister<PointT>::validateMatches(const std::vector<std::pair<SiftGPU::SiftKeypoint,SiftGPU::SiftKeypoint>>& matches,
                                      const cv::Mat& image1, const cv::Mat& image2,
                                      const typename pcl::PointCloud<PointT>::ConstPtr & cloud1, const typename pcl::PointCloud<PointT>::ConstPtr & cloud2,
                                      double depth_threshold,
                                      std::vector<std::pair<cv::Point2d, cv::Point2d>>& remaining_image_matches)
{
    std::vector<std::pair<PointT, PointT> > filtered_matches;

    // intermediate data structure
    pcl::CorrespondencesPtr correspondences_sift(new pcl::Correspondences);
    // filter nans and depth outside range
    for (size_t i=0; i<matches.size(); i++)
    {
        auto keypoint_pair = matches[i];
        // get point indices in point clouds
        long point_index1 = (int)keypoint_pair.first.y * image1.cols + (int)keypoint_pair.first.x;
        long point_index2 = (int)keypoint_pair.second.y * image2.cols + (int)keypoint_pair.second.x;
        PointT p1 = cloud1->points[point_index1];
        PointT p2 = cloud2->points[point_index2];

        if (!pcl::isFinite(p1) || !pcl::isFinite(p2))
            continue; // skip nan matches

        if ( (depth_threshold > 0.0) && ((p1.z > depth_threshold) || (p2.z > depth_threshold) ) )
            continue; // skip points with depth outside desired range

        pcl::Correspondence c;
        c.index_match = point_index2;
        c.index_query = point_index1;
        c.distance = 1.0;
        correspondences_sift->push_back(c);
    }

    // next do RANSAC filtering
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> cr;
    pcl::Correspondences sac_correspondences;
    cr.setInputSource(cloud1);
    cr.setInputTarget(cloud2);
    cr.setInputCorrespondences(correspondences_sift);
    cr.setInlierThreshold(0.05);
    cr.setMaximumIterations(10000);
    cr.getCorrespondences(sac_correspondences);

    if (sac_correspondences.size() == correspondences_sift->size()) // could not find a consensus
        return filtered_matches;

    for (auto corresp : sac_correspondences)
    {
        std::pair<PointT, PointT> filtered_match(cloud1->points[corresp.index_query], cloud2->points[corresp.index_match]);
        filtered_matches.push_back(filtered_match);

        cv::Point2d p1_2d, p2_2d;
        p1_2d.y =  (corresp.index_query / image1.cols);
        p1_2d.x =  (corresp.index_query % image1.cols);
        p2_2d.y =  (corresp.index_match / image2.cols);
        p2_2d.x =  (corresp.index_match % image2.cols);
        std::pair<cv::Point2d, cv::Point2d> filtered_match_2d(p1_2d, p2_2d);
        remaining_image_matches.push_back(filtered_match_2d);
    }

    return filtered_matches;
}

template<typename PointT>
bool
ViewRegister<PointT>::registerViews(
        const std::vector<typename pcl::PointCloud<PointT>::Ptr>& all_views,
        const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& all_initial_poses,
        std::vector<int>& number_of_constraints,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& registered_poses)
{
    std::vector<cv::Mat> vRGBImages (all_views.size());
    std::vector<cv::Mat> vDepthImages (all_views.size());

    // set up initial data
    for (size_t i=0; i<all_views.size(); i++)
    {
        auto image_pair = createRGBandDepthFromCloud(*all_views[i]);
        vRGBImages[i] = image_pair.first;
        vDepthImages[i] = image_pair.second;
    }

    // extract SIFT keypoints and descriptors
    struct SIFTData
    {
        int image_number;
        int desc_number;
        std::vector<float> descriptors;
        std::vector<SiftGPU::SiftKeypoint> keypoints;
    };

    SIFTWrapper sift_wrapper;
    std::vector<SIFTData> vSIFTData (all_views.size());

    for (size_t i=0; i<vRGBImages.size(); i++)
    {
        auto image = vRGBImages[i];
        SIFTData &sift = vSIFTData[i];
        sift_wrapper.extractSIFT(image, sift.desc_number, sift.descriptors, sift.keypoints);
        if (m_bVerbose)
            std::cout << "Extracted "<< sift.keypoints.size()<<" SIFT keypoints for image "<< i << std::endl;
    }

    // set up constraint problem
    struct ConstraintStructure
    {
        int image1;
        int image2;
        double depth_threshold;
        std::vector<std::pair<PointT, PointT>> correspondences;
        std::vector<std::pair<cv::Point2d, cv::Point2d>> correspondences_2d;
    };
    std::vector<ConstraintStructure> constraints_and_correspondences;

    for (size_t i=0; i<all_views.size()-1; i++)
    {
        for (size_t j=i+1; j<all_views.size(); j++)
        {
            ConstraintStructure constr;
            constr.image1 = i;
            constr.image2 = j;
            if (j!=i+1)
                constr.depth_threshold = 5.0; // 2.5
            else
                constr.depth_threshold = 5.0; //3.0; // 1.5

            const SIFTData &image1_sift = vSIFTData[constr.image1];
            const SIFTData &image2_sift = vSIFTData[constr.image2];

            std::vector<std::pair<SiftGPU::SiftKeypoint,SiftGPU::SiftKeypoint>> matches;
            sift_wrapper.matchSIFT(image1_sift.desc_number, image2_sift.desc_number,
                                   image1_sift.descriptors, image2_sift.descriptors,
                                   image1_sift.keypoints, image2_sift.keypoints,
                                   matches);
            if (m_bVerbose)
                std::cout << "Resulting number of matches for images "<<constr.image1<<" "<<constr.image2<<" is "<<matches.size() << std::endl;

            constr.correspondences = validateMatches(matches, vRGBImages[constr.image1], vRGBImages[constr.image2],
                    all_views[constr.image1], all_views[constr.image2],
                    constr.depth_threshold,
                    constr.correspondences_2d);
            if (m_bVerbose)
                std::cout << "After validating "<<constr.correspondences.size()<<"  "<<constr.correspondences_2d.size()<<" are left " << std::endl;

            constraints_and_correspondences.push_back(constr);
        }
    }

    // set up ceres problem
    struct Camera
    {
        double quaternion[4] = {1.0,0.0,0.0,0.0};
        double translation[3] = {0.0,0.0,0.0};
    };

    // ceres setup
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.function_tolerance = 1e-30;
    options.parameter_tolerance = 1e-20;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;
    options.num_linear_solver_threads = 8;
    double min_correspondences = 12;


    std::vector<Camera>  all_cameras (all_views.size());
    // initial solution
    for (size_t i=0; i<all_views.size(); i++)
    {
        Camera &cam = all_cameras[i];

        if (all_initial_poses.size() == all_views.size())
        {
            Eigen::Quaternionf q;
            Eigen::Vector3f trans;
            Mat4f2RotTrans(all_initial_poses[i], q, trans);
            cam.quaternion[0] = q.w();
            cam.quaternion[1] = q.x();
            cam.quaternion[2] = q.y();
            cam.quaternion[3] = q.z();
            cam.translation[0] = trans[0];
            cam.translation[1] = trans[1];
            cam.translation[2] = trans[2];
        }

        problem.AddParameterBlock(cam.quaternion,4);
        problem.AddParameterBlock(cam.translation,3);
        number_of_constraints.push_back(0);
    }

    // cost functions
    int total_constraints = 0;

    for (auto constr : constraints_and_correspondences)
    {
        Camera &cam1 = all_cameras[constr.image1];
        Camera &cam2 = all_cameras[constr.image2];

        if (constr.correspondences.size() < min_correspondences)
            continue;

        for (size_t k=0; k<constr.correspondences.size(); k++)
        {
            auto corresp = constr.correspondences[k];
            double point_original1[3] = {corresp.first.x, corresp.first.y, corresp.first.z};
            double point_original2[3] = {corresp.second.x, corresp.second.y, corresp.second.z};

            double weight = (corresp.first.getVector3fMap().squaredNorm() + corresp.second.getVector3fMap().squaredNorm())/2;

            ceres::CostFunction *cost_function_proj = ProjectionResidual::Create(point_original1, point_original2, weight);
            ceres::LossFunction *loss_function_proj = new ceres::HuberLoss(weight * 0.005);
            problem.AddResidualBlock(cost_function_proj,
                                     loss_function_proj,
                                     cam1.quaternion,
                                     cam1.translation,
                                     cam2.quaternion,
                                     cam2.translation);
            total_constraints++;
        }
        number_of_constraints[constr.image1] += constr.correspondences.size();
        number_of_constraints[constr.image2] += constr.correspondences.size();
    }

    if (all_cameras.empty())
        std::cout << "AdditionalViewRegistrationOptimizer ---- WARNING - no cameras defined for this object" << std::endl;


    problem.SetParameterBlockConstant(all_cameras[0].quaternion);
    problem.SetParameterBlockConstant(all_cameras[0].translation);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (m_bVerbose)
        std::cout << summary.FullReport() << std::endl;

    std::cout << "Additional view regitration constraints "<<total_constraints << std::endl;

    registered_poses.resize(all_cameras.size());
    for (size_t i=0; i<all_cameras.size();i++)
    {
        Eigen::Quaternionf q (all_cameras[i].quaternion[0], all_cameras[i].quaternion[1], all_cameras[i].quaternion[2], all_cameras[i].quaternion[3]);
        Eigen::Vector3f t; t << all_cameras[i].translation[0] , all_cameras[i].translation[1] , all_cameras[i].translation[2];
        Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
        tf.block<3,3>(0,0) = q.normalized().toRotationMatrix();
        tf.block<3,1>(0,3) = t;
        registered_poses[i] = tf;
    }

    // free memory
    // TODO check if this is necessary
    //        for (size_t i=0; i<allCFs.size();i++){
    //            delete allCFs[i];
    //        }
    //        for (size_t i=0; i<allLFs.size();i++){
    //            delete allLFs[i];
    //        }

    return true;
}


template class ViewRegister<pcl::PointXYZRGB>;
