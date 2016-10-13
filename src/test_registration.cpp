#include <glog/logging.h>

#include "additional_view_registration_server/additional_view_registration_optimizer.h"

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <tf_conversions/tf_eigen.h>
#include <boost/log/trivial.hpp>

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef typename Cloud::Ptr CloudPtr;

using namespace std;

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    string folder;

    if (argc > 1){
        folder = argv[1];
    } else {
        cout <<"Please provide folder with clouds"<<endl;
        return -1;
    }

    pcl::visualization::PCLVisualizer* pg;
    pg = new pcl::visualization::PCLVisualizer (argc, argv, "global_transform");
    pg->setBackgroundColor(255,255,255);


    // get input clouds
    vector<CloudPtr> input_clouds;
    boost::filesystem::recursive_directory_iterator it2(folder);
    boost::filesystem::recursive_directory_iterator endit;
    int pcd_counter = 0;
    std::string ext = ".pcd";
    while (it2 != endit){
        if(boost::filesystem::is_regular_file(*it2) && it2->path().extension() == ext){
           pcd_counter++;
           // load cloud
           boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> > new_cloud = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB> >(new pcl::PointCloud<pcl::PointXYZRGB>);
           pcl::io::loadPCDFile(it2->path().string().c_str(), *new_cloud);
           input_clouds.push_back(new_cloud);
        }
        ++it2;
    }
    if (pcd_counter > 0){
        cout<<"In data folder "<<folder <<" I found "<<pcd_counter<<" pcd files "<<endl;
    }

    // visualize
    for (int i=0; i<input_clouds.size();++i){
        stringstream ss; ss<<"Cloud"<<i;
        pg->addPointCloud(input_clouds[i], ss.str());
    }
    pg->spin();
    pg->removeAllPointClouds();

    // register!!
    bool verbose = true;
    std::vector<tf::StampedTransform> empty_transforms;
    std::vector<int> number_of_constraints;
    std::vector<tf::Transform> registered_poses;
    AdditionalViewRegistrationOptimizer optimizer(verbose);
    optimizer.registerViews(input_clouds, empty_transforms,
                            number_of_constraints, registered_poses);

    // visualize registered data
    for (size_t i=0; i<input_clouds.size();i++){
        CloudPtr transformedCloud(new Cloud);
        *transformedCloud = *input_clouds[i];
        pcl_ros::transformPointCloud(*transformedCloud, *transformedCloud,registered_poses[i]);

        stringstream ss; ss<<"Cloud";ss<<i;
        pg->addPointCloud(transformedCloud, ss.str());
    }

    pg->spin();
    pg->removeAllPointClouds();
}
