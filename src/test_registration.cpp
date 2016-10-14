#include <glog/logging.h>

#include "additional_view_registration_server/additional_view_registration_optimizer.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/log/trivial.hpp>

#include <boost/algorithm/string.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/regex.hpp>

namespace bf = boost::filesystem;

namespace io
{
bool
existsFolder ( const std::string &rFolder )
{
    bf::path dir = rFolder;
    return bf::exists (dir);
}

std::vector<std::string>
getFilesInDirectory (const std::string &dir,
                     const std::string &regex_pattern,
                     bool recursive)
{
    std::vector<std::string> relative_paths;

    if ( !existsFolder( dir ) ) {
        std::cerr << dir << " is not a directory!" << std::endl;
    }
    else
    {
        bf::path bf_dir  = dir;
        bf::directory_iterator end_itr;
        for (bf::directory_iterator itr ( bf_dir ); itr != end_itr; ++itr)
        {
            const std::string file = itr->path().filename().string ();

            //check if its a directory, then get files in it
            if (bf::is_directory (*itr))
            {
                if (recursive)
                {
                    std::vector<std::string> files_in_subfolder  = getFilesInDirectory ( dir + "/" + file, regex_pattern, recursive);
                    for (const auto &sub_file : files_in_subfolder)
                        relative_paths.push_back( file + "/" + sub_file );
                }
            }
            else
            {
                //check for correct file pattern (extension,..) and then add, otherwise ignore..
                boost::smatch what;
                const boost::regex file_filter( regex_pattern );
                if( boost::regex_match( file, what, file_filter ) )
                    relative_paths.push_back ( file);
            }
        }
        std::sort(relative_paths.begin(), relative_paths.end());
    }
    return relative_paths;
}
}

int main(int argc, char** argv)
{
    typedef pcl::PointXYZRGB PointT;
    std::string folder;

    if (argc > 1)
        folder = argv[1];
    else
    {
        cout <<"Please provide folder with clouds"<<endl;
        return -1;
    }


    // get input clouds
    std::vector<std::string> cloud_fns = io::getFilesInDirectory( folder, ".*.pcd", false );
    std::vector<pcl::PointCloud<PointT>::Ptr > input_clouds (cloud_fns.size());
    for(size_t i=0; i<cloud_fns.size(); i++)
    {
        input_clouds[i].reset (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile( folder + "/" + cloud_fns[i], *input_clouds[i]);
    }

    // register!!
    bool verbose = true;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > empty_transforms;
    std::vector<int> number_of_constraints;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > registered_poses;
    ViewRegister<PointT> optimizer(verbose);
    optimizer.registerViews(input_clouds, empty_transforms, number_of_constraints, registered_poses);

    // visualize registered data
    pcl::visualization::PCLVisualizer vis ("global_transform");
    vis.setBackgroundColor(255,255,255);

    for (size_t i=0; i<input_clouds.size();i++)
    {
        pcl::PointCloud<PointT>::Ptr transformedCloud(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*input_clouds[i], *transformedCloud, registered_poses[i]);
        std::stringstream ss; ss<<"Cloud";ss<<i;
        vis.addPointCloud(transformedCloud, ss.str());
    }
    vis.spin();
}
