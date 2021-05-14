#include <array>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "SuperPoint.h"
#include "SPextractor.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM2;

int main(int argc, const char* argv[])
{
    std::string image_path_1 = samples::findFile("../data/img_1.jpg");
    cv::Mat input_1 = cv::imread(image_path_1, 0); //Load as grayscale

    std::string image_path_2 = samples::findFile("../data/img_2.jpg");
    cv::Mat input_2 = cv::imread(image_path_2, 0); //Load as grayscale

    resize(input_1, input_1, Size(input_1.cols/2, input_1.rows/2));
    resize(input_2, input_2, Size(input_2.cols/2, input_2.rows/2));

    // std::cout << "image dimension (" << img.cols << "x" << img.rows << ")" << std::endl;

    imshow("Display window", input_1);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", input_1);
    }

    /*
    // SURF // 

    Ptr< cv::xfeatures2d::SURF> surf =  xfeatures2d::SURF::create();
    std::vector<cv::KeyPoint> keypoints;
    surf->detect(input_1, keypoints);

    // Add results to image and save.
    cv::Mat output_surf;
    cv::drawKeypoints(input_1, keypoints, output_surf);
    cv::imwrite("../data/result/surf_result.jpg", output_surf);


    // SIFT //

    cv::Ptr<Feature2D> sift = xfeatures2d::SIFT::create();

    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;    
    sift->detect(input_1, keypoints_1);
    sift->detect(input_2, keypoints_2);

    // Add results to image and save.
    cv::Mat output_sift;
    cv::drawKeypoints(input_1, keypoints_1, output_sift);
    cv::imwrite("../data/result/sift_result.jpg", output_sift);



    // //-- Step 2: Calculate descriptors (feature vectors)    
    // Mat descriptors_1, descriptors_2;   
    // sift->compute( input_1, keypoints_1, descriptors_1 );
    // sift->compute( input_2, keypoints_2, descriptors_2 );


    // //-- Step 3: Matching descriptor vectors using BFMatcher :
    // BFMatcher matcher;
    // std::vector< DMatch > matches;
    // matcher.match( descriptors_1, descriptors_2, matches );


    // cv::Mat output_sift_matches;
    // cv::drawMatches(input_1, keypoints_1, input_2, keypoints_2,                
    //                 matches, output_sift_matches);              
    // cv::namedWindow("matches");
    // cv::imshow("matches",output_sift_matches);
    // cv::waitKey(0);
    // cv::imwrite("../data/sift_result_matches.jpg", output_sift);


    // ORB //

    cv::Ptr<Feature2D> orb = ORB::create();

    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints_orb;    
    orb->detect(input_1, keypoints_orb);

    // Add results to image and save.
    cv::Mat output_orb;
    cv::drawKeypoints(input_1, keypoints_orb, output_orb);
    cv::imwrite("../data/result/orb_result.jpg", output_orb);
    */

    int nFeatures = 1000;
    float fScaleFactor = 1.2;
    int nLevels = 8;
    int fIniThFAST = 20;
    int fMinThFAST = 7;

    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvInvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::vector<cv::KeyPoint> mvKeys;
    ORBextractor* mpORBextractorLeft;
    const cv::Mat mDescriptors;
    int N;

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // // mnScaleLevels = mpORBextractorLeft->GetLevels();
    // // mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    // mfLogScaleFactor = log(mfScaleFactor);
    // mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    // mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    // mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    // mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
    

    // (*mpORBextractorLeft)(input_1, cv::Mat(), mvKeys, mDescriptors);
    // cout << "desc size: " << mDescriptors.rows << ' ' << mDescriptors.cols << endl;

    // N = mvKeys.size();

    // UndistortKeyPoints();

    // ComputeStereoMatches();

    // mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    // mvbOutlier = vector<bool>(N,false);

    // ComputeImageBounds(imLeft);

    // mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
    // mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

    // fx = K.at<float>(0,0);
    // fy = K.at<float>(1,1);
    // cx = K.at<float>(0,2);
    // cy = K.at<float>(1,2);
    // invfx = 1.0f/fx;
    // invfy = 1.0f/fy;

    // mb = mbf/fx;

    // AssignFeaturesToGrid();
    
    return 0;
}