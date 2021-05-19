#include <array>
#include <cmath>
#include <iostream>
#include <chrono>
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
using namespace std::chrono;

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

int main(int argc, const char* argv[])
{
    std::string image_path_1 = samples::findFile("data/img_1.jpg");
    cv::Mat input_1 = cv::imread(image_path_1, 0); //Load as grayscale

    std::string image_path_2 = samples::findFile("data/img_2.jpg");
    cv::Mat input_2 = cv::imread(image_path_2, 0); //Load as grayscale

    resize(input_1, input_1, cv::Size(input_1.cols/4, input_1.rows/4));
    resize(input_2, input_2, cv::Size(input_2.cols/4, input_2.rows/4));

    // std::cout << "image dimension (" << img.cols << "x" << img.rows << ")" << std::endl;

    #pragma region IMSHOW
    imshow("Display window", input_1);
    int s = waitKey(0); // Wait for a keystroke in the window
    if(s == 's')
    {
        imwrite("starry_night.png", input_1);
    }
    #pragma endregion

    #pragma region TEST ORB EXTRACTOR

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

    // int nFeatures = 1000;
    // float fScaleFactor = 1.2;
    // int nLevels = 8;
    // float fIniThFAST = 20;
    // float fMinThFAST = 7;

    // int mnScaleLevels;
    // float mfScaleFactor;
    // float mfLogScaleFactor;
    // std::vector<float> mvScaleFactors;
    // std::vector<float> mvInvScaleFactors;
    // std::vector<float> mvLevelSigma2;
    // std::vector<float> mvInvLevelSigma2;

    // std::vector<cv::KeyPoint> mvKeys;
    // ORBextractor* mpORBextractorLeft;
    // const cv::Mat mDescriptors;
    // int N;

    // std::cout << "2" << std::endl;
    
    // ORB extraction
    // mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    // std::cout << "Keys: "<< mvKeys << ", Descriptors: " << mDescriptors << std::endl;


    // (*mpORBextractorLeft)(input_1, cv::Mat(), mvKeys, mDescriptors);

    // N = mvKeys.size();

    // if(mvKeys.empty())
    // {
    //     std::cout << "EXIT" << std::endl;
    //     // return;
    //     exit(EXIT_FAILURE);
    // }
        

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

    #pragma endregion

    int nFeatures = 1000;
    float fScaleFactor = 1.2;
    int nLevels = 8;
    float fIniThFAST = 20;
    float fMinThFAST = 7;

    float threshold;
    cout << "What threshold do you want ? ";
    cin >> threshold;

    // SPextractor ORBextractor = SPextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // OutputArray _descriptors = ORBextractor.mDescriptors;

    // InputArray _image = input_1;
    // cv::Mat image = _image.getMat();

    // // #pragma region IMSHOW
    // // imshow("Display window", image);
    // // int k = waitKey(0); // Wait for a keystroke in the window
    // // if(k == 's')
    // // {
    // //     imwrite("starry_night.png", input_1);
    // // }
    // // #pragma endregion
    
    // // Mat image = input_1;
    // assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramid
    // ORBextractor.ComputePyramid(image);

    // cout << "Created " << nLevels - 1 << " scaled images." << endl;
    // cout << "Exemple of Image 6 : " << endl;

    // #pragma region IMSHOW
    // imshow("Image After ComputePyramid", ORBextractor.mvImagePyramid[6]);
    // int z = waitKey(0); // Wait for a keystroke in the window
    // if(z == 's')
    // {
    //     imwrite("hello.png", image);
    // }
    // #pragma endregion

    // vector < vector<KeyPoint> > allKeypoints;
    // ORBextractor.ComputeKeyPointsOctTree(allKeypoints, descriptors);
    // cout << descriptors.rows << endl;

    #pragma region INIT IMAGE 
    InputArray _image = input_1;
    cv::Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);
    #pragma endregion

    const float W = 30;
    cv::Mat frame;
    cv::VideoCapture cap;
    int deviceID = 0;
    int apiID = 0;
    cap.open(deviceID, apiID);

    std::shared_ptr<SuperPoint> model;

    model = make_shared<SuperPoint>();
    torch::load(model, "superpoint.pt");

    SPDetector detector(model);
    // detector.detect(image, false);
    
    auto slam_epoch = std::chrono::steady_clock::now();

    
    #pragma region Camera [FONCTIONNEL]
    for(;;)
    {
        cap.read(frame);
        cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // font = cv::FONT_HERSHEY_SIMPLEX;

        auto elapsed_time = std::chrono::steady_clock::now() - slam_epoch;
        double frame_timestamp_s = 1 / (elapsed_time.count() / 1000000000.0);
        // std::cout << "FPS: " << std::setprecision(4) << frame_timestamp_s << std::endl;
        
        slam_epoch = std::chrono::steady_clock::now();

        #pragma region cv::Mat SPdetect function
        std::vector<cv::KeyPoint> res;
        std::vector<cv::KeyPoint> vKeysCell;
        cv::Mat descriptors;

        res = detector.SPdetect(model, frame, vKeysCell, descriptors, threshold, true, false);
        // std::cout << res.size() << std::endl;

        for (auto i: res)
        {
            // std::cout << "(" << i.pt.x << "; " << i.pt.y << ")" << std::endl;
            cv::circle(frame, cv::Point(i.pt.x, i.pt.y), 1, cv::Scalar(0, 0, 255), cv::FILLED, 0, 0);
        }

        cv::putText(frame, to_string(frame_timestamp_s), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        cv::imshow("CAMERA", frame);
        if(cv::waitKey(5)>= 0)
        {
            break;
        }

        // cout << "Descriptors = " << endl << " "  << res << endl << endl;

        #pragma endregion
    }
    #pragma endregion

    #pragma region SPdetect for one image [FONCTIONNEL]
    // std::vector<cv::KeyPoint> res;
    // std::vector<cv::KeyPoint> vKeysCell;
    // cv::Mat descriptors;

    // res = detector.SPdetect(model, image, vKeysCell, descriptors, threshold, true, false);
    // std::cout << "TYPE: " << image.type() << std::endl;
    // std::cout << "res.size: " << res.size() << std::endl;

    // for (auto i: res)
    // {
    //     // std::cout << "(" << i.pt.x << "; " << i.pt.y << ")" << std::endl;
    //     cv::circle(image, cv::Point(i.pt.x, i.pt.y), 1, cv::Scalar(0, 0, 255), cv::FILLED, 0, 0);
    // }
    #pragma endregion

    #pragma region ComputeKeyPointsOctTree function in SPextractor
    // std::cout << "image.cols: " << image.cols << ", image.rows: " << image.rows << std::endl; 

    // const int minBorderX = EDGE_THRESHOLD-3;
    // const int minBorderY = minBorderX;
    // const int maxBorderX = image.cols-EDGE_THRESHOLD+3;
    // const int maxBorderY = image.rows-EDGE_THRESHOLD+3;

    // std::cout << "MaxBorderX: " << maxBorderX << ", MaxBorderY: " << maxBorderY << std::endl; 


    // vector<cv::KeyPoint> vToDistributeKeys;
    // vToDistributeKeys.reserve(nFeatures*10);

    // const float width = (maxBorderX-minBorderX);
    // const float height = (maxBorderY-minBorderY);

    // const int nCols = width/W;
    // const int nRows = height/W;
    // const int wCell = ceil(width/nCols);
    // const int hCell = ceil(height/nRows);

    // std::cout << "nCols: " << nCols << ", nRows: " << nRows << std::endl; 

    
    // for(int i=0; i<nRows; i++)
    // {
    //     const float iniY =minBorderY+i*hCell;
    //     float maxY = iniY+hCell+6;

    //     if(iniY>=maxBorderY-3)
    //         continue;
    //     if(maxY>maxBorderY)
    //         maxY = maxBorderY;

    //     for(int j=0; j<nCols; j++)
    //     {
    //         const float iniX =minBorderX+j*wCell;
    //         float maxX = iniX+wCell+6;
    //         if(iniX>=maxBorderX-6)
    //             continue;
    //         if(maxX>maxBorderX)
    //             maxX = maxBorderX;

    //         vector<cv::KeyPoint> vKeysCell;
    //         detector.getKeyPoints(0, iniX, maxX, iniY, maxY, vKeysCell, true);

    //         // if(vKeysCell.empty())
    //         // {
    //         //     // FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
    //         //     //      vKeysCell,minThFAST,true);
    //         //     detector.getKeyPoints(0, iniX, maxX, iniY, maxY, vKeysCell, true);
    //         // }

    //         if(!vKeysCell.empty())
    //         {
    //             for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
    //             {
    //                 (*vit).pt.x+=j*wCell;
    //                 (*vit).pt.y+=i*hCell;
    //                 vToDistributeKeys.push_back(*vit);
    //             }
    //         }
    //     }
    // }

    // vector<cv::KeyPoint> vKeysCell;
    // detector.getKeyPoints(0, 0, 1024, 0, 768, vKeysCell, true);

    // // std::cout << "vKeysCell: " << vKeysCell.size() << std::endl; 

    // for (auto i: vKeysCell)
    //     // std::cout << "(" << i.pt.x << "; " << i.pt.y << ")" << std::endl;
    //     cv::circle(image, cv::Point(i.pt.x, i.pt.y), 5, cv::Scalar(0, 0, 255), cv::FILLED, 0, 0);

    // exit(EXIT_FAILURE);
    #pragma endregion

    #pragma region SPextractor ZONE
    // int nkeypoints = 0;
    // for (int level = 0; level < nLevels; ++level)
    // {
    //     nkeypoints += (int)allKeypoints[level].size();
        
    //     cout << nkeypoints << endl;
    // }
    // if( nkeypoints == 0 )
    // {
    //     _descriptors.release();
    // }
    // else
    // {
    //     _descriptors.create(nkeypoints, 256, CV_32F);
    //     descriptors.copyTo(_descriptors.getMat());
    // }

    // ORBextractor.mvKeys.clear();
    // ORBextractor.mvKeys.reserve(nkeypoints);

    // cout << "Descriptors rows: " << descriptors.rows << endl;


    // int offset = 0;
    // for (int level = 0; level < nLevels; ++level)
    // {
    //     vector<KeyPoint>& keypoints = allKeypoints[level];
    //     int nkeypointsLevel = (int)keypoints.size();

    //     if(nkeypointsLevel==0)
    //         continue;

    //     // // preprocess the resized image
    //     // Mat workingMat = mvImagePyramid[level].clone();
    //     // GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

    //     // // Compute the descriptors
    //     // Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
    //     // computeDescriptors(workingMat, keypoints, desc, pattern);

    //     // offset += nkeypointsLevel;

    //     // Scale keypoint coordinates
    //     if (level != 0)
    //     {
    //         float scale = ORBextractor.mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
    //         for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
    //              keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    //             keypoint->pt *= scale;
    //     }
    //     // And add the keypoints to the output
    //     ORBextractor.mvKeys.insert(ORBextractor.mvKeys.end(), keypoints.begin(), keypoints.end());
    // }
    
    // for (auto i: ORBextractor.mvKeys)
    //     // std::cout << "(" << i.pt.x << "; " << i.pt.y << ")" << std::endl;
    //     cv::circle(image, cv::Point(i.pt.x, i.pt.y), 5, cv::Scalar(0, 0, 255), cv::FILLED, 0, 0);

    // // cout << "mvKeys (x, y): " << ORBextractor.mvKeys[0].pt.x << ", " << ORBextractor.mvKeys[0].pt.y << endl;
    #pragma endregion

    #pragma region IMSHOW [FONCTIONNEL]
    cv::imshow("Display window", image);
    int f = waitKey(0); // Wait for a keystroke in the window
    if(f == 's')
    {
        cv::imwrite("starry_night.png", input_1);
    }
    #pragma endregion
    
    return 0;
}