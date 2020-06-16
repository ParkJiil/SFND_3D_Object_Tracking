/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"      // 헤더파일 (h, hpp)으로 먼저 코드 만든 뒤 나중에 헤더파일로 불러서 해당 펑션들을 c, CPP로 만들어줘야 함 
#include "matching2D.hpp"        // -> 그런 다음 메인 함수에서는 헤더파일 (h, hpp)을 including하고 코드를 짜면 됨
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"          

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";           // .. 하면 상위 폴더로 이동 (cd ..)

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, 02는 color(01은 그레이 스케일 in mid-term)
    string imgFileType = ".png";
    int imgStartIndex = 0;   // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;    // last file index to load
    int imgStepWidth = 1;    // using to decrease frame rate (current frame rate = 10초 and 'imgStepWidth = 1' means using every image)
    int imgFillWidth = 4;    // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection (specify the base parameter of yolo)
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar point data 
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar (from kitti calibration data file)
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type);  // intrinsic -> 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type);  // ratation -> 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type);         // extrinsic -> rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // miscellaneous variable (기타 잡다한 변수들)
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera ('imgStepWidth'로 컨트롤 가능)
    int dataBufferSize = 2;                       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer;                 // list of data frames which are held in memory at the same time
    bool bVis = false;                             // visualize results (보통 false로 하고 one by one으로 체크하고자 하는 부분이 있을 때 true 설정)
    // student assignment 
    bool comp = true;                             // for print the comparing results between TTC camera & lidar
 
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);  // 미드텀과 다른 부분 (그레이로 컨버트하지 않고 라인 95에서 프레임 자체를 칼라 이미지로 push back 처리)

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        // cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;    // 완료시 출력됨


        /* DETECT & CLASSIFY OBJECTS */
        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        // perform the YOLO based object detection below (아웃풋으로 디텍션 된 오브젝트에 바운딩 박스가 그려진 결과가 나옴)
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        // cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */
        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties               // R은 reflectivity (0 ~ 1) 0이라고 반사되지 않는게 아니고 낮은 intensity value가 리턴됨
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane (to compute TTC)
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);                // crop the space (너무 심플한 코드라 road surface에 따라 에러 ex 방지턱)
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;          // push 'lidarPoints' into 'dataBuffer' 

        // cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */
        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);


        // Visualize 3D objects (1st visualization)
        bVis = false;              // false는 그냥 패싱하고 이미지 결과만 보여줌 (true로 설정하면 3d object 결과 별도의 창에 display)
        if(bVis)
        {   // 3D (white topview persepective)
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

        // cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        // until this line we create 3D objectf for each timesteo invidually
        
        
        /*  REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT  
        // 현재 루프 몸체의 끝으로 이동(일찍 종료) -> 만약 YOLO 기반 3D object detection을 하거나 라이다 포인트를 클러스터링만 할 경우 'continue;' */
        // continue;         // skips directly to the next image without processing what comes beneath 

        /* DETECT IMAGE KEYPOINTS */
        // convert current image to grayscale
        cv::Mat imgGray;       // keypoint detection은 그레이 스케일 필요 (but YOLO는 칼라이미지로 트레이닝 되었으므로 칼라 이미지 사용 필요)
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        // detector & descriptor type selection
        string detectorType = "ORB";  // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE", SIFT
        string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT 


        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        } 
        else if (detectorType.compare("HARRIS") == 0) // 결국엔 detectorType = "HARRIS"면 detKeypointsHarris를 적용하라는 뜻
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else   // 위 두가지 detectorType이 아닌 경우 detectorType(FAST, BRISK, ORB, AKAZE, SIFT)을 모던 펑션에 넣어 실행 
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }

        
        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;          // 여기선 불필요하니 false로 세팅 
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            // cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        // cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */
        cv::Mat descriptors;

        // string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT (위로 통합)


        // 아래 'descKeypoints' 평션은 inside 'matching2D.hpp' 있음 -> but 미드텀 프로젝트에서 상요한 펑션을 적용해도 됨
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        //cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        // have at least processed two images and this is place where we enter this area 
        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */  // 여기 역시 Descriptors 종류가 모두 포함된 미드텀 프로젝트 코드를 활용해도 됨
            vector<cv::DMatch> matches;

            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            // string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            // string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

            /* For descriptor type, select binary (BINARY) or histogram of gradients (HOG) */
            /* BINARY descriptors include: BRISK, BRIEF, ORB, FREAK, and (A)KAZE. */
            /* HOG descriptors include: SIFT (and SURF and GLOH, all patented). */
            string descriptorCategory {};
            if (0 == descriptorType.compare("SIFT"))     // SIFT가 HOG 타입이기 때문에 추가한 코드
            {  descriptorCategory = "DES_HOG";  }
            else 
            {  descriptorCategory = "DES_BINARY"; }

            /* For selector type, choose nearest neighbors (NN) or k nearest neighbors (KNN) */
            // string selectorType = "SEL_NN";
            string selectorType = "SEL_KNN";

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorCategory, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            // student assignment
            if (comp) 
            {
                cout << " detector = " << detectorType << ",";  
                cout << " descriptor = " << descriptorType << ",";  
                cout << " img #" << imgIndex << ",";
            }

            //cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            
            // until this line, what we do is a set of 3D objects in space, one for each time & have a whole set of keypoint match between 
            // images all over the image (without having looked at a single ROI or bounding box or object yet that has not been done)
            

            /* TRACK 3D OBJECT BOUNDING BOXES  
            // STUDENT ASSIGNMENT
            // TASK FP.1 -> match list of 3D objects(vector<BoundingBox>) between cur & pre frame(implement by 'matchBoundingBoxes'in camfusiohn_student.cpp) */ 
            map<int, int> bbBestMatches;  
            // 'matchBoundingBoxes' function which take input -> previous data buffer and that gives you as output a map which contains 
            // the indesx number of the bounding boxes which have been found to corresponds together between two time steps.
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            // input is 'bbBestMatches' is map structure, that's the result structure which you shold return (라인 238 선언)
            // '*(dataBuffer.end()-2)' is pointer to previous data buffer & '*(dataBuffer.end()-1)' is current data buffer
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;  // bounding box best match

            // cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */
            // loop over all BB match pairs 
            // 위 '(dataBuffer.end()-1)-> bbMatches = bbBestMatches'에서 얻은 결과로 loop
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB; 
                // give you a pointer to the bounding box in the current frame, the first match pair in the current frame     
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) 
                { 
                    if (it1->second == it2->boxID)    // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }
                // comparing bounding box IDs to other bounding boxes in our data buffer, we get the previous bounding box 
                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID)    // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }


                /* compute TTC for current match */
                // 2개의 포인터 존재 -> one to a bounding box in the previous frame('prevBB')& 2nd one to a bounding box in current frame('currBB')
                // and both mached using the method you implemented in 'bbMatches' above.
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // 라이다 데이터가 있어야 compute TTC 가능 (if문으로 라이다 데이터 존재여부 확인) 
                {

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    // 'computeTTCLidar' input -> lidar data in previous & lidar data in current bounding box, sensor frame rate, output result)
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar); // in camfusion & studebt 파일
                    //// EOF STUDENT ASSIGNMENT


                    //// STUDENT ASSIGNMENT (consist of 2 step below)
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)

                    double ttcCamera;
                    // need to associate keypoint matches with bounding boxes (현재는 we have a bounding box associated set of lidar points)
                    // and what we need in order to perform TTC based estimation using the camera for each bounding box 
                    // -> we need a set of matched keypoints which are enclosed by a specific bounding box
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
                     // implement cluster keypoint maches w/ ROI (위 라인) -> after then, can concentrate on compute TTC camera (아래 라인)_                 
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);      
                    // basic idea is look at the ROI for certain bounding box in the camera image, look at the keypoint,     
                    // correspondences keypoint matches are enclosed within this ROI, and then take those and associate them to the bounding box
                    // we will use those keypoint matches to find corresponding bounding box in next image to associate two bounding boxes over time

                   // student assignment
                    if (comp)      // () value가 true면 진행 
                    {
                        cout << " ttcLidar= " << ttcLidar << " ";  cout << " ttcCamera = " << ttcCamera <<" ";  cout << " ttc(Camera-Lidar) = " << ttcCamera - ttcLidar << " ";     cout << endl;
                    }
                    //// EOF STUDENT ASSIGNMENT

                    // visualizing the results
                    bVis = true;          // 결과를 이미지로 확인하기 우해서는 true / 시뮬레이션 결과 비교를 위해 결과만 보려면 false
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        // showLidarTopview(currBB->lidarPoints, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg); // look at an overay of lidar point over the time
                        // display the respective ROI (means the detected vdhicle or a truck or whatever)
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        // augment the image with information about the time collision computed from lidar
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);   // 이미지 내 문자 표현
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
                        // display code for the actual 
                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    bVis = false;
                } // eof TTC computation
            } // eof loop over all BB matches            
        }
    } // eof loop over all images
    return 0;
}
