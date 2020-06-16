
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp> // 헤더 파일로 쓰려면 cpp파일을 hpp파일로 심플한 헤더 파일 타입으로 변경해줘야 함 
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{  
    // load calibration data in OpenCV matirces and store calibration data in OpenCV matrices (이미 로딩했기 때문에 과제 11번 코드와 다르게 여기선 필요없음)
    // cv::Mat P_rect_xx(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    // cv::Mat R_rect_xx(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    // cv::Mat RT(4,4,cv::DataType<double>::type);        // rotation matrix and translation vector
    // loadCalibrationData(P_rect_xx, R_rect_xx, RT);     // 코드를 심플하게 만들기 위해 캘리브레이션 데이터를 로드하는 펑션  


    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    /* 총 2개의 for문 (1st 전체 포인트에 대한 loop)  * 2nd 바운딩박스 안에 있는 포인트에 대한 loop) */
    // 1st for문
    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)    // loop over all key points and over all bounding boxes
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);   // pixel coordinates which we recieved after reconverting 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);   // into Euclidean coordinates by simply dividing them by third component here

        double shrinkFactor = 0.10;                           // 0~1 사이 value (0은 오리지날 사이즈, 1은 100% shrink하는 것을 의미함)
        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose current Lidar point(a vector which hold iterator over bounding boxes)                       
        // the idea is to check for each point whether it has been enclosed by one or multiple boxes
       
        // 2nd for문 (shrink current bounding box -> 포인트가 current box에 포함되는지 확인하고 저장)
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) // 위 for문과 다르게 바운딩 박스 안에 있는 포인트만 loop
        {   
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;    // smallerBox 선언 (to shrink each bounding box which we get by iterating)
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);
            
            // check wether point is within current bounding box
            if (smallerBox.contains(pt))      // simply check if the smallerBox conatins all the point or not
            {   // 바운딩 박스에 있으면 enclosingBoxes[0]에 pushback               
                enclosingBoxes.push_back(it2);  //it2 주소에 저장된 포인트를 바운딩 박스에 속하도록 라벨링 하고 'enclosingBoxes'에 저장(?)
                // do not simply store this point & label it as belonging to this bounding box but push back this bounding box 
                // which is currently addressed by the iterator it2 into a vector which is called eclosing boxes('enclosingBoxes')
            }   
         } // eof loop over all Lidar points 


        /* check for each point whether it has been enclosed by one or multiple boxes (여기 if문은 모든 바운딩박스에 대해 루프를 돌고 난 뒤에 도달) */
        // ex) 만약 3개의 바운딩 박스에 의해 둘러쌓여 있다면 위 'enclosingBoxes'에 3개의 바운딩 박스 이터레이터가 저장됬을 것이고 'enclosingBoxes' size는 3일 것이다 
        // 즉. 포인트 하나가 바운딩 박스 3개에 포함된 상태 -> 이를 해결하기 위해 싱글 바운딩 박스에 keeping

        // mulipe -> single bounding box 처리
        if (enclosingBoxes.size() == 1)   
        {   // add lidar point to bounding box
            enclosingBoxes[0] -> lidarPoints.push_back(*it1);  // simply push back this lidar point which is denoted by 'it1' into this bounding box
            // enclosingBoxes 벡터는 1개의 value만 가지고 있기 때문에 enclosingBoxes[0]으로 표현 
        } 
    } // eof loop over all Lidar points 
}



void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{                       // BoundingBox 2D & 3D
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));   // CV_8UC3 (specific size of 3 channel), using white background

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);          // initialize the random nember generator(rng) with a unique value
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));  // 0~150 사이 random number generate(rng)

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;

        // loop over all lidar points
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates of the lidar sensor 
            float xw = (*it2).x;             // world position in m with x facing forward from sensor
            float yw = (*it2).y;             // world position in m with y facing left from sensor
            // stor min & max value
            xwmin = xwmin<xw ? xwmin : xw;   // store minimum value of lidar points in driving direction in x direction
            ywmin = ywmin<yw ? ywmin : yw;   // store minimum value of lidar points in y direction 
            ywmax = ywmax>yw ? ywmax : yw;   // 조건식 ? 반환 1(true) : 반환 2(false)

            // top-view coordinates (top view라 z value가 없는 듯)
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle (not only draw lidar point cloud & also want to draw an enclosing frame around it)
            top = top<y ? top : y;            // 조건식 ? 반환 1(true) : 반환 2(false) 
            left = left<x ? left : x;       
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point as circle using randomize color 'currColor'
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200]; // 문자열 포함 갯수 정의
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size()); // id of box(number of points which have been associated to lidar point)
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);             // minimum distance in driving direction, ..., width of the object
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);    // 행렬 내 특정 위치에 원하는 글자를 써서 영상(이미지)에 표시
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)    // bWait flag is simply used for augmenting a display with several pieces of information
    {            // so if want to display our 3d object -> bWait would be true
        cv::waitKey(0); // wait for key to be pressed
    }
}


/* student assignment */

// Associate a given bounding box with the keypoints it contains (idea here is to cluster all the keypoint matches which are within this bounding box)
// input -> loop over all bounding box (if 5개라면 'clusterKptMatchesWithROI' 펑션이 5번 호출됨) & 리절트 벡터는 없어 리턴 안됨지만 will augment the bounding box with our finding
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{  // input -> respective bounding box, a vector of keypoint in previous frame, a vector of keypoint in current frame, a set of keypoint matches
   // bounding box 개수(by find bounding box function)에 따라 loop가 진행됨 ex) n개면 loop가 n번 실행되는 개념 
   // idear is to cluster all keypoints matches which are within this bounding box (모든 키포인트들이 fit or not 여부에 따라) -> augment bounding box with finding 
    
    // Loop over all matches in the current frame
    for (cv::DMatch match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            boundingBox.kptMatches.push_back(match);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images (앞에 과제 그대로 사용)
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)  // '*visImg' -> optional visualzation image
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios;         // stores the distance ratios for all keypoints between curr. and prev. frame

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);        // 현재(이후) frame의 it1번째 index(trainIdx) keypoint를 새로 선언한 kpOuterCurr에 저장 
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);        // 이전 frame의 it1번째 index(queryIdx) keypoint를 새로 선언한 kpOuterPrev에 저장

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)   // kptMatches는 위에서 <cv::DMatch> 타입으로 선언  
        { // inner kpt.-loop                                          // <cv::DMatch> : matching keypoints between images are packaged into an OpenCV data structure

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);    // at : 문자열에서 특정위치의 문자를 엑세스  ex) at(3) -> 3번째 문자열에 엑세스?
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);     // curr frame에서의 keypoint 사이 거리(기준점이 inner, 피계산되는 점이 outer)
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);     // pre frame에서의 keypoint 사이 거리(기준점이 inner, 피계산되는 점이 outer)

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // end of inner loop over all matched kpts
    }     // end of outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);         // medIndex를 계산하여 평균이 아닌 가운데 value를 추출 (미디언 구할때 짝수인 경우와 홀수인 경우 다르기 때문에 이를 고려해야 함 -> 아래 코드가 해담됨)
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // End OF STUDENT TASK
}

/* // 아래 코드는 TTC lidar value가 정확하지 않음 
// Compute time-to-collision (TTC) based on relevant lidar points
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
// take reference to lidar point in previous frame, current frame in the respect bouning box and take the frame rate -> and return estimate to 'double &TTC'
{
    // auxiliary variables
    double dT = 0.1 / frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0;             // assumed width of the ego lane(내 차선 너비) -> 이 4m를 기준으로 좌우측 포인트 제거됨 

    // find closest distance to Lidar points within ego lane (이고 레인으로 하는 이유는 거리를 빠르고 정확하게 계산하기 위함)
    vector<double> xPrev, xCurr;     // vector 클래스는 모든 자료형 저장
    // double xPrev, xCurr;          // 에러 발생

    // find Lidar points within ego lane
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        // previous value
        if (abs(it->y) <= laneWidth / 2.0)   // y축 거리가차선 절반보다 작은 경우 
        { // 3D point within ego lane?       -> 즉 에고 차선 안에 있다면 
            xPrev.push_back(it->x);
        } // 삼항 연산자 : 조건식(minXPrev > it->x) ? value 1(it->x) : value 2(minXPrev)
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        // current value  
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            xCurr.push_back(it->x);
        }
    }

    double minXPrev = 0;
    double minXCurr = 0;
    if (xPrev.size() > 0)
    {
       for (auto x: xPrev)
            minXPrev += x;
       minXPrev = minXPrev / xPrev.size();
    }
    if (xCurr.size() > 0)
    {
       for (auto x: xCurr)
           minXCurr += x;
       minXCurr = minXCurr / xCurr.size();
    }

    // compute TTC from both measurements
    cout << "minXCurr: " << minXCurr << endl;
    cout << "minXPrev: " << minXPrev << endl;
    TTC = minXCurr * dT / (minXPrev - minXCurr); // Distance compute by previous & current
}   */

/* 위 정확하지 않아 아래 헬퍼 평션과 새로운 TTC_lidar 코드 적용 (2개가 함께 있어야 실행됨) */
// Helper function to sort lidar points based on their X (longitudinal) coordinate
void sortLidarPointsX(std::vector<LidarPoint> &lidarPoints)
{
    // This std::sort with a lambda mutates lidarPoints, a vector of LidarPoint
    std::sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint a, LidarPoint b) {
        return a.x < b.x;  // Sort ascending on the x coordinate only
    });
}

// Compute time-to-collision (TTC) based on relevant lidar points
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // In each frame, take the median x-distance as our more robust estimate.
    // If performance is suffering, consider taking the median of a random subset of the points.
    sortLidarPointsX(lidarPointsPrev);
    sortLidarPointsX(lidarPointsCurr);
    double d0 = lidarPointsPrev[lidarPointsPrev.size()/2].x;
    double d1 = lidarPointsCurr[lidarPointsCurr.size()/2].x;

    // Using the constant-velocity model (as opposed to a constant-acceleration model)
    // TTC = d1 * delta_t / (d0 - d1)
    // where: d0 is the previous frame's closing distance (front-to-rear bumper)
    //        d1 is the current frame's closing distance (front-to-rear bumper)
    //        delta_t is the time elapsed between images (1 / frameRate)
    // Note: this function does not take into account the distance from the lidar origin to the front bumper of our vehicle.
    //       It also does not account for the curvature or protrusions from the rear bumper of the preceding vehicle.
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);
}



// 'cv::DescriptorMatcher::match' function -> each DMatch
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // contains two keypoint indices (queryIdx and trainIdx)
    // -> queryIdx = index into one set of keypoints & trainIdx = index into the other set of keypoints
    std::multimap<int, int> mmap {};  // map 컨테이너에 중복 원소(key)를 허용하는 경우에 사용 -> std::multimap<int, int> 첫번째 변수가 key, 두번째 변수가 value임)
    int maxPrevBoxID = 0;

    for (auto match : matches)
    {
        cv::KeyPoint prevKp = prevFrame.keypoints[match.queryIdx];     // prevFrame.keypoints is indexed by queryIdx
        cv::KeyPoint currKp = currFrame.keypoints[match.trainIdx];     // currFrame.keypoints is indexed by trainIdx
        
        int prevBoxID = -1;
        int currBoxID = -1;

        // For each bounding box in the previous frame
        for (auto bbox : prevFrame.boundingBoxes)        // auto로 자동으로 변수를 인식하고 range에 대한 loop를 진행 in previous image (element : container)
        {
            if (bbox.roi.contains(prevKp.pt))            // ()내 문자열이 포함되어 있다면 true, 없다면 false  위에서 선언한 'prevKp'를 ROI예 포함되어 있으면 
               prevBoxID = bbox.boxID;
        }

        // For each bounding box in the current frame
        for (auto bbox : currFrame.boundingBoxes)        // auto로 자동으로 변수를 인식하고 range에 대한 loop를 진행 in current image  
        {
            if (bbox.roi.contains(currKp.pt)) 
               currBoxID = bbox.boxID;
        }
        
        // Amultimap에 containing boxID for each match를 추가 
        mmap.insert({currBoxID, prevBoxID});

        maxPrevBoxID = std::max(maxPrevBoxID, prevBoxID);    // maxPrevBoxID(초기 0)와 prevBoxID 중 최대 value가 maxPrevBoxID로 저장됨 
    }

    // Setup a list of boxID int values to iterate over in the current frame
    vector<int> currFrameBoxIDs {};

    // 현재 이미지에서 each boxID에 대해 loop 진행한 뒤 -> 이전 이미지에 적용하기 위해 (boxID와 연관된) 가장 frequent한 value을 찾아냄 
    for (auto box : currFrame.boundingBoxes) currFrameBoxIDs.push_back(box.boxID);

    // Count the greatest number of matches in the multimap, where each element is {key=currBoxID, val=prevBoxID}
    for (int k : currFrameBoxIDs)   // 위에서 선언되고 push_back된 'currFrameBoxIDs'에 대해 loop 진행
    {        
        auto rangePrevBoxIDs = mmap.equal_range(k);   // mm.equal_range(key) -> key에 해당하는 원소의 범위(range)를 pair 객체(first, second)로 반환
        // ex) equal_range(k)는 key인 k와 매칭되는 모든 elements에 대한 range를 return한다는 의미

        // Create a vector of counts (per current bbox) of prevBoxIDs
        std::vector<int> counts(maxPrevBoxID + 1, 0);

        // Accumulator loop (위에서 정해진 pair 타입(시작, 끝)의 range에 대해 loop)
        for (auto it = rangePrevBoxIDs.first; it != rangePrevBoxIDs.second; ++it)      // .first & .second는 위에서 반환된 range \ != means not equal
        {   // for (초기식; 조건식; 변화식) -> 따라사 위 뜻은 초기it = rangePrevBoxIDs.first가 조건 it != rangePrevBoxIDs.second과 not equal인 경우 +1 한다는 뜻)
            if (-1 != (*it).second)           // loop문 내에서 두번째 value가 -1이 아닌 경우 두번째 value에 1씩 더해줌 
                counts[(*it).second] += 1;
        }

        // Get the index of the maximum count (the mode) of the previous frame's boxID
        int modeIndex = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));  // counts the number of maximum index 

        // Set best matching bounding box map with key & value (key = Previous frame's most likely matching boxID, value = Current frame's boxID, k)
        bbBestMatches.insert({modeIndex, k});
    }
} 