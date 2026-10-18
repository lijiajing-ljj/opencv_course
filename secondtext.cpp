#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace cv;
int main() {
    int COLS = 11;    
   int ROWS = 8;           
    double SQUARE_MM = 50.0;
    string IMG_DIR = "D:/openCV图片/calibration_images"; 
   string OUT_FILE = "camera.yml";

    vector<Point3f> objp;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            objp.push_back(cv::Point3f((float)(c * SQUARE_MM), (float)(r * SQUARE_MM), 0));
    vector<string> files, tmp;
    for (const char* ext : { "*.jpg", "*.jpeg", "*.png", "*.bmp",
                             "*.JPG", "*.JPEG", "*.PNG", "*.BMP" }) {
        tmp.clear();
        try {
            cv::glob(IMG_DIR + "/" + ext, tmp, false);
        } catch (const cv::Exception& e) {
            std::cerr << "glob 扫描出错: " << e.what() << std::endl;
        }
        files.insert(files.end(), tmp.begin(), tmp.end());
    }
    if (files.empty()) {
        std::cerr << "错误: 在 " << IMG_DIR << " 下没找到任何图片，请检查路径是否正确\n";
        return -1;
    }
    cout << "找到图片 " << files.size() << " 张\n";

    vector<vector<Point3f>> objPoints;
    vector<vector<Point2f>> imgPoints;
    Size imgSize;
    string firstImg;
    for (const auto& f : files) {
        Mat img = imread(f);
        if (img.empty()) continue;
        if (imgSize == Size()) { imgSize = img.size(); firstImg = f; }
       Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, Size(COLS, ROWS), corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
        if (!found) continue;
        cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
            TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));
        objPoints.push_back(objp);
        imgPoints.push_back(corners);
    }
    cout << "成功提取角点 " << objPoints.size() << " / " << files.size() << " 张\n";
    if (objPoints.size() < 10) { std::cerr << "有效图片太少，检查内点数是否填对\n"; return -1; }

    Mat K, dist;
    vector<Mat> rvecs, tvecs;
    double rmse =calibrateCamera(objPoints, imgPoints, imgSize, K, dist, rvecs, tvecs);//核心函数
    cout << "图像尺寸: " << imgSize << "\n";
    cout << "重投影误差 RMSE = " << rmse << " px\n";
    cout << "内参 K =\n" << K << "\n";
    cout << "畸变 dist = " << dist.t() << "\n";


    cv::FileStorage fs(OUT_FILE, cv::FileStorage::WRITE);
    fs << "K" << K << "dist" << dist << "imgW" << imgSize.width << "imgH" << imgSize.height;
    fs.release();
    std::cout << "已保存标定结果到 " << OUT_FILE << "\n";
    cv::Mat sample = imread(firstImg);
    if (!sample.empty()) {
        cv::Mat undist;
        undistort(sample, undist, K, dist);
        imwrite("undistort_before.jpg", sample);
        imwrite("undistort_after.jpg", undist);
        cout << "已保存去畸变对比图 undistort_before/after.jpg\n";
    }
    return 0;
}