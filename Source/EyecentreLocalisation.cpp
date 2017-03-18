#include "stdafx.h"
#include "EyecentreLocalisation.h"


using namespace cv;
using namespace std;
Mat original;
Mat gaussOrig;
CascadeClassifier faceClassifier;
double eyeCropPortion = 0.2;
int binaryThreshold = 75;
int xOffset, yOffset;
int xRightoffset, yRightOffset;
EyecentreLocalisation::EyecentreLocalisation()
{
	
}

vector<Vec3f> EyecentreLocalisation::getEyecentre(Mat image) {
	original = image.clone();
	if (original.data) {

		//Gaussian BLur

		std::vector<cv::Rect, std::allocator<cv::Rect>> faces;
		faces.clear();
		vector<Rect> f;
		Mat gaussianBlur;
		int derivation = 0.05*original.cols;
		GaussianBlur(original, gaussianBlur, Size(5, 5), derivation, derivation);

		//Viola Jones
		if (faceClassifier.load("../haarcascades/haarcascade_frontalface_alt.xml"))
		{
			faceClassifier.detectMultiScale(gaussianBlur, faces, 1.05, 3, CV_HAAR_FIND_BIGGEST_OBJECT);
			if (faces.size() > 0) {
				Rect face = faces[0];
				xOffset = face.x;
				yOffset = face.y;

				gaussianBlur = gaussianBlur(face);

				int pixelCut = 0.2*gaussianBlur.rows;
				Rect crop(0, 0.2*gaussianBlur.rows, gaussianBlur.cols, gaussianBlur.rows - 2*pixelCut);

				yOffset += pixelCut;

				gaussianBlur = gaussianBlur(crop);

				xRightoffset = xOffset;
				yRightOffset = yOffset;
				int middle = 0.5*gaussianBlur.cols;
				Rect left(0, 0, middle, gaussianBlur.rows);
				Rect right(middle, 0, gaussianBlur.cols - middle, gaussianBlur.rows);

				xRightoffset += middle;
				Mat leftPic = gaussianBlur(left);
				Mat rightPic = gaussianBlur(right);

				if (faceClassifier.load("../haarcascades/haarcascade_eye.xml")) {
					Vec3f leftIris = getIris(leftPic, &xOffset, &yOffset);
					Vec3f rightIris = getIris(rightPic, &xRightoffset, &yRightOffset);

					vector<Vec3f> result = { leftIris, rightIris };
					return result;
				}
			}
			return vector<Vec3f>();
		}
		return vector<Vec3f>();
	}
	return vector<Vec3f>();

}

Vec3f EyecentreLocalisation::getIris(Mat image, int* x, int* y) {
	vector<Rect> eyes;
	faceClassifier.detectMultiScale(image, eyes, 1.5, 3);
	if (eyes.size() > 0 && x  && y ) {
		int xTemp = *x;
		int yTemp = *y;

		Rect eye = eyes[0];
		xTemp += eye.x;
		yTemp += eye.y;
		Mat eyeFound = image(eye);

		Rect crop(0, eyeCropPortion*eyeFound.rows, eyeFound.cols, eyeFound.rows-eyeCropPortion*eyeFound.rows);
		yTemp += eyeCropPortion*eyeFound.rows;
		Mat eyeCropped = eyeFound(crop);

		equalizeHist(eyeCropped, eyeCropped);

		threshold(eyeCropped, eyeCropped, binaryThreshold, 255, 0);

		Mat kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(5,5));

		morphologyEx(eyeCropped, eyeCropped, 2, kernel);

		vector<Vec3f> circles;
		HoughCircles(eyeCropped, circles, CV_HOUGH_GRADIENT, 1, 10, 100.0, 9, 6, 100);
		if (circles.size() > 0) {
			Mat withCircle;
			eyeCropped.copyTo(withCircle);
			Mat withCircleRGB(withCircle.size(), CV_8UC3);
			cvtColor(withCircle, withCircleRGB, CV_GRAY2RGB);
			Vec3f c = circles[0];
			Point center(c[0], c[1]);
			float radius = c[2];
			circle(withCircleRGB, center, radius, Scalar(0,0,255),1);

			c[0] += xTemp;
			c[1] += yTemp;
			circle(original, Point(c[0], c[1]), radius, Scalar(255,255,0), 1);

			int r = floor(radius + 0.5);
			int rSmall = r / 2;
			Point minIntensity(c[0], c[1]);
			int min = INT32_MAX;
			int xIndex = c[0];
			int yIndex = c[1];
			for (int i = c[0] - r; i < c[0] + r; i++) {
				for (int j = c[1] - r; j < c[1] + r; j++) {

					float difference = sqrt(pow(abs(xIndex - i), 2) + pow(abs(yIndex - j), 2));
					if (difference <= radius) {
						int sum = 0;
						for (int k = i - rSmall; k < i + rSmall; k++) {
							for (int l = j - rSmall; l < j + rSmall; l++) {

								difference = sqrt(pow(abs(xIndex - k), 2) + pow(abs(yIndex - l), 2));
								if (difference > radius) {
									sum += 255;
								}

								difference = sqrt(pow(abs(i - k), 2) + pow(abs(j - l), 2));
								if (difference <= radius/2) {

									difference = sqrt(pow(abs(xIndex - k), 2) + pow(abs(yIndex - l), 2));
									if (difference > radius) {
										sum += 255;
									}
									else {
										if (k < original.rows && l < original.cols && k>=0 && l>= 0) {
											sum += original.at<uchar>(k, l);
										}
									}
								}
							}
						}

						if (sum < min) {
							min = sum; 
							minIntensity.x = i;
							minIntensity.y = j;
						}
					}
				}
			}
			return Vec3f(minIntensity.x, minIntensity.y, radius*0.5);
			Mat originalRGB(original.size(), CV_8UC3);
			cvtColor(original, originalRGB, CV_GRAY2RGB);
			circle(originalRGB, minIntensity, radius*0.5, Scalar(0, 0, 255), 1);
		}
		return Vec3f(0,0,0);
	}
	return Vec3f(0,0,0);
}

void EyecentreLocalisation::showHist(Mat img) {
	Mat hist;
	float hranges[] = { 0,256 };
	const float *ranges[] = { hranges };
	int channels[] = { 0 };
	int histSize[] = { 256 };
	int nImages = 1;
	int dims = 1;

	calcHist(&img, nImages, channels, Mat(), hist, dims, histSize, ranges, true);

	int h = 200;
	int w = 256;
	int b = cvRound((double)w / 256);
	Mat histImg(h, w, original.type(), Scalar(0,0,0));
	normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < 256; i++)
	{
		line(histImg, Point(b*(i - 1), h - cvRound(hist.at<float>(i - 1))),
			Point(b*(i), h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	imshow("Histogramm", histImg);

	cout << hist << endl;
}

//IplImage* EyecentreLocalisation::drawHist(CvHistogram *hist) {
//	float histMax = 0; 
//	cvGetMinMaxHistValue(hist, 0, &histMax, 0, 0); 
//	IplImage* imgHist = cvCreateImage(cvSize(256 * scaleX, 64 * scaleY), 8, 1); 
//	cvZero(imgHist);
//	for (int i = 0; i<255; i++) 
//	{ 
//		float histValue = cvGetReal1D(hist, i);
//		float nextValue = cvGetReal1D(hist, i + 1);
//		CvPoint pt1 = cvPoint(i*scaleX, 64 * scaleY); 
//		CvPoint pt2 = cvPoint(i*scaleX + scaleX, 64 * scaleY); 
//		CvPoint pt3 = cvPoint(i*scaleX + scaleX, (64 - nextValue * 64 / histMax)*scaleY); 
//		CvPoint pt4 = cvPoint(i*scaleX, (64 - histValue * 64 / histMax)*scaleY); 
//		int numPts = 5; CvPoint pts[] = { pt1, pt2, pt3, pt4, pt1 }; 
//		cvFillConvexPoly(imgHist, pts, numPts, cvScalar(255)); 
//	}
//	return imgHist;
//}


EyecentreLocalisation::~EyecentreLocalisation()
{
}
