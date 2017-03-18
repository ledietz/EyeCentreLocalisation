#pragma once
class EyecentreLocalisation
{
public:
	EyecentreLocalisation();
	~EyecentreLocalisation();
	cv::Vec3f getIris(cv::Mat image, int* x, int* y);
	std::vector<cv::Vec3f> getEyecentre(cv::Mat image);
	void showHist(cv::Mat img);
	IplImage* drawHist(CvHistogram *hist, float scaleX = 1, float scaleY = 1);
};

