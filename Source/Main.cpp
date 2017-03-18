// ConsoleApplication1.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include "stdafx.h"
#include "EyecentreLocalisation.h"
#include <ctime>

#ifndef ADD_H_INCLUDED
#define ADD_H_INCLUDED
std::vector<std::string> seglist(std::string text, char seg) {
	std::stringstream test(text);
	std::string segment;
	std::vector<std::string> seglist;

	while (std::getline(test, segment, seg))
	{
		seglist.push_back(segment);
	}
	return seglist;
};
#endif


using namespace std;
using namespace cv;

//string filePath = "../carlton.jpg";
string filePath = "../BioID-FaceDatabase-V1.2/BioID_";
int _tmain(int argc, _TCHAR* argv[])
{
	std::cout << "0 Webcam" << endl;
	std::cout << "1 BioID FaceDB" << endl;
	int in;
	cin >> in;

	if (in == 0) 
	{
		Mat image;
		VideoCapture stream(0);
		EyecentreLocalisation* alg = new EyecentreLocalisation();
		while (true) {
			stream.read(image);
			if (image.data) {
				//imshow("test", image);

				cvtColor(image, image, CV_BGR2GRAY);

				vector<Vec3f> eyes = alg->getEyecentre(image);

				if (eyes.size() == 2) {
					if (eyes[0][2] != 0) {
						circle(image, Point(eyes[0][0], eyes[0][1]), eyes[0][2], Scalar(255, 255, 0), 1);
					}
					if (eyes[1][2] != 0) {
						circle(image, Point(eyes[1][0], eyes[1][1]), eyes[1][2], Scalar(255, 255, 0), 1);
					}
					imshow("Eyes", image);
				}
			}
			if (cv::waitKey(10) == 27) {
				break;
			}
		}
		return 0;
	}
	else if (in == 1) 
	{
		EyecentreLocalisation* alg = new EyecentreLocalisation();

		int sumError1 = 0, sumError2 = 0, sumError3 = 0, sumError4 = 0, sumError5 = 0;
		int sumSuccess = 0;
		float sum_sek_c = 0;
		char * buffer;
		long size;
		std::string number;
		for (int i = 0; i <= 1520; i++) {

			std::cout << "Bild " << i << endl;
			if (i >= 1000) {
				number = to_string(i);
			}
			if (i < 1000) {
				number = "0" + to_string(i);
			}
			if (i < 100) {
				number = "00" + to_string(i);
			}
			if (i < 10) {
				number = "000" + to_string(i);
			}
			if (i == 0) {
				number = "0000";
			}
			ifstream file(filePath + number + ".eye", ios::in | ios::binary | ios::ate);
			size = file.tellg();
			file.seekg(0, ios::beg);
			buffer = new char[size];
			file.read(buffer, size);
			file.close();
			std::vector<std::string> a = seglist(buffer, '\r');
			a = seglist(a[1], '\n');
			a = seglist(a[1], '\t');

			int leftX = stoi(a[0]);
			int leftY = stoi(a[1]);
			int rightX = stoi(a[2]);
			int rightY = stoi(a[3]);

			float distance = sqrt(pow(abs(leftX - rightX), 2) + pow(abs(leftY - rightY), 2));

			Mat image = imread(filePath + number + ".pgm", IMREAD_GRAYSCALE);

			clock_t start_c;
			float sek_c;
			
			time_t start_t, ende_t;
			float sek_t;

			time(&start_t);
			start_c = clock();

			vector<Vec3f> eyes = alg->getEyecentre(image);

			sek_c = (float)(clock() - start_c) / CLOCKS_PER_SEC;
			time(&ende_t);
			sek_t = (float)(ende_t - start_t);


			if (eyes.size() == 2) {
				if (eyes[0][2] != 0 && eyes[1][2] != 0) {

					sumSuccess++;
					sum_sek_c += sek_c;

					int algLeftX = eyes[1][0];
					int algLeftY = eyes[1][1];
					int algRightX = eyes[0][0];
					int algRightY = eyes[0][1];

					float leftDistance = sqrt(pow(abs(leftX - algLeftX), 2) + pow(abs(leftY - algLeftY), 2));
					float rightDistance = sqrt(pow(abs(algRightX - rightX), 2) + pow(abs(algRightY - rightY), 2));
					float maxDistance = max(leftDistance, rightDistance);

					float error = maxDistance / distance;

					if (error <= 0.25) {
						sumError5++;
					}
					if (error <= 0.2) {
						sumError4++;
					}
					if (error <= 0.15) {
						sumError3++;
					}
					if (error <= 0.1) {
						sumError2++;
					}
					if (error <= 0.05) {
						sumError1++;
					}
				}
			}

		}
		sum_sek_c = sum_sek_c / sumSuccess;
		//std::cout << "Durchschnittliche Laufzeit: " << sum_sek_c << endl;
		std::cout << "Error <=0.05: " << sumError1 << endl;
		std::cout << "Error <=0.10: " << sumError2 << endl;
		std::cout << "Error <=0.15: " << sumError3 << endl;
		std::cout << "Error <=0.20: " << sumError4 << endl;
		std::cout << "Error <=0.25: " << sumError5 << endl;
		std::cout << "Success: " << sumSuccess << endl;

		cv::waitKey(500000);
		return 0;
	}
}


