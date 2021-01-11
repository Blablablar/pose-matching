#include "net.h"
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<fstream>

#include"Timer.h"
#include"poseMatching.h"


std::vector<std::pair<cv::Point, float>> getKPloc(ncnn::Mat& m);
std::vector<cv::Point> postProc(ncnn::Mat& m);
float sigmoid(float x);


int main()
{
	std::string testVideo = 
		"./testVideo\\kaihetiao_new1_crop.avi";
	cv::VideoCapture video(testVideo);
	if (!video.isOpened())
	{
		std::cout << "error! video can't open!" << std::endl;
	}
	int w = video.get(3);
	int h = video.get(4);

	std::string modelPath = "./model\\";
	std::string modelName = "model_torch_256x192_heatmap_sim8";
	std::string paramName = modelPath + modelName + ".param";
	std::string binName = modelPath + modelName + ".bin";
	ncnn::Net net;
	net.load_param(paramName.data());
	net.load_model(binName.data());


	cv::Mat img;

	poseMatching matching;
	matching.initialize("kaihetiao");

	Timer time;

	//跳过几帧读视频，模拟实际帧率较低的情况
	for (int i = 0; i < 1000; i=i+4)
	{
		video.set(CV_CAP_PROP_POS_FRAMES, i);
		if (video.read(img))
		{
			ncnn::Mat in = ncnn::Mat::from_pixels_resize(
				img.data, ncnn::Mat::PIXEL_RGB, w, h, 192, 256);

			float mean[3] = { 128.f,128.f,128.f };
			float norm[3] = { 1 / 128.f ,1 / 128.f ,1 / 128.f };
			in.substract_mean_normalize(mean, norm);

			ncnn::Extractor ex = net.create_extractor();
			ex.set_light_mode(true);
			ex.set_num_threads(8);

			ex.input("input", in);

			ncnn::Mat out;
			ex.extract("output", out);


			//std::vector<std::pair<cv::Point, float>> kp;
			//kp = getKPloc(out);
			std::vector<cv::Point> kp;
			kp = postProc(out);

			time.start();
			//matching
			matching.loadKP(kp);
			if (matching.calibration())
			{
				matching.affineWithStandard();
				matching.getSimilarity();
				std::cout << matching.countAction() << std::endl;
				matching.keepTime();
				matching.getSuggestion();
			}
			
			time.stop();
			std::cout << "matching time: " << time.getElapsedTimeInMilliSec() << std::endl;


			cv::resize(img, img, cv::Size(192, 256));
			for (int i = 0; i < matching.userKPAffine.size(); i++)
			{
				cv::circle(img, matching.userKPAffine[i], 2, cv::Scalar(255, 0, 0), -1);
			}
			matching.clear();
			//writer << img;

			cv::imshow("test", img);
			cv::waitKey(0);
		}
		else
		{
			std::cout << "video ran out" << std::endl;
			break;
		}

	}
	net.clear();

	return 0;

}


std::vector<std::pair<cv::Point, float>> getKPloc(ncnn::Mat& m)
{
	//找到每个channel的最大值对应的x和y坐标
	std::vector<std::pair<cv::Point, float>> loc;

	for (int q = 0; q < 14; q++)   //只用前14个点，最后一个点是多余的
	{
		const float* ptr = m.channel(q);
		float maxNum = -100.0;
		cv::Point maxLoc(0, 0);
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x++)
			{
				if (ptr[x] > maxNum)
				{
					maxNum = ptr[x];
					maxLoc.x = x;
					maxLoc.y = y;
				}
			}
			ptr += m.w;
		}
		//std::cout << maxNum << std::endl;
		//std::cout << maxLoc << std::endl;
		loc.push_back({ maxLoc, maxNum });
	}
	return loc;
}

std::vector<cv::Point> postProc(ncnn::Mat& m)
{
	//找到每个channel的最大值对应的x和y坐标
	std::vector<cv::Point> loc;

	for (int q = 0; q < m.c; q++)
	{
		const float* ptr = m.channel(q);
		float maxNum = -100.0;
		cv::Point maxLoc(0, 0);
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x = x + 2)
			{
				float t1 = ptr[x];
				float t2 = ptr[x + 1];
				t1 = sigmoid(t1);
				t2 = sigmoid(t2);
				cv::Point pt;
				pt.x = t1 * 192;
				pt.y = t2 * 256;
				loc.push_back(pt);
				//std::cout << pt << std::endl;
			}
			ptr += m.w;
			//printf("\n");
		}
		//std::cout << maxNum << std::endl;
		loc.push_back(maxLoc);
		//printf("------------------------\n");
	}
	/*std::cout << loc << std::endl;*/
	return loc;
}

float sigmoid(float x)
{
	return (1 / (1 + exp(-x)));
}