#include "net.h"
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<fstream>

#include"Timer.h"
#include"poseMatching.h"


std::vector<std::pair<cv::Point, float>> getKPloc(ncnn::Mat& m);


int main()
{
	std::string testVideo =
		"F:\\Kexin\\posture\\pose_matching\\matching_VS2017\\matching_VS2017\\testVideo\\cepingju_new1_crop.avi";
	cv::VideoCapture video(testVideo);
	if (!video.isOpened())
	{
		std::cout << "error! video can't open!" << std::endl;
	}
	int w = video.get(3);
	int h = video.get(4);

	std::string modelPath = "F:\\Kexin\\posture\\pose_matching\\matching_VS2017\\matching_VS2017\\model\\";
	std::string modelName = "model_torch_heatmap_sim";
	std::string paramName = modelPath + modelName + ".param";
	std::string binName = modelPath + modelName + ".bin";
	ncnn::Net net;
	net.load_param(paramName.data());
	net.load_model(binName.data());


	cv::Mat img;

	poseMatching matching;
	matching.initialize("cepingju");

	Timer time;

	//跳过几帧读视频，模拟实际帧率较低的情况
	for (int i = 0; i < 1000; i=i+4)
	{
		video.set(CV_CAP_PROP_POS_FRAMES, i);
		if (video.read(img))
		{
			ncnn::Mat in = ncnn::Mat::from_pixels_resize(
				img.data, ncnn::Mat::PIXEL_RGB, w, h, 168, 224);

			float mean[3] = { 128.f,128.f,128.f };
			float norm[3] = { 1 / 128.f ,1 / 128.f ,1 / 128.f };
			in.substract_mean_normalize(mean, norm);

			ncnn::Extractor ex = net.create_extractor();
			ex.set_light_mode(true);
			ex.set_num_threads(8);

			ex.input("input", in);

			ncnn::Mat out;
			ex.extract("output", out);

			std::vector<std::pair<cv::Point, float>> kp;

			kp = getKPloc(out);

			//time.start();
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
			matching.clear();
			//time.stop();
			//std::cout << "matching time: " << time.getElapsedTimeInMilliSec() << std::endl;


			cv::resize(img, img, cv::Size(168, 224));
			for (int i = 0; i < kp.size(); i++)
			{
				cv::circle(img, kp[i].first * 4, 2, cv::Scalar(255, 0, 0), -1);
			}
			
			//writer << img;

			cv::imshow("test", img);
			cv::waitKey(1);
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