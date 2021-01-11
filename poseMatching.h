#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>

#include"Timer.h"

#define DEBUG 0

class poseMatching
{
public:

	//初始化，输入动作名称小写全拼
	bool initialize(std::string movement);
	//读取已处理好的，存储在txt文件中的标准帧KP
	void loadStandardKPFromFile();
	//读入当前帧的kp
	void loadKP(std::vector<std::pair<cv::Point, float>> kp);
	bool calibration();
	void affineWithStandard();
	void getSimilarity();
	int countAction();
	float keepTime();

	float getSuggestion();

	//count
	int count;  //计数

	//打分和获取建议
	float score;
	float allScore;

	void clear();

private:
	//动作名称，小写全拼
	std::string movementName;
	//当前帧的kp和confidence
	std::vector<std::pair<cv::Point, float>> userData;
	std::vector<cv::Point> userKP;
	std::vector<cv::Point> userKPAffine;
	std::vector<float> userConfidence;

	//标准帧的kp，从txt文件中读取，取自教练视频
	std::vector<cv::Point> standardKP;   
	//用于做affine的躯干6个点（1头，2脖子，3左肩，4右肩，9左胯，10右胯）
	int affineIdx[14];
	//对不同的任务，和标准帧比较时所用的点不同
	int matchingKPIdx[14];
	//calibration所用的点
	int calibrationIdx[14];


	//calibration
	int calibrationWindow;  //calibration所取的时间窗，即confidenceVec的长度，维持当前帧前20帧（6fps）,帧率提高或降低，需更改这个值
	//int smoothCaliWindow;  //滑动平均的窗口大小，即smoothConfidenceVec的长度，对于6fps的检测速度，可取3，帧率变化时需更改
	std::vector<float> confidenceVec;
	//std::vector<float> smoothConfidenceVec;
	int calibrationNum;
	bool calibrationDone;  //判断用户是否完成calibration


	//count
	bool state;  //判断用户是否处于标准动作中
	int smoothSimilarityWindow; //相似性滑动平均窗口，6fps时可取3
	std::vector<float> smoothCosVec;  //余弦相似性
	std::vector<float> smoothDisVec;  //欧式距离相似性
	float threCosine;   //余弦相似性阈值
	float threDist;    //欧氏距离相似性阈值
	float countConfThre;    //计数时，若此帧的动作关键点置信度低于阈值，此帧不采用


	//计时
	Timer timerForOneAction;
	float oneActionTime;
	float standardTime;
	float allActionTime;
	bool timekeeping;


	//打分和获取建议
	std::vector<float> scoreAccumulate;
	std::vector<std::vector<cv::Point>> poseAccumulate;
	std::string suggestion;
	float perfectThre;
};


float get2ptdistance(cv::Point2f p1, cv::Point2f p2);
float getMold(const std::vector<float>& vec);
float getCosineSimilarity(const std::vector<float>& lhs, const std::vector<float>& rhs);
float getMean(std::vector<float>& vec);