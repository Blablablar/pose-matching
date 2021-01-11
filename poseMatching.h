#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>

#include"Timer.h"

#define DEBUG 1

class poseMatching
{
public:

	//��ʼ�������붯������Сдȫƴ
	bool initialize(std::string movement);
	//��ȡ�Ѵ���õģ��洢��txt�ļ��еı�׼֡KP
	std::vector<cv::Point> loadKPFromFile(std::string movementName);
	//���뵱ǰ֡��kp
	void loadKP(std::vector<cv::Point> kp);
	bool calibration();
	void affineWithStandard();
	void getSimilarity();
	int countAction();
	float keepTime();

	float getSuggestion();

	//count
	int count;  //����

	//��ֺͻ�ȡ����
	float score;
	float allScore;

	void clear();

private:
	//�������ƣ�Сдȫƴ
	std::string movementName;
	//��ǰ֡��kp
	std::vector<cv::Point> userKP;
	std::vector<cv::Point> userKPAffine;


	//��׼֡��kp����txt�ļ��ж�ȡ��ȡ�Խ�����Ƶ
	std::vector<cv::Point> standardKP;   
	//������affine������6���㣨1ͷ��2���ӣ�3��磬4�Ҽ磬9��裬10�ҿ裩
	int affineIdx[14];
	//�Բ�ͬ�����񣬺ͱ�׼֡�Ƚ�ʱ���õĵ㲻ͬ
	int matchingKPIdx[14];
	//calibration���õĵ�
	int calibrationIdx[14];


	//calibration
	//������calibration��kp
	std::vector<cv::Point> caliKP;
	int calibrationWindow;  //calibration��ȡ��ʱ�䴰����confidenceVec�ĳ��ȣ�ά�ֵ�ǰ֡ǰ20֡��6fps��,֡����߻򽵵ͣ���������ֵ
	//int smoothCaliWindow;  //����ƽ���Ĵ��ڴ�С����smoothConfidenceVec�ĳ��ȣ�����6fps�ļ���ٶȣ���ȡ3��֡�ʱ仯ʱ�����
	std::vector<float> caliSimivec;
	//std::vector<float> smoothConfidenceVec;
	int calibrationNum;
	bool calibrationDone;  //�ж��û��Ƿ����calibration


	//count
	bool state;  //�ж��û��Ƿ��ڱ�׼������
	int smoothSimilarityWindow; //�����Ի���ƽ�����ڣ�6fpsʱ��ȡ3
	std::vector<float> smoothCosVec;  //����������
	std::vector<float> smoothDisVec;  //ŷʽ����������
	float threCosine;   //������������ֵ
	float threDist;    //ŷ�Ͼ�����������ֵ
	float countConfThre;    //����ʱ������֡�Ķ����ؼ������Ŷȵ�����ֵ����֡������


	//��ʱ
	Timer timerForOneAction;
	float oneActionTime;
	float standardTime;
	float allActionTime;
	bool timekeeping;


	//��ֺͻ�ȡ����
	std::vector<float> scoreAccumulate;
	std::vector<std::vector<cv::Point>> poseAccumulate;
	std::string suggestion;
	float perfectThre;


};


float get2ptdistance(cv::Point2f p1, cv::Point2f p2);
float getMold(const std::vector<float>& vec);
float getCosineSimilarity(const std::vector<cv::Point>& lhsP, const std::vector<cv::Point>& rhsP);
float getDistSimilarity(const std::vector<cv::Point>& lhsP, const std::vector<cv::Point>& rhsP);
float getMean(std::vector<float>& vec);

std::vector<cv::Point> affine(std::vector<cv::Point> user, std::vector<cv::Point> standard);