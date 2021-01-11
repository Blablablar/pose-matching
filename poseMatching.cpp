#include"poseMatching.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::string;
using cv::Point;
using cv::Point2f;

bool comp(float c)
{
	return 0.5 < c;
}

bool poseMatching::initialize(std::string movement)
{
	movementName = movement;
	if (movementName == "kaihetiao")
	{
		//用得到的点为1，不用的点为0
		int tempIdx[14] = { 0,0,0,0,1,1,1,1,0,0,0,0,0,0 };
		memcpy(matchingKPIdx, tempIdx, sizeof(tempIdx));
		//如果计数不准确，这些参数需要调整
		smoothSimilarityWindow = 2;
		countConfThre = 0.3;
		threCosine = 0.95;
		threDist = 0.7;
		timekeeping = false;
		standardTime = 0.0;
		suggestion = "收紧核心，动作幅度加大";
		perfectThre = 95;
	}
	else if (movementName == "cepingju")
	{
		int tempIdx[14] = { 0,0,0,0,1,1,1,1,0,0,0,0,0,0 };
		memcpy(matchingKPIdx, tempIdx, sizeof(tempIdx));
		//如果计数不准确，这些参数需要调整
		smoothSimilarityWindow = 4;
		countConfThre = 0.4;
		threCosine = 0.95;
		threDist = 0.72;
		timekeeping = true;
		standardTime = 2;
		suggestion = "手臂与地面平行，不要耸肩";
		perfectThre = 97.5;
	}
	else if (movementName == "jugangling")
	{
		int tempIdx[14] = { 1,1,1,1,1,1,1,1,0,0,0,0,0,0 };
		memcpy(matchingKPIdx, tempIdx, sizeof(tempIdx));
		//如果计数不准确，这些参数需要调整
		smoothSimilarityWindow = 2;
		countConfThre = 0.3;
		threCosine = 0.95;
		threDist = 0.7;
		timekeeping = false;
		standardTime = 0.0;
		suggestion = "下蹲时腰背挺直，发力时收紧核心，脚尖着地";
		perfectThre = 95;
	}
	else
	{
		cout << "error!! input can't fit any movement" << endl;
		return false;
	}
	loadStandardKPFromFile();

	//用于做affine的躯干3个点（1头，9左胯，10右胯）
	int tempIdx[14] = { 1,0,0,0,0,0,0,0,1,1,0,0,0,0 };
	memcpy(affineIdx, tempIdx, sizeof(tempIdx));

	//用于做calibration的8个点
	int tempIdx2[14] = { 1,1,1,1,1,1,1,1,0,0,0,0,0,0 };
	memcpy(calibrationIdx, tempIdx2, sizeof(tempIdx2));

	//参数初始化
	//calibration 相关
	calibrationWindow = 20;
	calibrationNum = 15;
	calibrationDone = false;

	//count 相关
	count = 0;
	state = false;

	//计时
	oneActionTime = 0.0;
	allActionTime = 0.0;

	//打分
	score = 0.0;
	allScore = 0.0;
}

//根据实际路径更改standardFile路径
void poseMatching::loadStandardKPFromFile()
{
	string fileName = 
		"F:\\Kexin\\posture\\pose_matching\\matching_VS2017\\matching_VS2017\\teacher\\" 
		+ movementName + "_standard.txt";
	std::ifstream fin(fileName);
	if (!fin)
	{
		cout << "error! can't open file" << endl;
		return;
	}

	vector<double> in;
	string s;

	while (getline(fin, s))
	{
		//string to char
		char *s_input = (char*)s.c_str();
		const char * split = " ";
		char *buf;
		char *p = strtok_s(s_input, split, &buf);
		double a;
		while (p != NULL)
		{
			a = atof(p);
			in.push_back(a);
			p = strtok_s(NULL, split, &buf);
		}

	}
	fin.close();

	Point pt;
	for (int i = 4; i < in.size(); i = i + 3)
	{
		pt.x = in[i];
		pt.y = in[i + 1];
		standardKP.push_back(pt);
	}

	cout << "load " << standardKP.size() << " points from standard kp file!" << endl;
}

void poseMatching::loadKP(std::vector<std::pair<cv::Point, float>> kp)
{
	userData = kp;
	for (int i = 0; i < userData.size(); i++)
	{
		userKP.push_back(userData[i].first);
		userConfidence.push_back(userData[i].second);
	}
}

bool poseMatching::calibration()
{
	if (!calibrationDone)
	{
		vector<float> cali;
		for (int i = 0; i < 14; i++)
		{
			if (calibrationIdx[i] == 1)
			{
				cali.push_back(userConfidence[i]);
			}
		}
		float meanConfidence = getMean(cali);
		if (DEBUG)
		{
			cout << meanConfidence << endl;
		}

		confidenceVec.push_back(meanConfidence);
		if (confidenceVec.size() > calibrationWindow)
		{
			//删除首元素
			confidenceVec.erase(confidenceVec.begin());
		}

		//6fps下，confidenceVec中超过15帧平均confidence>Thre,视为calibration成功
		if (count_if(confidenceVec.begin(), confidenceVec.end(), comp)>= calibrationNum)
		{
			calibrationDone = true;
			cout << "Calibration Done" << endl;
		}
		else
		{
			cout << "Please stand up in front of mirror" << endl;
		}
	}

	return calibrationDone;
}

void poseMatching::affineWithStandard()
{
	//用于仿射变换的躯体3点
	//TO TEST: opencv的仿射变换函数只能用3个点
	//选择头、左髋、右髋
	Point2f bodycenter[3];
	Point2f standardcenter[3];
	//for (int i = 0; i < 14; i++)
	//{
	//	if (affineIdx[i] == 1)
	//	{
	//		bodycenter[i] = userKP[i];
	//		standardcenter[i] = standardKP[i];
	//	}
	//}
	bodycenter[0] = userKP[0];
	bodycenter[1] = userKP[8];
	bodycenter[2] = userKP[9];
	standardcenter[0] = standardKP[0];
	standardcenter[1] = standardKP[8];
	standardcenter[2] = standardKP[9];

	//opencv函数做仿射变换
	//求仿射矩阵，2*3
	//[a, b, c]
	//[d, e, f]
	cv::Mat affineMatrix = cv::getAffineTransform(bodycenter, standardcenter);
	double affineMatrixData[2][3];
	for (int i = 0; i < affineMatrix.rows; i++)
	{
		for (int j = 0; j < affineMatrix.cols; j++)
		{
			affineMatrixData[i][j] = affineMatrix.at<double>(i, j);
		}
	}

	for (int i = 0; i < userKP.size(); i++)
	{
		float x = affineMatrixData[0][0] * userKP[i].x + affineMatrixData[0][1] * userKP[i].y + affineMatrixData[0][2];
		float y = affineMatrixData[1][0] * userKP[i].x + affineMatrixData[1][1] * userKP[i].y + affineMatrixData[1][2];
		userKPAffine.push_back(Point(x, y));
	}
}

void poseMatching::getSimilarity()
{
	vector<Point> userActionKP;
	vector<Point> standardKP_toUse;
	
	float meanConfi = 0.0;
	for (int i = 0; i < 14; i++)
	{
		if (matchingKPIdx[i] == 1)
		{
			userActionKP.push_back(userKPAffine[i]);
			standardKP_toUse.push_back(standardKP[i]);
			meanConfi = meanConfi + userConfidence[i];
		}
	}
	meanConfi = meanConfi / userActionKP.size();
	if (DEBUG)
	{
		cout << "meanConfi: "<<meanConfi << endl;
	}

	if (meanConfi > countConfThre)  //计数用的关键点的confidence  大于一定值
	{
		//欧氏距离求相似性
		float meanD = 0.0;
		for (int i = 0; i < userActionKP.size(); i++)
		{
			float d = get2ptdistance(userActionKP[i], standardKP_toUse[i]);
			d = 100.0 / (100.0 + d);
			meanD = meanD + d;
		}
		meanD = meanD / userActionKP.size();

		if (smoothDisVec.size() >= smoothSimilarityWindow)
		{
			smoothDisVec.erase(smoothDisVec.begin());
		}
		smoothDisVec.push_back(meanD);


		//余弦相似性
		//拉成一条直线
		vector<float> userActionKP_line;
		vector<float> standardKP_line;
		for (int i = 0; i < userActionKP.size(); i++)
		{
			userActionKP_line.push_back(userActionKP[i].x);
			userActionKP_line.push_back(userActionKP[i].y);
			standardKP_line.push_back(standardKP_toUse[i].x);
			standardKP_line.push_back(standardKP_toUse[i].y);
		}
		float cosineSimi = getCosineSimilarity(userActionKP_line, standardKP_line);

		if (smoothCosVec.size() >= smoothSimilarityWindow)
		{
			smoothCosVec.erase(smoothCosVec.begin());
		}
		smoothCosVec.push_back(cosineSimi);
	}
}

int poseMatching::countAction()
{
	//求距离相似性和余弦相似性在滑动平均窗口中的均值
	float distSimi = getMean(smoothDisVec);
	float cosSimi = getMean(smoothCosVec);
	if (DEBUG)
	{
		cout << "distSimilarity: " << distSimi << endl;
		cout << "cosineSimilarity: " << cosSimi << endl;
	}

	if (state == false)
	{
		if (distSimi >= threDist && cosSimi >= threCosine)  //动作开始
		{

			if (DEBUG)
			{
				cout << "-----------one action start-------------" << endl;
			}
			state = true;
			count = count + 1;
			timerForOneAction.start();
		}
	}
	if (state == true)
	{
		oneActionTime = timerForOneAction.getElapsedTimeInSec();
		scoreAccumulate.push_back(distSimi + cosSimi);
		poseAccumulate.push_back(userKP);

		if (distSimi < threDist || cosSimi < threCosine)  //动作结束
		{
			state = false;
			timerForOneAction.stop();

			//动作结束时打分
			float meanScore = getMean(scoreAccumulate);
			score = (meanScore / 1.8) * 100;
			if (score > 100)
			{
				score = 100;
			}
			allScore = allScore + score;
			scoreAccumulate.clear();
			poseAccumulate.clear();

			if (DEBUG)
			{
				cout << "meanScore: " << meanScore << endl;
				cout << "score: "<<score << endl;
				cout << "allScore: " << allScore << endl;
				cout << "-----------one action done-------------" << endl;
			}
		}
	}

	return count;
}

float poseMatching::keepTime()
{
	if (timekeeping)
	{
		if (oneActionTime < standardTime)
		{
			cout << "please hold on long time" << endl;
		}
		else
		{
			cout << "finish!!" << endl;
		}

		return oneActionTime;
	}
	return 0.0f;
}

float poseMatching::getSuggestion()
{
	if (score != 0.0)
	{
		if (score > perfectThre)
		{
			cout << "perfect!!" << endl;
			if (count < 3)
			{
				cout << suggestion << endl;
			}
		}
		else
		{
			cout << "good!!" << endl;
			cout << suggestion << endl;
		}
	}
	return score;
}

void poseMatching::clear()
{
	userData.clear();
	userKP.clear();
	userKPAffine.clear();
	userConfidence.clear();
	score = 0.0;
}

float get2ptdistance(Point2f p1, Point2f p2)
{
	return  pow(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2), 0.5);
}

//求向量的模长
float getMold(const vector<float>& vec) 
{   
	int n = vec.size();
	float sum = 0.0;
	for (int i = 0; i < n; ++i)
		sum += vec[i] * vec[i];
	return sqrt(sum);
}

//求两个向量的余弦相似性
float getCosineSimilarity(const vector<float>& lhs, const vector<float>& rhs) 
{
	int n = lhs.size();
	assert(n == rhs.size());
	float tmp = 0.0;  //内积
	for (int i = 0; i < n; ++i)
		tmp += lhs[i] * rhs[i];
	float result = tmp / (getMold(lhs)*getMold(rhs));
	return result * 0.5 + 0.5;
}

//求一个vector中的数据均值
float getMean(vector<float>& vec)
{
	float mean = 0.0;
	for (int i = 0; i < vec.size(); i++)
	{
		mean = mean + vec[i];
	}
	mean = mean / float(vec.size());

	return mean;
}