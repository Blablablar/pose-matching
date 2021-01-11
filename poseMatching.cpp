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
	return 1.8 < c;
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
		threCosine = 0.95;
		threDist = 0.72;
		timekeeping = true;
		standardTime = 2;
		suggestion = "手臂与地面平行，不要耸肩";
		perfectThre = 97.5;
	}
	else if (movementName == "jugangling")
	{
		int tempIdx[14] = { 0,0,0,1,1,1,1,1,1,1,1,1,0,0 };
		memcpy(matchingKPIdx, tempIdx, sizeof(tempIdx));
		//如果计数不准确，这些参数需要调整
		smoothSimilarityWindow = 2;
		threCosine = 0.95;
		threDist = 0.72;
		timekeeping = false;
		standardTime = 0.0;
		suggestion = "下蹲时腰背挺直，发力时收紧核心，脚尖着地";
		perfectThre = 97;
	}
	else
	{
		cout << "error!! input can't fit any movement" << endl;
		return false;
	}
	standardKP = loadKPFromFile(movementName);

	//用于做affine的躯干3个点（1头，9左胯，10右胯）
	int tempIdx[14] = { 1,0,0,0,0,0,0,0,1,1,0,0,0,0 };
	memcpy(affineIdx, tempIdx, sizeof(tempIdx));


	//从文件中读取用于calibration的kp
	caliKP = loadKPFromFile("standing");
	//用于做calibration的14个点
	int tempIdx2[14] = { 1,1,1,1,0,0,0,0,1,1,1,1,1,1 };
	memcpy(calibrationIdx, tempIdx2, sizeof(tempIdx2));

	//参数初始化
	//calibration 相关
	calibrationWindow = 20;
	calibrationNum = 16;
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
std::vector<cv::Point> poseMatching::loadKPFromFile(std::string movementName)
{
	string fileName = "./teacher\\" + movementName + "_standard.txt";
	std::ifstream fin(fileName);
	if (!fin)
	{
		cout << "error! can't open file" << endl;
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
	std::vector<cv::Point> kp;
	for (int i = 4; i < in.size(); i = i + 3)
	{
		pt.x = in[i];
		pt.y = in[i + 1];
		kp.push_back(pt);
	}

	cout << "load " << kp.size() << " points from standard kp file!" << endl;

	return kp;
}

void poseMatching::loadKP(std::vector<cv::Point> kp)
{
	userKP = kp;
}

bool poseMatching::calibration()
{
	if (!calibrationDone)
	{
		userKPAffine = affine(userKP, caliKP);

		vector<Point> userKP_touse;
		vector<Point> caliKP_touse;

		for (int i = 0; i < 14; i++)
		{
			if (calibrationIdx[i] == 1)
			{
				userKP_touse.push_back(userKPAffine[i]);
				caliKP_touse.push_back(caliKP[i]);
			}
		}

		float distSimi = getDistSimilarity(userKP_touse, caliKP_touse);
		float cosSimi = getCosineSimilarity(userKP_touse, caliKP_touse);

		caliSimivec.push_back(distSimi + cosSimi);
		if (caliSimivec.size() > calibrationWindow)
		{
			//删除首元素
			caliSimivec.erase(caliSimivec.begin());
		}

		//6fps下，confidenceVec中超过15帧平均confidence>Thre,视为calibration成功
		if (count_if(caliSimivec.begin(), caliSimivec.end(), comp)>= calibrationNum)
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
	userKPAffine = affine(userKP, standardKP);
}

void poseMatching::getSimilarity()
{
	vector<Point> userActionKP;
	vector<Point> standardKP_toUse;
	
	for (int i = 0; i < 14; i++)
	{
		if (matchingKPIdx[i] == 1)
		{
			userActionKP.push_back(userKPAffine[i]);
			standardKP_toUse.push_back(standardKP[i]);
		}
	}

	//欧氏距离求相似性
	float distSimi = getDistSimilarity(userActionKP, standardKP_toUse);

	if (smoothDisVec.size() >= smoothSimilarityWindow)
	{
		smoothDisVec.erase(smoothDisVec.begin());
	}
	smoothDisVec.push_back(distSimi);


	//余弦相似性
	float cosineSimi = getCosineSimilarity(userActionKP, standardKP_toUse);
	if (smoothCosVec.size() >= smoothSimilarityWindow)
	{
		smoothCosVec.erase(smoothCosVec.begin());
	}
	smoothCosVec.push_back(cosineSimi);

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

	userKP.clear();
	userKPAffine.clear();
	score = 0.0;
}

std::vector<cv::Point> affine(std::vector<cv::Point> user, std::vector<cv::Point> standard)
{
	if (user.size() < 14 || standard.size() < 14)
	{
		cout << "There should be 14 KP points" << endl;
	}


	std::vector<cv::Point> userKPAffine;
	//用于仿射变换的躯体3点
	//TO TEST: opencv的仿射变换函数只能用3个点
	//选择头、左髋、右髋
	Point2f bodycenter[3];
	Point2f standardcenter[3];

	bodycenter[0] = user[0];
	bodycenter[1] = user[8];
	bodycenter[2] = user[9];
	standardcenter[0] = standard[0];
	standardcenter[1] = standard[8];
	standardcenter[2] = standard[9];

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

	for (int i = 0; i < user.size(); i++)
	{
		float x = affineMatrixData[0][0] * user[i].x + affineMatrixData[0][1] * user[i].y + affineMatrixData[0][2];
		float y = affineMatrixData[1][0] * user[i].x + affineMatrixData[1][1] * user[i].y + affineMatrixData[1][2];
		userKPAffine.push_back(Point(x, y));
	}
	return userKPAffine;

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
float getCosineSimilarity(const vector<Point>& lhsP, const vector<Point>& rhsP)
{
	vector<float> lhs;
	vector<float> rhs;
	for (int i = 0; i < lhsP.size(); i++)
	{
		lhs.push_back(lhsP[i].x);
		lhs.push_back(lhsP[i].y);
		rhs.push_back(rhsP[i].x);
		rhs.push_back(rhsP[i].y);
	}

	int n = lhs.size();
	assert(n == rhs.size());
	float tmp = 0.0;  //内积
	for (int i = 0; i < n; ++i)
		tmp += lhs[i] * rhs[i];
	float result = tmp / (getMold(lhs)*getMold(rhs));
	return result * 0.5 + 0.5;
}

float getDistSimilarity(const vector<Point>& lhsP, const vector<Point>& rhsP)
{
	float meanD = 0.0;
	for (int i = 0; i < lhsP.size(); i++)
	{
		float d = get2ptdistance(lhsP[i], rhsP[i]);
		d = 100.0 / (100.0 + d);
		meanD = meanD + d;
	}
	meanD = meanD / rhsP.size();

	return meanD;
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