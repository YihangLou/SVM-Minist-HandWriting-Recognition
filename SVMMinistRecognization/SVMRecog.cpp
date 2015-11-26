// DigitsRec_HOG_SVM.cpp : 定义控制台应用程序的入口点。
#include "opencv2/opencv.hpp"
#include "fstream"
#include "svm.h"
using namespace std;
using namespace cv;

#define srcfeature

vector<string> trainImageList;//训练图像列表，此处路径
vector<int> trainLabelList;   //标签
vector<string> testImageList;//训练图像列表，此处路径
string trainImageFile= "D:\\WorkSpace\\homework\\PatternRecognization\\第一次作业\\minist\\train_image\\imagelist.txt";
string testImageFile = "D:\\WorkSpace\\homework\\PatternRecognization\\第一次作业\\minist\\test_image\\imagelist.txt";
string testBasePath = "D:\\WorkSpace\\homework\\PatternRecognization\\第一次作业\\minist\\test_image\\";
string trainBasePath = "D:\\WorkSpace\\homework\\PatternRecognization\\第一次作业\\minist\\train_image\\";
string SVMModel ="svm_model.xml";
CvMat * dataMat;
CvMat * labelMat;

//***************************************************************
// 名称:    readTrainFileList
// 功能:    读取训练的图像列表和图像的位置
// 权限:    public 
// 返回值:  void
// 参数:    string trainImageFile 文件列表
// 参数:    string basePath 基地址
// 参数:    vector<string> & trainImageList 图像路径list
// 参数:    vector<int> & trainLabelList 图像标签list
//***************************************************************
void readTrainFileList(string trainImageFile, string basePath, vector<string> &trainImageList,  vector<int> &trainLabelList)
{
	ifstream readData( trainImageFile );
	string buffer;
	while( readData )
	{    
		if( getline( readData, buffer))    
		{    
			int label = int((buffer[0])-'0');//在我这里路径中第一个文件夹就是类别
			trainLabelList.push_back( label);  
			trainImageList.push_back( buffer );//图像路径       
		}    
	}    
	readData.close();
	cout<<"Read Train Data Complete"<<endl;
}

//***************************************************************
// 名称:    readTestFileList
// 功能:    读测试文件
// 权限:    public 
// 返回值:  void
// 参数:    string testImageFile
// 参数:    string basePath
// 参数:    vector<string> & testImageList 测试图像列表
//***************************************************************
void readTestFileList(string testImageFile, string basePath, vector<string> &testImageList)
{
	ifstream readData( testImageFile );  //加载测试图片集合
	string buffer;
	while( readData )
	{    
		if( getline( readData, buffer))    
		{    
			testImageList.push_back( buffer );//图像路径       
		}    
	}    
	readData.close();
	cout<<"Read Test Data Complete"<<endl;
}

//***************************************************************
// 名称:    processHogFeature
// 功能:    计算Hog特征
// 权限:    public 
// 返回值:  void
// 参数:    vector<string> trainImageList
// 参数:    vector<int> trainLabelList
// 参数:    CvMat *  & dataMat
// 参数:    CvMat *  & labelMat
//***************************************************************
void processHogFeature(vector<string> trainImageList,vector<int> trainLabelList, CvMat * &dataMat,CvMat * &labelMat)
{
	
	 int trainSampleNum = trainImageList.size();
	 dataMat = cvCreateMat( trainSampleNum, 324, CV_32FC1 );  //324为Hog feature Size
	 cvSetZero( dataMat );     
	 labelMat = cvCreateMat( trainSampleNum, 1, CV_32FC1 );    
	 cvSetZero( labelMat );    
	 IplImage* src;   
	 IplImage* trainImg=cvCreateImage(cvSize(20,20),8,3);//20 20

	for( int i = 0; i != trainImageList.size(); i++ ) 
	{    
		src=cvLoadImage( (trainBasePath  + trainImageList[i]).c_str(),1);    
		if( src == NULL )    
		{    
			cout<<" can not load the image: "<<(trainBasePath  + trainImageList[i]).c_str()<<endl;    
			continue;    
		}    
		//cout<<"Calculate Hog Feature "<<(trainBasePath  + trainImageList[i]).c_str()<<endl;    

		cvResize(src,trainImg);     
		HOGDescriptor *hog=new HOGDescriptor(cvSize(20,20),cvSize(10,10),cvSize(5,5),cvSize(5,5),9);      
		vector<float>descriptors;    
		hog->compute(trainImg, descriptors,Size(1,1), Size(0,0));     

		int j =0; 
		for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
		{    
			cvmSet(dataMat,i,j,*iter);//存储HOG特征 
			j++;    
		}       
		cvmSet( labelMat, i, 0, trainLabelList[i] );    
		//cout<<"Image and label "<<trainImageList[i].c_str()<<" "<<trainLabelList[i]<<endl;    
	}    
	cout<<"Calculate Hog Feature Complete"<<endl;
	cout<<dataMat<<endl;
}

void processNonFeature(vector<string> trainImageList,vector<int> trainLabelList, CvMat * &dataMat,CvMat * &labelMat)
{

	int trainSampleNum = trainImageList.size();
	dataMat = cvCreateMat( trainSampleNum, 400, CV_32FC1 );  //324为Hog feature 大小，需提前设置 
	cvSetZero( dataMat );     
	labelMat = cvCreateMat( trainSampleNum, 1, CV_32FC1 );    
	cvSetZero( labelMat );    
	IplImage* src;   
	IplImage* resizeImg=cvCreateImage(cvSize(20,20),8,3);//20 20是训练样本的大小

	for( int i = 0; i != trainImageList.size(); i++ ) 
	{    
		src=cvLoadImage( (trainBasePath  + trainImageList[i]).c_str(),1);    
		if( src == NULL )    
		{    
			cout<<" can not load the image: "<<(trainBasePath  + trainImageList[i]).c_str()<<endl;    
			continue;    
		}    
		//cout<<"Calculate Hog Feature "<<(trainBasePath  + trainImageList[i]).c_str()<<endl;    

		cvResize(src,resizeImg);     
		IplImage * grayImage = cvCreateImage(cvGetSize(resizeImg), IPL_DEPTH_8U, 1);
		cvCvtColor(resizeImg,grayImage,CV_BGR2GRAY);
		//二值化图像
		IplImage * binaryImage = cvCreateImage(cvGetSize(grayImage),IPL_DEPTH_8U,1);
		cvThreshold(grayImage,binaryImage,25,255,CV_THRESH_BINARY);
		//cvNamedWindow("src");
		//cvShowImage("src", src);

		//cvNamedWindow("show");
		//cvShowImage("show", binaryImage);
		//cvWaitKey(0);//这里是看一下二值化的效果怎么样
		HOGDescriptor *hog=new HOGDescriptor(cvSize(20,20),cvSize(10,10),cvSize(5,5),cvSize(5,5),9);      
		vector<float>descriptors;    


		int j =0; //j为矩阵的水平坐标，要把特征从vector中拷贝过来
		uchar * tmp = new uchar;
		for(int n=0;n<binaryImage->height;n++)  
		{	for(int m=0;m<binaryImage->width;m++)  
		{	
			*tmp=((uchar *)(binaryImage->imageData + n*binaryImage->widthStep))[m];  
			cvmSet(dataMat,i,j,*tmp);//存储HOG特征 
			j++;
		}
		}  

		cvmSet( labelMat, i, 0, trainLabelList[i] );    
		//cout<<"Image and label "<<trainImageList[i].c_str()<<" "<<trainLabelList[i]<<endl;    
	}    
	cout<<"Calculate Hog Feature Complete"<<endl;

}

//***************************************************************
// 名称:    trainSVM
// 功能:    此处用的是opencv的SVM训练
// 权限:    public 
// 返回值:  void
// 参数:    CvMat *  & dataMat
// 参数:    CvMat *  & labelMat
//***************************************************************
void trainSVM(CvMat * & dataMat,CvMat * & labelMat )
{
	cout<<"train svm start"<<endl;
	cout<<dataMat<<endl;
	CvSVM svm;
	CvSVMParams param;//这里是SVM训练相关参数  
	CvTermCriteria criteria;      
	criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );      
	param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );          

	svm.train( dataMat, labelMat, NULL, NULL, param );//训练数据          
	svm.save( SVMModel.c_str());  
	cout<<"SVM Training Complete"<<endl;
}

//***************************************************************
// 名称:    trainLibSVM
// 功能:    此处用的是LibSVM库的SVM训练
// 权限:    public 
// 返回值:  void
// 参数:    CvMat * & dataMat
// 参数:    CvMat *  & labelMat
//***************************************************************
void trainLibSVM(CvMat *& dataMat, CvMat * & labelMat)
{
	cout<<"LibSVM start"<<endl;
	//配置SVM参数
	svm_parameter param;
	//param.svm_type = C_SVC;
	param.svm_type = EPSILON_SVR;
	param.kernel_type = RBF;
	param.degree = 10.0;
	param.gamma = 0.09;
	param.coef0 = 1.0;
	param.nu = 0.5;
	param.cache_size = 1000;
	param.C = 10.0;
	param.eps = 1e-3;
	param.p = 1.0;

	//svm_prob读取
	svm_problem svm_prob;
	
	int sampleNum = dataMat->rows;
	int vectorLength = dataMat->cols;

	svm_prob.l = sampleNum;
	svm_prob.y = new double [sampleNum];

	for (int i = 0; i < sampleNum; i++)
	{
		svm_prob.y[i] = cvmGet(labelMat,i,0);
	}

	cout<<"LibSVM middle"<<endl;
	svm_prob.x = new  svm_node * [sampleNum];

	for (int i = 0; i < sampleNum; i++)
	{
		svm_node * x_space = new svm_node [vectorLength + 1];
		for (int j = 0; j < vectorLength; j++)
		{
			x_space[j].index = j;
			x_space[j].value = cvmGet(dataMat,i,j);
			
		}
		x_space[vectorLength].index = -1;//注意，结束符号，一开始忘记加了

		svm_prob.x[i] = x_space;
	}

	cout<<"LibSVM end"<<endl;
	svm_model * svm_model = svm_train(&svm_prob, &param);
#ifdef srcfeature
	svm_save_model("libsvm_minist_src_feature_model_.model",svm_model);
#else
	svm_save_model("libsvm_minist_model.model",svm_model);
#endif
	for (int i=0 ; i < sampleNum; i++)
	{
		delete [] svm_prob.x[i];
	}

	delete [] svm_prob.y;
	svm_free_model_content(svm_model);
}

//***************************************************************
// 名称:    testSVM
// 功能:    测试opencv训练的SVM准确率
// 权限:    public 
// 返回值:  void
// 参数:    vector<string> testImageList
// 参数:    string SVMModel
//***************************************************************
void testSVM(vector<string> testImageList, string SVMModel)
{
	CvSVM svm;
	svm.load(SVMModel.c_str());//加载模型文件

	IplImage* testImage;  
	IplImage* tempImage;
	char buffer[512]; 

	ofstream ResultOutput( "predict_result.txt" );//把预测结果存储在这个文本中   
	for( int j = 0; j != testImageList.size(); j++ )//依次遍历所有的待检测图片    
	{    
		testImage = cvLoadImage( (testBasePath+testImageList[j]).c_str(), 1);    
		if( testImage == NULL )    
		{    
			cout<<" can not load the image: "<<(testBasePath+testImageList[j]).c_str()<<endl;    
			continue;    
		}
		tempImage =cvCreateImage(cvSize(20,20),8,3);
		cvZero(tempImage);    
		cvResize(testImage,tempImage);    
		HOGDescriptor *hog=new HOGDescriptor(cvSize(20,20),cvSize(10,10),cvSize(5,5),cvSize(5,5),9);       
		vector<float>descriptors; 

		hog->compute(tempImage, descriptors,Size(1,1), Size(0,0));       
		CvMat* TempMat=cvCreateMat(1,descriptors.size(),CV_32FC1);    
		int n=0;    
		for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
		{    
			cvmSet(TempMat,0,n,*iter);    
			n++;    
		}       

		int resultLabel = svm.predict(TempMat);//检测结果
		sprintf( buffer, "%s  %d\r\n",testImageList[j].c_str(),resultLabel );
		ResultOutput<<buffer;  
	}
	cvReleaseImage(&testImage);
	cvReleaseImage(&tempImage);
	ResultOutput.close();   
	cout<<"SVM Predict Complete"<<endl;
}

//***************************************************************
// 名称:    testLibSVM
// 功能:    测试LisbSVM训练的模型的分类性能
// 权限:    public 
// 返回值:  void
// 参数:    string LibSVMModelFile
// 参数:    vector<string> testImageList
// 参数:    string SVMModel
//***************************************************************
void testLibSVM(string LibSVMModelFile, vector<string> testImageList, string SVMModel)
{

	svm_model * svm = svm_load_model(LibSVMModelFile.c_str());

	IplImage* testImage;  
	IplImage* tempImage;
	char buffer[512]; 

	ofstream ResultOutput( "libsvm_predict_result.txt" ); 
	for( int j = 0; j != testImageList.size(); j++ )//依次遍历所有的待检测图片    
	{    
		testImage = cvLoadImage( (testBasePath+testImageList[j]).c_str(), 1);    
		if( testImage == NULL )    
		{    
			cout<<" can not load the image: "<<(testBasePath+testImageList[j]).c_str()<<endl;    
			continue;    
		}
		tempImage =cvCreateImage(cvSize(20,20),8,3);
		cvZero(tempImage);    
		cvResize(testImage,tempImage);    
		HOGDescriptor *hog=new HOGDescriptor(cvSize(20,20),cvSize(10,10),cvSize(5,5),cvSize(5,5),9);       
		vector<float>descriptors; 

		hog->compute(tempImage, descriptors,Size(1,1), Size(0,0));  

		svm_node * inputVector = new svm_node [ descriptors.size()+1];
		int n = 0;
		for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
		{     
			inputVector[n].index = n;
			inputVector[n].value = *iter;
			n++;
		}       
		inputVector[n].index = -1;
		
		int resultLabel = svm_predict(svm,inputVector);//分类结果
		sprintf( buffer, "%s  %d\r\n",testImageList[j].c_str(),resultLabel );
		ResultOutput<<buffer;  
		delete [] inputVector;
	}
	svm_free_model_content(svm);
	cvReleaseImage(&testImage);
	cvReleaseImage(&tempImage);
	ResultOutput.close();   
	cout<<"SVM Predict Complete"<<endl;
}

//***************************************************************
// 名称:    releaseAll
// 功能:    释放相应的资源
// 权限:    public 
// 返回值:  void
//***************************************************************
void releaseAll()
{

	cvReleaseMat( &dataMat ); 
	cvReleaseMat( &labelMat);
	cout<<"Release All Complete"<<endl;
}

//***************************************************************
// 名称:    main
// 功能:    这里用了两种SVM，一种是opencv中的，一种是libsvm中的，训练测试需要选择相对应的svm
// 权限:    public 
// 返回值:  int
//***************************************************************
int main()
{

	readTrainFileList(trainImageFile,trainBasePath,trainImageList,trainLabelList);      
	processHogFeature(trainImageList,trainLabelList, dataMat,labelMat);
	//trainSVM(dataMat,labelMat );
	//processNonFeature(trainImageList,trainLabelList, dataMat,labelMat);
	trainLibSVM(dataMat,labelMat);
	//readTestFileList( testImageFile,  testBasePath, testImageList);
	testLibSVM("libsvm_minist_model.model",testImageList,SVMModel);
	//testSVM( testImageList, SVMModel);
	releaseAll();
	return 0;
}

