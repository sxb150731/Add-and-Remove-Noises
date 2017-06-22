#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace cv;
using namespace std;

//Count Apple
void countApple(Mat src)
{
	int count = 0;
	vector<vector<Point> > contours;  
	vector<Vec4i> hierarchy;  
	findContours(src,contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);  
  
	Mat contoursImage(src.rows,src.cols,CV_8U,Scalar(255));  
	for(int i=0;i<contours.size();i++)
	{  
    if(hierarchy[i][3]!=-1) 
		drawContours(contoursImage,contours,i,Scalar(0),20);  
	}  
	for (int i = 0; i < (int)contours.size(); i++)
	{
		contourArea(contours[i]);
		if (contourArea(contours[i]) > 1000)
			count++;
	}
	cout << "count of apple: " << count << endl;
}

//sort the window using insertion sort
void insertionSort(int window[])
{
    int temp, i , j;
    for(i = 0; i < 9; i++){
        temp = window[i];
        for(j = i-1; j >= 0 && temp < window[j]; j--){
            window[j+1] = window[j];
        }
        window[j+1] = temp;
    }
}

void createFilter(double gKernel[][3])
{
    // set standard deviation to 1.0
    double sigma = 0.15;
    double r, s = 2.0 * sigma * sigma;
	const double pi = 3.1415927;
 
    // sum is for normalization
    double sum = 0.0;
 
    // generate 3x3 kernel
    for (int x = 0; x <= 2; x++)
    {
        for(int y = 0; y <= 2; y++)
        {
            r = x*x + y*y;
            gKernel[x][y] = (exp(-r/s))/(pi * s);
            sum += gKernel[x][y];
        }
    }
    // normalize the Kernel
    for(int i = 0; i < 3; ++i)
	{
        for(int j = 0; j < 3; ++j)
		{
            gKernel[i][j] /= sum;
		}
	}
}

	//Salt  
	void salt(cv::Mat image_N2, int n) 
	{
		int i,j;  
		for (int k=0; k<n/2; k++) 
		{  
			i = std::rand()%image_N2.cols;   
			j = std::rand()%image_N2.rows;   
  
			if (image_N2.type() == CV_8UC3) 
			{
				image_N2.at<Vec3b>(j,i)[0]= 250; 
				image_N2.at<Vec3b>(j,i)[1]= 150;  
				image_N2.at<Vec3b>(j,i)[2]= 250;  
			}
		}  
	}  
  
	//Pepper  
	void pepper(cv::Mat image_N2, int n) 
	{
		int i,j;  
		for (int k=0; k<n/2; k++) 
		{
			i = std::rand()%image_N2.cols;  
			j = std::rand()%image_N2.rows;   
			if (image_N2.type() == CV_8UC3) 
			{
				image_N2.at<Vec3b>(j,i)[0]= 250;
				image_N2.at<Vec3b>(j,i)[1]= 150;  
				image_N2.at<Vec3b>(j,i)[2]= 150;     
			}
		}  
	}

int erosion_elem = 0;
int erosion_size = 0;

int main( int argc, char** argv )
{
	//Load and Display Original Image
	Mat image_I = imread("GreenRedApple.png", IMREAD_COLOR);
	if (! image_I.data ) 
           {
                    cout << "Could not open or find the image" << std::endl ;
                    return -1;
           }
	imshow( "Image_I", image_I ); 
	waitKey(0); 
//*******************************************************************************
	//Add Gaussian Noise
	Mat image_N1 = image_I.clone();
	Mat image_H1 = image_N1.clone();
	randn(image_N1, 0, 15);
	image_N1 = image_N1 + image_I;
	imshow( "Image_N1", image_N1); 	
	waitKey(0);
	
	
	for(int x = 1; x < image_H1.rows - 1; x++)
	{
		for(int y = 1; y < image_H1.cols - 1; y++)
		{
			for (int c = 0; c < 3; c++)
			{
				double gKernel[3][3];
				createFilter(gKernel);
				int window[9];
				window[0] = image_H1.at<Vec3b>(x-1,y-1)[c] * gKernel[0][0];
				window[1] = image_H1.at<Vec3b>(x-1,y)[c] * gKernel[0][1];
				window[2] = image_H1.at<Vec3b>(x-1,y+1)[c] * gKernel[0][2];
				window[3] = image_H1.at<Vec3b>(x,y-1)[c] * gKernel[1][0];
				window[4] = image_H1.at<Vec3b>(x,y)[c] * gKernel[1][1];
				window[5] = image_H1.at<Vec3b>(x,y+1)[c] * gKernel[1][2];
				window[6] = image_H1.at<Vec3b>(x+1,y-1)[c] * gKernel[2][0];
				window[7] = image_H1.at<Vec3b>(x+1,y)[c] * gKernel[2][1];
				window[8] = image_H1.at<Vec3b>(x+1,y+1)[c] * gKernel[2][2];

				window[4] = window[0] + window[1] + window[2] + window[3] + window[4] + window[5] + window[6] + window[7] + window[8];
			}
		}
	}

	imshow( "Image_H1", image_H1); 	
	waitKey(0);
//*******************************************************************************
	//Add Salt and Pepper Noise
	Mat image_N2 = image_I.clone();
	int num = 1000*622*0.02;
	salt(image_N2, num);
	pepper(image_N2, num);

	imshow( "Image_N2", image_N2); 	
	waitKey(0); 

	//Remove Salt and Pepper Noise
	int window[9];
	Mat image_H2 =  image_N2.clone();

	for(int x = 1; x < image_H2.rows - 1; x++)
	{
		for(int y = 1; y < image_H2.cols - 1; y++)
		{
			for (int c = 0; c < 3; c++)
			{
				window[0] = image_H2.at<Vec3b>(x-1 ,y-1)[c];
                window[1] = image_H2.at<Vec3b>(x-1, y)[c];
                window[2] = image_H2.at<Vec3b>(x-1, y+1)[c];
                window[3] = image_H2.at<Vec3b>(x, y-1)[c];
                window[4] = image_H2.at<Vec3b>(x, y)[c];
                window[5] = image_H2.at<Vec3b>(x, y+1)[c];
                window[6] = image_H2.at<Vec3b>(x+1, y-1)[c];
                window[7] = image_H2.at<Vec3b>(x+1, y)[c];
                window[8] = image_H2.at<Vec3b>(x+1, y+1)[c];

				// sort the window to find median
                insertionSort(window);

				// assign the median to centered element of the matrix
                image_H2.at<Vec3b>(x,y)[c] = window[4];
			  }
		}
	}
	//Left edge
	for(int x = 0; x < image_H2.rows-1; x++)
	{
		for (int c = 0; c < 3; c++)
			{
				window[0] = image_H2.at<Vec3b>(x,0)[c];
				window[1] = image_H2.at<Vec3b>(x, 1)[c];
				window[2] = image_H2.at<Vec3b>(x, 2)[c];
				window[3] = image_H2.at<Vec3b>(x, 0)[c];
				window[4] = image_H2.at<Vec3b>(x, 1)[c];
				window[5] = image_H2.at<Vec3b>(x, 2)[c];
				window[6] = image_H2.at<Vec3b>(x+1, 0)[c];
				window[7] = image_H2.at<Vec3b>(x+1, 1)[c];
				window[8] = image_H2.at<Vec3b>(x+1, 2)[c];

				insertionSort(window);

				image_H2.at<Vec3b>(x,0)[c] = window[4];
			}
	}

	//Top edge
	for(int y = 0; y < image_H2.cols-1; y++)
	{
		for (int c = 0; c < 3; c++)
			{
				window[0] = image_H2.at<Vec3b>(0,y)[c];
				window[1] = image_H2.at<Vec3b>(0, y)[c];
				window[2] = image_H2.at<Vec3b>(0, y+1)[c];
				window[3] = image_H2.at<Vec3b>(1, y)[c];
				window[4] = image_H2.at<Vec3b>(1, y)[c];
				window[5] = image_H2.at<Vec3b>(1, y+1)[c];
				window[6] = image_H2.at<Vec3b>(2, y)[c];
				window[7] = image_H2.at<Vec3b>(2, y)[c];
				window[8] = image_H2.at<Vec3b>(2, y+1)[c];

				insertionSort(window);

				image_H2.at<Vec3b>(0,y)[c] = window[4];
			}
	}

	imshow( "Image_H2", image_H2); 	
	waitKey(0); 
//*******************************************************************************
	//Add brightness by 50
	Mat image_B1 = image_I.clone();
	for (int x = 0; x < image_B1.rows; x++)
	{
		for (int y = 0; y < image_B1.cols; y++)
		{
			for (int c = 0; c < 3; c++)
			{
				image_B1.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(image_B1.at<Vec3b>(x, y)[c] + 50);
			}
		}
	}

	imshow("Image_B1", image_B1);
	waitKey(0);
//*******************************************************************************
	//Count image_I
	Mat image_I_Count = image_I.clone();
	cvtColor(image_I_Count, image_I_Count, CV_BGR2GRAY);
	threshold( image_I_Count, image_I_Count, 110, 255, 0);
	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(image_I_Count, image_I_Count, element1);
	cout << "Image_I: ";
	countApple(image_I_Count);

	//Count image_N1
	Mat image_N1_Count = image_N1.clone();
	cvtColor(image_N1_Count, image_N1_Count, CV_BGR2GRAY);
	threshold( image_N1_Count, image_N1_Count, 110, 255, 0);
	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(image_N1_Count, image_N1_Count, element2);
	cout << "Image_N1: ";
	countApple(image_N1_Count);

	//Count image_N2
	Mat image_N2_Count = image_N2.clone();
	cvtColor(image_N2_Count, image_N2_Count, CV_BGR2GRAY);
	threshold( image_N2_Count, image_N2_Count, 110, 255, 0);
	Mat element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(image_N2_Count, image_N2_Count, element3);
	cout << "Image_N2: ";
	countApple(image_N2_Count);

	//Count image_H1
	Mat image_H1_Count = image_H1.clone();
	cvtColor(image_H1_Count, image_H1_Count, CV_BGR2GRAY);
	threshold( image_H1_Count, image_H1_Count, 110, 255, 0);
	Mat element4 = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(image_H1_Count, image_H1_Count, element4);
	cout << "Image_H1: ";
	countApple(image_H1_Count);

	//Count image_H2
	Mat image_H2_Count = image_H2.clone();
	cvtColor(image_H2_Count, image_H2_Count, CV_BGR2GRAY);
	threshold( image_H2_Count, image_H2_Count, 115, 255, 0);
	Mat element5 = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(image_H2_Count, image_H2_Count, element5);
	cout << "Image_H2: ";
	countApple(image_H2_Count);

	//Count image_B1
	Mat image_B1_Count = image_B1.clone();
	cvtColor(image_B1_Count, image_B1_Count, CV_BGR2GRAY);
	threshold( image_B1_Count, image_B1_Count, 160, 255, 0);
	Mat element6 = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(image_B1_Count, image_B1_Count, element6);
	cout << "Image_B1: ";
	countApple(image_B1_Count);

	//imshow("Image_I_Count", image_I_Count);
	//waitKey(0);
	system("pause");
	return 0;
}
