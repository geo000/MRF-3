
#ifndef _CLC_UTILITY_H_
#define _CLC_UTILITY_H_
#include <io.h>

#include<opencv2\opencv.hpp>
#include<sstream>

namespace TK
{
	static  bool tk_is_file_existed(const char* filename)
	{
		if (_access(filename, 0) != -1)
		{
			std::cout << "File " << filename << " already exists.\n" << std::endl;
			return true;
		}

		return false;
	}

	template<class T>
	static bool tk_normalize(T** data, int Y,int X)
	{
		T max_value = -1;
		T min_value = 100000000;
		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				max_value = std::max(max_value, data[y][x]);
				min_value = std::min(min_value, data[y][x]);
			}

		}
		
		assert(max_value > 0);
		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				data[y][x] = (data[y][x] / max_value) * 255;
			}

		}

		printf("Normalize Done,max = %f , min = %f\n", max_value,min_value);
		return true;
	}


	template<class T>
	static bool tk_check(T** data, int Y, int X)
	{

		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				if (data[y][x] < 0 )

					printf("data[%d][%d] = %f \n", y,x,data[y][x]);
			}

		}

		printf("check Done\n");
		return true;
	}

	template<class T>
	static bool tk_memset(T** data, int Y, int X)
	{

		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				data[y][x] = 0;
			}

		}

		printf("memset  Done\n");
		return true;
	}


	static bool tk_save_img(const cv::Mat img,const char* filename = "./subtotal/result.png")
	{
		if (!img.data)
		{
			printf("The image to be saved is empty.\n");
			return false;
		}

		cv::imwrite(filename, img);

		return true;
	}

	template<class T>
	static bool tk_dump_vec(const std::vector<std::vector<T>> data, int Y, int X,const char* filename)
	{
		std::fstream out(filename,std::ios::out);

		if (!out)
		{
			std::cout << "Open File Error" << std::endl;
			return false;
		}

		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				out << data[y][x] << " ";
			}
			out << std::endl;
		}
		out.close();

		printf("dump file Done --->%s\n",filename);
		return true;
	}

	template<class T>
	static bool tk_elicit_vec(std::vector<std::vector<T>> & data, int Y, int X, const char* filename)
	{
		std::fstream in(filename, std::ios::in);

		if (!in)
		{
			std::cout << "Open File Error" << std::endl;
			return false;
		}

		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				in >> data[y][x] ;
			}
			
		}
		in.close();

		printf("get file Done --->%s\n", filename);
		return true;
	}


	static bool tk_dump_points(const std::vector<std::vector<cv::Point>> data, const char* filename)
	{
		std::fstream out(filename, std::ios::out);

		if (!out)
		{
			std::cout << "Open File Error" << std::endl;
			return false;
		}

		for (int i = 0; i < data.size(); ++i)
		{
			out << data[i].size() << std::endl;

			for (int j = 0; j < data[i].size(); ++j)
			{
				out << data[i][j].x << " " << data[i][j].y << std::endl;
			}

		}

		out.close();

		printf("dump points Done --->%s\n", filename);

		return true;
	}


	static bool tk_elicit_points(std::vector<std::vector<cv::Point>> & data,const char* filename)
	{
		std::fstream in(filename, std::ios::in);

		if (!in)
		{
			std::cout << "Open File Error" << std::endl;
			return false;
		}

		int num = 0;
	    
		while (in >> num)
		{
			assert(num);
			std::vector<cv::Point> slicPoints;
			slicPoints.resize(num);
			for (int i = 0; i < num; ++i)
			{
				in >> slicPoints[i].x >> slicPoints[i].y;
			}
			data.push_back(slicPoints);

		}

		in.close();

		printf("get points Done --->%s\n", filename);

		return true;
	}

	template<class T>
	static bool tk_dump_malloc(T** data, int Y, int X, const char* filename)
	{
		std::fstream out(filename, std::ios::out);

		if (!out)
		{
			std::cout << "Open File Error" << std::endl;
			return false;
		}

		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				out << data[y][x] << " ";
			}
			out << std::endl;
		}
		out.close();

		printf("dump file Done --->%s\n", filename);
		return true;
	}

	template<class T>
	static bool tk_elicit_malloc(T** & data, int Y, int X, const char* filename)
	{
		std::fstream in(filename, std::ios::in);

		if (!in)
		{
			std::cout << "Open File Error" << std::endl;
			return false;
		}

		for (int y = 0; y < Y; ++y)
		{
			for (int x = 0; x < X; ++x)
			{
				in >> data[y][x];
			}

		}
		in.close();

		printf("get file Done --->%s\n", filename);
		return true;
	}


	template<class T>
	static std::string tk_toString(const T& t)
	{
		std::ostringstream oss;//创建一个流


		oss.clear();

		oss << t;//把值传递如流中

		return oss.str();//获取转换后的字符转并将其写入result

	}



}
#endif