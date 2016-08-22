
#include<fstream>

#include"GraphCut4MRF.h"


void GetMatArray(const char* filename,MatArray& output)
{
	std::string matfile;
	std::ifstream in(filename);

	while (in>>matfile)
	{
		cv::Mat m = cv::imread(matfile, CV_LOAD_IMAGE_UNCHANGED);

		if (!m.data)
		{
			std::cout << "read image error,aborting.."<<std::endl;
			break;
		}

		output.push_back(m);
	}

	in.close();
}

int main(void)
{

	MatArray m_source, m_slic, m_dist,m_pointMask;
	
	GetMatArray("./data_H1_1/source.txt", m_source);

	GetMatArray("./data_H1_1/labels.txt", m_slic);

	GetMatArray("./data_H1_1/distField.txt", m_dist);

	GetMatArray("./data_H1_1/pointMask.txt", m_pointMask);



	GraphCut4MRF* m_GraphCut = new GraphCut4MRF(m_source, m_slic, m_dist,m_pointMask);

	m_GraphCut->edgeDetect();

	m_GraphCut->initGraph("./data_H1_1/InitLabel.png");
	m_GraphCut->saveMask("./subtotal/InitMask.png");

	m_GraphCut->graphCutOptimization(1, "");
       
	m_GraphCut->showResult();

	m_GraphCut->saveMask("./subtotal/mask_patch.png");

	TK::tk_save_img(m_GraphCut->GetResult());

	//system("pause");




	return 0;
}