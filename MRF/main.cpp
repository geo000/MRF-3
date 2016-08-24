

#include"GraphCut4MRF.h"

#include"clc_functions.h"



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

int main(int argc,char** argv)
{
	FLAGS_alsologtostderr = 1;

	gflags::SetUsageMessage("command line actions\n"
		"usage: MRF <command> <args>\n\n"
		"commands:\n"
		"	edge_detection\n(detect edges of source image using canny,sobel,laplacian )\n "
		"	graph_cut\n(solve a graph_cut problem using alpha-expansion or alpha_beta swap)\n"
		"	device_query\n(show GPU diagnostic information)\n"
		"   slic \n (superpixel of an image)\n"
		"   extract_feature \n(extract kinds of features)\n"
		"   blending \n(pyramid blending , gradient domain blending..)\n");


	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);
	google::InstallFailureSignalHandler();

	if (argc == 2)
		return getCommandFunction(std::string(argv[1]))();
	else
	{
		gflags::ShowUsageWithFlagsRestrict(argv[0], "../data");
	}

	//std::cout << FLAGS_infolder << std::endl;

	//MatArray m_source, m_slic, m_dist,m_pointMask;
	
	//GetMatArray("./data_H1_1/source.txt", m_source);

	//GetMatArray("./data_H1_1/labels.txt", m_slic);

	//GetMatArray("./data_H1_1/distField.txt", m_dist);

	//GetMatArray("./data_H1_1/pointMask.txt", m_pointMask);



	//GraphCut4MRF* m_GraphCut = new GraphCut4MRF(m_source, m_slic, m_dist,m_pointMask);

	//m_GraphCut->edgeDetect();

	//m_GraphCut->initGraph("./data_H1_1/InitLabel.png");
	//m_GraphCut->saveMask("./subtotal/InitMask.png");

	//m_GraphCut->graphCutOptimization(1, "");
 //      
	//m_GraphCut->showResult();

	//m_GraphCut->saveMask("./subtotal/mask_patch.png");

	//TK::tk_save_img(m_GraphCut->GetResult());

	//system("pause");




	return 0;
}