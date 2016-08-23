

#include"GraphCut4MRF.h"






// 1.  edge detection ,sobel,canny,laplacian operator
int edge_detection(void)
{


	CHECK_GT(FLAGS_imageName.size(), 0) << " image to be detected should not be empty..";

	//std::cout << FLAGS_imageName << std::endl;
	cv::Mat m = cv::imread(FLAGS_imageName, CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat gray, result;

	if (!m.data)
	{
		std::cout << "read image error,aborting.." << std::endl;
		return -1;
	}

	if (FLAGS_edgeSolver == "canny")	
		cv::Canny(m, result, 50, 150,3);	
	else if (FLAGS_edgeSolver == "sobel")	
		cv::Sobel(m, result, m.depth(), 1, 1);
	else if (FLAGS_edgeSolver == "laplacian")
		cv::Laplacian(m,result,m.depth());
	else
		LOG(ERROR) << " No other edge detection method.";
	

	if (FLAGS_showEdge)
	{
		cv::imshow("edge detection.", result);
		cv::waitKey(0);
	}

	

	if (FLAGS_dumpImage)
	{
		CHECK_GT(FLAGS_dumpName.size(), 0) << "dump name should not be empty.";

		cv::imwrite(FLAGS_dumpName, result);
	}
	return  1;
}
doRegisteration(edge_detection);

int device_query(void)
{
	LOG(INFO) << "Queryings GPUS " << FLAGS_gpu;

	std::vector<int> gpus;

	return 1;
}
doRegisteration(device_query);



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
		"	edge_detection  detect edges of source image using(canny,sobel,laplacian)\n "
		"	graphcut  solve a graphcut problem using alpha-expansion or alpha_beta swap"
		"	device_query  show GPU diagnostic information\n");


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