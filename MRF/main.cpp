

#include"GraphCut4MRF.h"

#include"clc_functions.h"




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





	return 0;
}