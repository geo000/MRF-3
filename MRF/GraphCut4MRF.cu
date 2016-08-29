//#include"GraphCut4MRF.h"
//
//#include"cuda_utility.cuh"
//
//GraphCut4MRF::~GraphCut4MRF()
//{
//	if (m_gc) delete m_gc;
//	if (m_datacost_functor)     delete m_datacost_functor;
//	if (m_smoothcost_functor)   delete m_smoothcost_functor;
//	if (!m_source.empty())       m_source.clear();
//	if (!m_slic_labels.empty())  m_slic_labels.clear();
//	if (!m_dist_field.empty())   m_dist_field.clear();
//	//if (!m_labels)  m_labels.clear();
//
//}
//
//bool GraphCut4MRF::initGraph(void)
//{
//	//step 1: get scribble points and slic region points
//
//	//check for exists
//	if (TK::tk_is_file_existed("./subtotal/m_points.txt") && TK::tk_is_file_existed("./subtotal/m_slicPoints.txt"))
//	{
//
//		//TK::tk_elicit_points(m_points, "./subtotal/m_points.txt");
//		//TK::tk_elicit_points(m_slicPoints, "./subtotal/m_slicPoints.txt");
//
//	}
//	else
//	{
//
//
//
//		for (int i = 0; i < m_labels_num; ++i)
//		{
//			std::vector<cv::Point> points;
//			std::vector<cv::Point> slicPoints;
//
//			for (int j = 0; j < m_numpixels; ++j)
//			{
//				int x, y;
//				decode(x, y, j);
//				 
//				uchar ch1 = (m_pointMask[i].at<uchar>(y, x));
//				if (ch1 < MAX_VALUE)
//					points.push_back(cv::Point(y, x));
//
//				uchar ch2 = (m_dist_field[i].at<uchar>(y, x));
//				if (ch2 < MAX_VALUE)
//					slicPoints.push_back(cv::Point(y, x));
//
//			}
//
//			printf("points size = %d  ,slic points size = %d \n", points.size(), slicPoints.size());
//			if (points.size() < 10) { points.clear(); slicPoints.clear(); }
//
//
//			m_points.push_back(points);
//			m_slicPoints.push_back(slicPoints);
//
//		}
//
//		dump
//
//		TK::tk_dump_points(m_points, "./subtotal/m_points.txt");
//		TK::tk_dump_points(m_slicPoints, "./subtotal/m_slicPoints.txt");
//
//
//	}
//
//
//
//
//	printf("Initial  Graph done!\n");
//
//
//
//
//
//	return true;
//}
//
//void GraphCut4MRF::graphCutOptimization(int max_iter, std::string tag, SolverMethod solver)
//{
//	try{
//
//		//showResult(true);
//
//		float e1 = m_gc->giveDataEnergy();
//		float e2 = m_gc->giveSmoothEnergy();
//
//		float energy = m_gc->compute_energy();
//
//		printf("Before Optimization energy is : %.3f : %.3f + %.3f\n", energy, e1, e2);
//
//		if (m_label_order)
//		{
//			srand(time(NULL));
//			m_gc->setLabelOrder(1);
//		}
//
//
//
//		m_gc->expansion(max_iter);
//		//m_gc->swap(max_iter);
//
//
//		e1 = m_gc->giveDataEnergy();
//		e2 = m_gc->giveSmoothEnergy();
//		energy = m_gc->compute_energy();
//
//		printf("After Optimization energy is : %.3f : %.3f + %.3f \n", energy, e1, e2);
//
//
//		for (int s = 0; s < m_numpixels; ++s)
//		{
//			int label = m_gc->whatLabel(s);
//			int x, y;
//			decode(x, y, s);
//			m_labels[y][x] = label;
//
//		}
//
//
//	}
//	catch (GCException& e){
//		e.Report();
//	}
//
//}
//
//void GraphCut4MRF::showResult(bool saveimage, const std::string& filename)
//{
//	int nr = m_result.rows;
//	int nc = m_result.cols;
//
//	for (int i = 0; i < nr; ++i)
//	{
//		for (int j = 0; j < nc; j++)
//		{
//			m_result.at<cv::Vec3b>(i, j) = m_source[m_labels[i][j]].at<cv::Vec3b>(i, j);
//
//		}
//
//	}
//
//	cv::imshow("result", m_result);
//
//	if (saveimage)
//		TK::tk_save_img(m_result, filename.c_str());
//
//	cv::waitKey(0);
//
//}
//
//MyDataCostFunctor::~MyDataCostFunctor()
//{
//	//if (!m_source.empty())       m_source.clear();
//	//if (!m_slic_labels.empty())  m_slic_labels.clear();
//	//if (!m_dist_field.empty())   m_dist_field.clear();
//}
//
//bool MyDataCostFunctor::initUnaryCost(void)
//{
//	if (m_gc_controller != NULL)
//	{
//
//
//		if (TK::tk_is_file_existed("./subtotal/m_unarycost.txt"))
//		{
//			TK::tk_elicit_malloc(m_unarycost, m_gc_controller->m_labels_num, m_gc_controller->m_numpixels, "./subtotal/m_unarycost.txt");
//			return true;
//		}
//		else
//		{
//
//			std::vector< std::vector<cv::Point> > temp_points;
//
//			std::copy(m_gc_controller->m_points.begin(), m_gc_controller->m_points.end(), std::back_inserter(temp_points));  // copy  duplicate  to temp
//
//
//			for (int i = 0; i < m_gc_controller->m_labels_num; ++i)
//			{
//				int len = temp_points[i].size();
//
//				if (len < 10) continue;
//
//				for (int j = 0; j < len; ++j)
//				{
//					int y = temp_points[i][j].x;
//					int x = temp_points[i][j].y;
//
//					int s = m_gc_controller->encode(x, y);
//
//					for (int k = 0; k < m_gc_controller->m_labels_num; ++k)
//					{
//						if (i == k) continue;
//						m_unarycost[k][s] = INF_VALUE;
//
//					}
//
//
//				}
//
//			}
//
//
//			dump 
//			TK::tk_dump_malloc(m_unarycost, m_gc_controller->m_labels_num, m_gc_controller->m_numpixels, "./subtotal/m_unarycost.txt");
//
//
//		}
//
//		return  true;
//
//	}
//
//	return false;
//
//}
//
//GCoptimization::EnergyTermType MyDataCostFunctor::compute(int s, int l)
//{
//	return m_unarycost[l][s];
//}
// 
//MySmoothCostFunctor::~MySmoothCostFunctor()
//{
//	//if (!m_source.empty())       m_source.clear();
//	//if (!m_slic_labels.empty())  m_slic_labels.clear();
//	//if (!m_dist_field.empty())   m_dist_field.clear();
//}
//
//GCoptimization::EnergyTermType MySmoothCostFunctor::compute(int s1, int s2, int l1, int l2)
//{
//
//	if (l1 == l2) return 0;
//
//	GCoptimization::EnergyTermType*  neighbors[4];
//	int*   Flags[4];
//
//	GetSiteNeighbors(s1, l1, neighbors[0], Flags[0]);
//	GetSiteNeighbors(s1, l2, neighbors[1], Flags[1]);
//	GetSiteNeighbors(s2, l1, neighbors[2], Flags[2]);
//	GetSiteNeighbors(s2, l2, neighbors[3], Flags[3]);
//
//
//	GCoptimization::EnergyTermType result = GetPatchDiff(neighbors[0], neighbors[1], Flags[0], Flags[1]);
//
//	result += GetPatchDiff(neighbors[2], neighbors[3], Flags[2], Flags[3]);
//
//
//	return result;
//
//
//	//float p11_1 = m_source[l1].data[s1 * 3];
//	//float p11_2 = m_source[l1].data[s1 * 3 + 1];
//	//float p11_3 = m_source[l1].data[s1 * 3 + 2];
//
//	//float p12_1 = m_source[l1].data[s2 * 3];
//	//float p12_2 = m_source[l1].data[s2 * 3 + 1];
//	//float p12_3 = m_source[l1].data[s2 * 3 + 1];
//
//	//float p21_1 = m_source[l2].data[s1 * 3];
//	//float p21_2 = m_source[l2].data[s1 * 3 + 1];
//	//float p21_3 = m_source[l2].data[s1 * 3 + 2];
//
//
//	//float p22_1 = m_source[l2].data[s2 * 3];
//	//float p22_2 = m_source[l2].data[s2 * 3 + 1];
//	//float p22_3 = m_source[l2].data[s2 * 3 + 2];
//
//	////float  scale1 = 1.0, scale2 = 1.0;
//
//	////if (m_gc_controller->m_edges[l1].data[s1] == 255) scale1 = 0.2;
//	////if (m_gc_controller->m_edges[l2].data[s2] == 255) scale2 = 0.2;
//
//	//float cost1 = (p11_1 - p21_1)*(p11_1 - p21_1) + (p11_2 - p21_2)*(p11_2 - p21_2) + (p11_3 - p21_3)*(p11_3 - p21_3);
//	//float cost2 = (p12_1 - p22_1)*(p12_1 - p22_1) + (p12_2 - p22_2)*(p12_2 - p22_2) + (p12_3 - p22_3)*(p12_3 - p22_3);
//
//
//	//return (cost1+cost2);
//
//
//}