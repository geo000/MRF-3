#pragma once

#include"Utility.h"
#include"GCoptimization.h"




class MyDataCostFunctor :public GCoptimization::DataCostFunctor
{
public:
	MyDataCostFunctor(const MatArray& source, const MatArray& slic_labels, const MatArray& dist_field) :
		m_source(source), m_slic_labels(slic_labels), m_dist_field(dist_field)
	{
	    m_gc_controller = NULL;

		m_unarycost = NULL;

		//assert(source.size() || slic_labels.size() || dist_field.size() );
		//assert((source.size() == slic_labels.size()) || (source.size() == dist_field.size()) || (slic_labels.size() == dist_field.size()));

		int num = source.size();
		int site = source[0].cols*source[0].rows;

		m_unarycost = (GCoptimization::EnergyTermType**)malloc(num*sizeof(GCoptimization::EnergyTermType*));
		for (int i = 0; i < num; ++i)
		{
			m_unarycost[i] = (GCoptimization::EnergyTermType*)malloc(site*sizeof(GCoptimization::EnergyTermType));

			//memset(m_unarycost[i], 0, site*sizeof(GCoptimization::EnergyTermType));	
		}
		TK::tk_memset(m_unarycost, num, site);
		TK::tk_check(m_unarycost, num, site);


		//TK::tk_memset();
	}

	~MyDataCostFunctor();


	inline void setGCController(GraphCut4MRF* const& m_gc)
	{
		m_gc_controller = m_gc;
		//m_width = m_gc->m_width;
		//m_height = m_gc->m_height;
	}

	bool initUnaryCost(void);



	virtual GCoptimization::EnergyTermType compute(int s, int l);

public:

	//friend class GraphCut4MRF;

private:

	MatArray m_source;
	MatArray m_slic_labels;
	MatArray m_dist_field;



	GraphCut4MRF*  m_gc_controller;

	GCoptimization::EnergyTermType** m_unarycost;


	DISABLE_COPY_AND_ASSIGN(MyDataCostFunctor);
};
class MySmoothCostFunctor :public GCoptimization::SmoothCostFunctor
{
public:
	MySmoothCostFunctor(const MatArray& source, const MatArray& slic_labels, const MatArray& dist_field) :
		m_source(source), m_slic_labels(slic_labels), m_dist_field(dist_field)
	{



	}
	~MySmoothCostFunctor();

	void setGCController(GraphCut4MRF* const& m_gc)
	{
		m_gc_controller = m_gc;
	}
	   
	 
	virtual GCoptimization::EnergyTermType compute(int s1, int s2, int l1, int l2);

	inline void GetSiteNeighbors(int s, int l, GCoptimization::EnergyTermType* & neighbors, int*& Flags)
	{
		neighbors = (GCoptimization::EnergyTermType*)malloc(9 * 3 * sizeof(GCoptimization::EnergyTermType));
		Flags = (int*)malloc(9 * 3 * sizeof(int));

		memset(neighbors, 0, 9 * 3 * sizeof(GCoptimization::EnergyTermType));
		memset(Flags, 0, 9 * 3*sizeof(int));

		int w = m_source[0].cols;

		int step_size[9] = { s - w - 1, s - w, s - w + 1, s - 1, s, s + 1, s + w - 1, s + w, s + w + 1 };

		//m_gc_controller->decode(x, y, s);


		if (m_gc_controller != NULL)
		{
			for (int i = 0; i < 9; ++i)
			{
				if (step_size[i] < 0 || step_size[i] >= m_source[0].cols*m_source[0].rows)
				{
					continue;
				}

				neighbors[i * 3] = m_source[l].data[step_size[i] * 3];
				neighbors[i * 3 + 1] = m_source[l].data[step_size[i] * 3 + 1];
				neighbors[i * 3 + 2] = m_source[l].data[step_size[i] * 3 + 2];

				Flags[i * 3] = 1;
				Flags[i * 3 + 1] = 1;
				Flags[i * 3 + 2] = 1;
			}




		}


	}
	inline void GetSiteNeighbors(int x, int y, int l, GCoptimization::EnergyTermType* & neighbors)
	{
		neighbors = (GCoptimization::EnergyTermType*)malloc(4 * sizeof(GCoptimization::EnergyTermType));

		if (m_gc_controller != NULL)
		{




		}


	}

	inline GCoptimization::EnergyTermType GetPatchDiff(const GCoptimization::EnergyTermType*  neighbors1, const GCoptimization::EnergyTermType*  neighbors2, const int*  Flags1, const int*  Flags2)
	{
		float weight[9] = { 0.0751, 0.1238, 0.0751, 0.1238, 0.2042, 0.1238, 0.0751, 0.1238, 0.0751 }; //sigma = 1

		GCoptimization::EnergyTermType result = 0.0f;

		for (int i = 0; i < 9; ++i)
		{
			result += weight[i] * Flags1[i * 3] * Flags2[i * 3] * (neighbors1[i * 3] - neighbors2[i * 3])*(neighbors1[i * 3] - neighbors2[i * 3]);
			result += weight[i] * Flags1[i * 3 + 1] * Flags2[i * 3 + 1] * (neighbors1[i * 3 + 1] - neighbors2[i * 3 + 1])*(neighbors1[i * 3 + 1] - neighbors2[i * 3 + 1]);
			result += weight[i] * Flags1[i * 3 + 2] * Flags2[i * 3 + 2] * (neighbors1[i * 3 + 2] - neighbors2[i * 3 + 2])*(neighbors1[i * 3 + 2] - neighbors2[i * 3 + 2]);

		}

		return result;

	}


private:

	MatArray m_source;
	MatArray m_slic_labels;
	MatArray m_dist_field;

	GraphCut4MRF* m_gc_controller;

	DISABLE_COPY_AND_ASSIGN(MySmoothCostFunctor);
};


class GraphCut4MRF
{
public:
	GraphCut4MRF(const MatArray& source, const MatArray& slic_labels, const MatArray& dist_field,const MatArray& pointMask):
		m_source(source), m_slic_labels(slic_labels), m_dist_field(dist_field), m_pointMask(pointMask), m_label_order(true)
	{

		// var definition
		
		assert(source.size() || slic_labels.size() || dist_field.size() || pointMask.size());
		assert((source.size() == slic_labels.size()) || (source.size() == dist_field.size()) || (slic_labels.size() == dist_field.size()));

		m_labels_num = source.size();


		m_width = source[0].cols;  // x index-----> width ----> cols
		m_height = source[0].rows; // y index-----> height ----> rows
		
		m_result.create(m_height, m_width, source[0].type());

		m_numpixels = m_width*m_height;
		
		m_labels.resize(m_height);
		//m_labels_minimum_energy.resize(m_height);
		

		// assign  labels  and corresponding value
		//GCoptimization::EnergyTermType max_label_energy = m_width*m_width + m_height*m_height;
		for (int i = 0; i < m_height; ++i)
		{
			m_labels[i].resize(m_width);
		//	m_labels_minimum_energy[i].resize(m_width);

		//	for (int j = 0; j < m_width; ++j)
		//	{
		//		m_labels_minimum_energy[i][j] = max_label_energy;
		//	}
		//	
		}

		//m_datacost = (GCoptimization::EnergyTermType**)malloc(m_labels_num * sizeof(GCoptimization::EnergyTermType*));
		//for (int i = 0; i < m_labels_num; ++i)
		//{
		//	m_datacost[i] = (GCoptimization::EnergyTermType*)malloc(m_numpixels * sizeof(GCoptimization::EnergyTermType));
		//	memset(m_datacost[i], 0, sizeof(m_numpixels * sizeof(GCoptimization::EnergyTermType)));

		//	
		//}
		//TK::tk_memset(m_datacost, m_labels_num, m_numpixels);
		//TK::tk_check(m_datacost, m_labels_num, m_numpixels);

		// assign  labels  and corresponding value

		// optimization graph initialization

		initGraph();

		//m_gc->setLabelOrder(true);
		
		m_gc = new GCoptimizationGridGraph(m_width, m_height, m_labels_num);
		
		m_datacost_functor = new MyDataCostFunctor(source, slic_labels, dist_field);
		m_datacost_functor->setGCController(this);
		m_datacost_functor->initUnaryCost();

		m_smoothcost_functor = new MySmoothCostFunctor(source, slic_labels, dist_field);
		m_smoothcost_functor->setGCController(this);

		m_gc->setDataCostFunctor(m_datacost_functor);
		m_gc->setSmoothCostFunctor(m_smoothcost_functor);

		
	}
	~GraphCut4MRF();

public:
	enum SolverMethod
	{
		Alpha_Expansion,
		Alpha_Beta_Swap
	};

	bool  initGraph(void);

	inline void  initGraph(const char* filename)
	{
		cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);

		if (!m.data)
		{
			std::cout << "read image"<<filename<<"error,aborting.." << std::endl;
			return;
		}

		int nr = m.rows;
		int nc = m.cols;

		for (int i = 0; i < nr; ++i)
		{
			for (int j = 0; j < nc; j++)
			{
				m_labels[i][j]= m.at<uchar>(i, j);

			}

		}
		
		showResult();
	}
	
	void  graphCutOptimization(int max_iter, std::string tag, SolverMethod  solver = SolverMethod::Alpha_Expansion);

	void  showResult(bool saveimage = false);

public:

	inline void decode(int & x, int&y, const int s)
	{
		y = (int)(s / m_width);
		x = s%m_width;
	}

	inline int  encode(const int x,const int y)
	{
		return m_width*y + x;
	}

	inline cv::Mat& GetResult()
	{
		return m_result;
	}

	inline  void saveMask(const char* filename = "./subtotal/mask.png")
	{

		int nr = m_result.rows;
		int nc = m_result.cols;

		cv::Mat mask;
		mask.create(nr, nc, CV_8UC1);

		for (int i = 0; i < nr; ++i)
		{
			for (int j = 0; j < nc; j++)
			{
				//m_result.at<cv::Vec3b>(i, j) = m_source[m_labels[i][j]].at<cv::Vec3b>(i, j);
				if (m_labels[i][j] == 0) mask.at<uchar>(i, j) = 0;
				else
					mask.at<uchar>(i, j) = 255;
				//mask.at<uchar>(i, j) = (m_labels[i][j]);
			}

		}

		cv::imshow("result", mask);

		
		TK::tk_save_img(mask, filename);

		cv::waitKey(0);

	}

	inline void  edgeDetect(bool isShow = false,bool isSave = false)
	{
		assert(m_source.size());

		int num = m_source.size();

		//m_edges.resize(num);

		for (int i = 0; i < num; ++i)
		{
			cv::Mat src,dst;

			cv::cvtColor(m_source[i], src, CV_BGR2GRAY);

			//cv::Sobel(m_source[i], dst, m_source[i].depth(), 1, 1);
			cv::Canny(m_source[i], dst, 50, 150, 3);

			m_edges.push_back(dst);

		}

		if (isShow)
		{
			for (int i = 0; i < num; ++i)
			{
				cv::imshow("", m_edges[i]);
				cv::waitKey(0);
			}
		}
		if (isSave)
		{
			for (int i = 0; i < num; ++i)
			{
				std::string filename;
				
				filename.assign("./subtotal/").append(TK::tk_toString(i)).append(".png");
				cv::imwrite(filename, m_edges[i]);
			}
		}

	}

public:
	friend class MyDataCostFunctor;
	friend class MySmoothCostFunctor;

private:


	GCoptimizationGridGraph* m_gc;

	MyDataCostFunctor*   m_datacost_functor;
	MySmoothCostFunctor* m_smoothcost_functor;


	MatArray m_source;
	MatArray m_slic_labels;
	MatArray m_dist_field;
	MatArray m_pointMask;

	MatArray m_edges;

	int m_width;
	int m_height;

    int m_labels_num;

	bool m_label_order;
	std::vector<std::vector<int> > m_labels;
	
	GCoptimization::EnergyTermType** m_datacost;

	int m_numpixels;

	float m_energy;

	cv::Mat m_result;

	std::vector< std::vector<cv::Point> > m_points;
	std::vector< std::vector<cv::Point> > m_slicPoints;

	DISABLE_COPY_AND_ASSIGN(GraphCut4MRF);

};




