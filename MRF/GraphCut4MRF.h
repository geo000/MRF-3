#pragma once

#include"Utility.h"
#include"GCoptimization.h"



class MyDataCostFunctor :public GCoptimization::DataCostFunctor
{
public:

	MyDataCostFunctor(int labels_num, int sites_num) :m_labels_num(labels_num), m_sites_num(sites_num)
	{
		m_gc_controller = NULL;
		m_unarycost = NULL;


		m_unarycost = (GCoptimization::EnergyTermType**)malloc(m_labels_num*sizeof(GCoptimization::EnergyTermType*));
		for (int i = 0; i < m_labels_num; ++i)
			m_unarycost[i] = (GCoptimization::EnergyTermType*)malloc(m_sites_num*sizeof(GCoptimization::EnergyTermType));

			
		TK::tk_memset(m_unarycost, m_labels_num, m_sites_num);
		TK::tk_check(m_unarycost,  m_labels_num, m_sites_num);

	}
	~MyDataCostFunctor();


	inline void setGCController(GraphCut4MRF* const& m_gc)
	{
		m_gc_controller = m_gc;
	}
	

	bool initUnaryCost(std::string& dumpfolder);
	virtual GCoptimization::EnergyTermType compute(int s, int l);


private:


	int m_labels_num, m_sites_num;

	GraphCut4MRF*  m_gc_controller;

	GCoptimization::EnergyTermType** m_unarycost;


	DISABLE_COPY_AND_ASSIGN(MyDataCostFunctor);
};
class MySmoothCostFunctor :public GCoptimization::SmoothCostFunctor
{
public:
	MySmoothCostFunctor(const MatArray& source) :
		m_source(source)
	{
		m_labels_num = source.size();

		CHECK_GT(m_labels_num, 0) << "label num must greater that 0";

		m_width = m_source[0].cols;
		m_height = m_source[0].rows;

		m_sites_num = m_width * m_height;


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

		int step_size[9] = { s - m_width - 1, s - m_width, s - m_width + 1, s - 1, s, s + 1, s + m_width - 1, s + m_width, s + m_width + 1 };


		if (m_gc_controller != NULL)
		{
			for (int i = 0; i < 9; ++i)
			{
				if (step_size[i] < 0 || step_size[i] >= m_sites_num)
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
	
	int m_labels_num, m_sites_num,m_width,m_height;

	GraphCut4MRF* m_gc_controller;

	DISABLE_COPY_AND_ASSIGN(MySmoothCostFunctor);
};

class GraphCut4MRF
{
public:
	GraphCut4MRF(const std::string& infolder,const std::string& subtotal,const MatArray& source, const MatArray& slic_labels, const MatArray& dist_field,const MatArray& pointMask):
		m_infolder(infolder), m_subtotal(subtotal), m_source(source), m_slic_labels(slic_labels), m_dist_field(dist_field), m_pointMask(pointMask), m_label_order(true)
	{

		// var definition
		
		CHECK_GT(source.size(), 0);

		CHECK_EQ(source.size(), slic_labels.size()) << "source size != slic_labels size\n";
		CHECK_EQ(source.size(), dist_field.size()) << "source size != dist_field size\n";
		CHECK_EQ(slic_labels.size(), dist_field.size()) << "dist_field size != slic_labels size\n";

		m_labels_num = source.size();


		m_width = source[0].cols;  // x index-----> width ----> cols
		m_height = source[0].rows; // y index-----> height ----> rows
		
		m_result.create(m_height, m_width, source[0].type());

		m_numpixels = m_width*m_height;
		
		m_labels.resize(m_height);
		//m_labels_minimum_energy.resize(m_height);
		

		// assign  labels  and corresponding value
	
		for (int i = 0; i < m_height; ++i)		
			m_labels[i].resize(m_width);

		initGraph();
	
		m_gc = new GCoptimizationGridGraph(m_width, m_height, m_labels_num);
		
		m_datacost_functor = new MyDataCostFunctor(m_labels_num, m_numpixels);
		m_datacost_functor->setGCController(this);
		m_datacost_functor->initUnaryCost(m_subtotal);

		m_smoothcost_functor = new MySmoothCostFunctor(source);
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

	void  initGraph(const std::string & filename);
	
	void  graphCutOptimization(int max_iter, std::string tag, SolverMethod  solver = SolverMethod::Alpha_Expansion);

	void  showResult(bool saveimage = false, const std::string& filename="");

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

	inline  void saveMask(const std::string filename)
	{

		int nr = m_result.rows;
		int nc = m_result.cols;

		cv::Mat mask;
		mask.create(nr, nc, CV_8UC1);

		int step = 255 / m_labels_num;

		CHECK_GT(step,0)<<"step  must be greater than 0";
		CHECK_LE(step, 255) << "step must be less than 255";

		for (int i = 0; i < nr; ++i)
		{
			for (int j = 0; j < nc; j++)
			{
				mask.at<uchar>(i, j) = (m_labels[i][j]) * step;
			}

		}

		cv::imshow("result", mask);

		
		TK::tk_save_img(mask, filename.c_str());

		cv::waitKey(0);

	}



public:
	friend class MyDataCostFunctor;
	friend class MySmoothCostFunctor;

private:


	GCoptimizationGridGraph* m_gc;

	MyDataCostFunctor*   m_datacost_functor;
	MySmoothCostFunctor* m_smoothcost_functor;

	std::string m_infolder;
	std::string m_subtotal;
	MatArray m_source;
	MatArray m_slic_labels;
	MatArray m_dist_field;
	MatArray m_pointMask;


	int m_width;
	int m_height;

    int m_labels_num;

	bool m_label_order;
	std::vector<std::vector<int> > m_labels;
	

	int m_numpixels;
	cv::Mat m_result;
	PointsArrays m_points;
	PointsArrays m_slicPoints;

	DISABLE_COPY_AND_ASSIGN(GraphCut4MRF);

};




