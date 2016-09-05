#ifndef _CLC_MOUSE_H_
#define _CLC_MOUSE_H_

#include"Utility.h"

#define  ERROR_HANDLER_NAME  ("liygcheng")

DECLARE_string(scribble_type);

DEFINE_string(scribble_type,"line","scribble_type : line or area");

//
typedef int(*KeyboardAction)(void*);
typedef std::map<std::string, KeyboardAction> RegisterKeyboardActionMap;
extern RegisterKeyboardActionMap mouseActionMap;
RegisterKeyboardActionMap mouseActionMap;

#define RegisterMouseAction(str,fun)	\
		namespace {						\
		class _Register_Mouse_Action_##fun{			\
		public:	_Register_Mouse_Action_##fun()		\
						{							\
		mouseActionMap[#str] = &(fun);			\
						}							\
				};							\
		_Register_Mouse_Action_##fun m_register_mouse_action_##fun;\
				}							

static KeyboardAction  getMouseActionCommand(const std::string& name)
{
	if (mouseActionMap.count(name)){
		return mouseActionMap[name];
	}
	else
	{

		LOG(INFO) << "Available Actions:";
		for (RegisterKeyboardActionMap::iterator it = mouseActionMap.begin(); it != mouseActionMap.end(); ++it)
		{
			if (ERROR_HANDLER_NAME != it->first)
			LOG(INFO) << "\t" << it->first;
		}
		LOG(INFO) << "unknown actions :" << name;
		return mouseActionMap[ERROR_HANDLER_NAME];

	}

}

//Mouse CallBack Fucntion


typedef struct MouseData{
	  cv::Point m_current, m_start;
	  bool  lButtonDown ; 
	  bool  rButtonDown ;
	  int   scribbleRadius ;
	  std::string m_imageName, m_outfolder;
	  cv::Mat  m_source,m_scribble,m_fgmask,m_bgmask;
	  MouseData(const std::string& imageName,const std::string& outfolder):m_imageName(imageName),m_outfolder(outfolder){
		
		  m_source = cv::imread(imageName, CV_LOAD_IMAGE_UNCHANGED);

		  if (!m_source.data)
		  {
			  LOG(ERROR) << "image read error,check for imageName:"<<imageName;
			  LOG(ERROR) << "instantiation failure...";
			  //return NULL;
			  
		  }
	
		  m_scribble = m_source.clone();
		  
		  m_fgmask.create(2, m_source.size, CV_8UC1);
		  m_fgmask = 0;
		  m_bgmask.create(2, m_source.size, CV_8UC1);
		  m_bgmask = 0;
		  lButtonDown = false;
		  rButtonDown = false;
		  scribbleRadius = 3;

		  m_current.x = 0;
		  m_current.y = 0;
		  m_start.x = 0;
		  m_start.y = 0;


	  }
	  ~MouseData(){
		 if (!m_source.empty()) m_source.release();
		 if (!m_scribble.empty()) m_scribble.release();
		 if (!m_fgmask.empty()) m_fgmask.release();
		 if (!m_bgmask.empty()) m_bgmask.release();
	  }

	  //inline void setImageName(const std::string& imageName){

		 // m_imageName.assign(imageName);

		 // if (!m_source.empty()) m_source.release();
		 // if (!m_scribble.empty()) m_scribble.release();
		 // if (!m_fgmask.empty()) m_fgmask.release();
		 // if (!m_bgmask.empty()) m_bgmask.release();

		 // m_source = cv::imread(imageName, CV_LOAD_IMAGE_UNCHANGED);

		 // if (!m_source.data)
		 // {
			//  LOG(ERROR) << "image read error,check for imageName:" << imageName;
			//  LOG(ERROR) << "instantiation failure...";

		 // }

		 // m_scribble = m_source.clone();

		 // m_fgmask.create(2, m_source.size, CV_8UC1);
		 // m_fgmask = 0;
		 // m_bgmask.create(2, m_source.size, CV_8UC1);
		 // m_bgmask = 0;
		 // lButtonDown = false;
		 // rButtonDown = false;
		 // scribbleRadius = 5;
		 // 
	  //}

}MouseData;


static void onMouseScribble(int event, int x, int y, int flag, void* userData){

	//LOG(INFO) << "Mouse Callback Function..";

	MouseData* m_mouseData = static_cast<MouseData*>(userData);

	if (m_mouseData == NULL){
		std::cerr << "userData is empty.." << std::endl;
		return;
	}

	if (event == CV_EVENT_LBUTTONDOWN) m_mouseData->lButtonDown = true;
	else if (event == CV_EVENT_RBUTTONDOWN) m_mouseData->rButtonDown = true;
	else if (event == CV_EVENT_LBUTTONUP) m_mouseData->lButtonDown = false;
	else if (event == CV_EVENT_RBUTTONUP) m_mouseData->rButtonDown = false;
	else if (event == CV_EVENT_MOUSEMOVE){ // start to move 

		if (m_mouseData->lButtonDown){  //Background 

			cv::circle(m_mouseData->m_bgmask, cv::Point(x, y), m_mouseData->scribbleRadius, 255, -1);
			cv::circle(m_mouseData->m_scribble, cv::Point(x, y), m_mouseData->scribbleRadius, CV_RGB(0, 0, 255), -1);

		}
		else if (m_mouseData->rButtonDown)//Foreground
		{
			cv::circle(m_mouseData->m_fgmask, cv::Point(x, y), m_mouseData->scribbleRadius, 255, -1);
			cv::circle(m_mouseData->m_scribble, cv::Point(x, y), m_mouseData->scribbleRadius, CV_RGB(255, 0, 0), -1);
		}

	}
	cv::imshow("Scribble Image", m_mouseData->m_scribble);
	cv::imshow("foreground mask", m_mouseData->m_fgmask);
	cv::imshow("background mask", m_mouseData->m_bgmask);






}

//some global var  for  mouse  action


static void onMouseMatting(int event, int x,int y,int flag,void* userData){

	MouseData* m_mouseData = static_cast<MouseData*>(userData);

	if (m_mouseData == NULL){
		std::cerr << "userData is empty.." << std::endl;
		return;
	}

	if (event == CV_EVENT_LBUTTONDOWN) {
		m_mouseData->lButtonDown = true;
		m_mouseData->m_start.x = x;
		m_mouseData->m_start.y = y; 
		m_mouseData->m_current.x = x;
		m_mouseData->m_current.y = y; 

		//cv::imshow("Scribble Image", m_mouseData->m_scribble);
		//cv::imshow("background mask", m_mouseData->m_bgmask);
	}
	else if (event == CV_EVENT_RBUTTONDOWN) { 
		m_mouseData->rButtonDown = true; 
		m_mouseData->m_start.x = x; 
		m_mouseData->m_start.y = y; 
		m_mouseData->m_current.x = x;
		m_mouseData->m_current.y = y; 

		//cv::imshow("Scribble Image", m_mouseData->m_scribble);
		//cv::imshow("background mask", m_mouseData->m_bgmask);
	}
	else if (event == CV_EVENT_LBUTTONUP) {
		m_mouseData->lButtonDown = false;
		cv::Point tmp(x, y);
		cv::line(m_mouseData->m_scribble, m_mouseData->m_start, tmp, cv::Scalar(0, 255, 0), 2, 8, 0);
		cv::line(m_mouseData->m_bgmask, m_mouseData->m_start, tmp, 255, 2, 8, 0);
		//cv::floodFill(m_mouseData->m_scribble, m_mouseData->m_bgmask, tmp, 255);

		//FindContours(gray, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		cv::vector<cv::vector<cv::Point> > contours;
		cv::vector<cv::Vec4i> hierarchy;
		
		cv::findContours(m_mouseData->m_bgmask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		//cv::findContours(m_mouseData->m_bgmask, &m_contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size();++i)
		cv::drawContours(m_mouseData->m_bgmask, contours, i, cv::Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, cv::Point());
		//cvDrawContours(gray, contour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), -1, CV_FILLED, 8);
		//cv::drawContours()
		cv::imshow("Scribble Image", m_mouseData->m_scribble);
		cv::imshow("background mask", m_mouseData->m_bgmask);
	}
	else if (event == CV_EVENT_RBUTTONUP) {
		m_mouseData->rButtonDown = false; 
		cv::Point tmp(x, y);
		cv::rectangle(m_mouseData->m_scribble, m_mouseData->m_current, tmp, cv::Scalar(0, 255, 0), 2);
		cv::rectangle(m_mouseData->m_bgmask, m_mouseData->m_current, tmp, 255, CV_FILLED);

		cv::imshow("Scribble Image", m_mouseData->m_scribble);
		cv::imshow("background mask", m_mouseData->m_bgmask);
	}
	else if (event == CV_EVENT_MOUSEMOVE){ // start to move 

		if (m_mouseData->lButtonDown){  //Background 

			cv::Point tmp(x, y);
			cv::line(m_mouseData->m_scribble, m_mouseData->m_current, tmp, cv::Scalar(0, 255, 0), 2, 8, 0);
			cv::line(m_mouseData->m_bgmask, m_mouseData->m_current, tmp, 255, 2, 8, 0);
			m_mouseData->m_current.x = x;
			m_mouseData->m_current.y = y;

			cv::imshow("Scribble Image", m_mouseData->m_scribble);
			cv::imshow("background mask", m_mouseData->m_bgmask);
			
		}
		else if (m_mouseData->rButtonDown)//Foreground
		{
			cv::Point tmp(x, y);
			cv::Mat   tmpMat = m_mouseData->m_scribble.clone();
			cv::Mat   tmpMask = m_mouseData->m_bgmask.clone();

			cv::rectangle(tmpMat, m_mouseData->m_start, tmp, cv::Scalar(0, 255, 0), 2);
			cv::rectangle(tmpMask, m_mouseData->m_start, tmp, 255, CV_FILLED);

			cv::imshow("Scribble Image", tmpMat);
			cv::imshow("background mask", tmpMask);
		}

	}



}



//some global var  for  mouse  action


/*********************************************************************************/

extern int mouse_quit(void* userData);
RegisterMouseAction(q,mouse_quit);

#define MOUSE_SAVE mouse_quit
RegisterMouseAction(s, MOUSE_SAVE);


extern int mouse_restart(void* userData);
RegisterMouseAction(r, mouse_restart);

extern int mouse_brush_radius_dec(void* userData);
RegisterMouseAction(-, mouse_brush_radius_dec);

extern int mouse_brush_radius_inc(void* userData);
RegisterMouseAction(+, mouse_brush_radius_inc);

extern int mouse_new_source(void* userData);
RegisterMouseAction(n, mouse_new_source);

extern int error_handler(void* userData);
RegisterMouseAction(liygcheng, error_handler);



int mouse_quit(void* userData){

	MouseData* m_data = static_cast<MouseData*>(userData);
	if (m_data == NULL){

		LOG(ERROR) << "mouse  data  is empty..";
		return 0;
	}
	//save background ,foreground ,and scribble image

	std::string folder(m_data->m_outfolder), path,name, ext;

	TK::tk_truncate_name(m_data->m_imageName,path, name, ext);

	folder.append("/").append(name);

	TK::tk_make_file(folder.c_str());


	bool flag = cv::imwrite(folder + "/background_mask.png", m_data->m_bgmask);
		 flag &= cv::imwrite(folder + "/foreground_mask.png", m_data->m_fgmask);
		 flag &= cv::imwrite(folder + "/scribble.png", m_data->m_scribble);
		 flag &= cv::imwrite(folder + "/source.png", m_data->m_source);
		 if (!flag){
			 LOG(INFO) << "some errors occur when come to archive.";
		 }
	
	return -1; 
}
int mouse_restart(void* userData){

	MouseData* m_data = static_cast<MouseData*>(userData);
	if (m_data == NULL){

		LOG(ERROR) << "mouse  data  is empty..";
		return 0;
	}

	m_data->m_scribble = m_data->m_source.clone();
	m_data->m_fgmask = 0;
	m_data->m_bgmask = 0;
	m_data->lButtonDown = false;
	m_data->rButtonDown = false;
	m_data->scribbleRadius = 5;
	return 1;
}
int mouse_brush_radius_dec(void* userData){

	MouseData* m_data = static_cast<MouseData*>(userData);
	if (m_data == NULL){

		LOG(ERROR) << "mouse  data  is empty..";
		return 0;
	}
	m_data->scribbleRadius = std::max(m_data->scribbleRadius-1,1);
	return 1;
}
int mouse_brush_radius_inc(void* userData){
	MouseData* m_data = static_cast<MouseData*>(userData);
	if (m_data == NULL){

		LOG(ERROR) << "mouse  data  is empty..";
		return 0;
	}
	m_data->scribbleRadius = std::min(m_data->scribbleRadius + 1, 20);
	return 1;
}
int mouse_new_source(void* userData){

	//
	LOG(INFO) << "old result dump to folder..";
	mouse_quit(userData);

	return 2;
}




int error_handler(void* userData){
	MouseData* m_data = static_cast<MouseData*>(userData);
	if (m_data == NULL){

		LOG(ERROR) << "mouse  data  is empty..";
		return 0;
	}
	return 1;
}




#endif

