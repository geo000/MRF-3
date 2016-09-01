#ifndef _CLC_MOUSE_H_
#define _CLC_MOUSE_H_

#include"Utility.h"

#define  ERROR_HANDLER_NAME  ("liygcheng")

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
		  scribbleRadius = 5;

	  }
	  ~MouseData(){
		 if (!m_source.empty()) m_source.release();
		 if (!m_scribble.empty()) m_scribble.release();
		 if (!m_fgmask.empty()) m_fgmask.release();
		 if (!m_bgmask.empty()) m_bgmask.release();
	  }

}MouseData;


static void onMouse(int event, int x, int y, int flag, void* userData){

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


//some global var  for  mouse  action


/*********************************************************************************/

extern int mouse_quit(void* userData);
RegisterMouseAction(q,mouse_quit);

extern int mouse_restart(void* userData);
RegisterMouseAction(r, mouse_restart);

extern int mouse_brush_radius_dec(void* userData);
RegisterMouseAction(-, mouse_brush_radius_dec);

extern int mouse_brush_radius_inc(void* userData);
RegisterMouseAction(+, mouse_brush_radius_inc);

extern int error_handler(void* userData);
RegisterMouseAction(liygcheng, error_handler);



int mouse_quit(void* userData){

	MouseData* m_data = static_cast<MouseData*>(userData);
	if (m_data == NULL){

		LOG(ERROR) << "mouse  data  is empty..";
		return 0;
	}
	//save background ,foreground ,and scribble image
	bool flag  = cv::imwrite(m_data->m_outfolder + "/background_mask.png", m_data->m_bgmask);
		 flag &= cv::imwrite(m_data->m_outfolder + "/foreground_mask.png", m_data->m_fgmask);
		 flag &= cv::imwrite(m_data->m_outfolder + "/scribble.png", m_data->m_scribble);
		 flag &= cv::imwrite(m_data->m_outfolder + "/source.png", m_data->m_source);
		 if (!flag){
			 LOG(INFO) << "some errors occur when come to archive.";
		 }
	exit(-1);
	return 1;
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





int error_handler(void* userData){
	MouseData* m_data = static_cast<MouseData*>(userData);
	if (m_data == NULL){

		LOG(ERROR) << "mouse  data  is empty..";
		return 0;
	}
	return 1;
}




#endif

