#ifndef _CLC_MOUSE_H_
#define _CLC_MOUSE_H_

#include"Utility.h"

#define  ERROR_HANDLER_NAME  ("liygcheng")

//
typedef int(*KeyboardAction)(void);
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
		for (RegisterFunMap::iterator it = mouseActionMap.begin(); it != mouseActionMap.end(); ++it)
		{
			if (ERROR_HANDLER_NAME != it->first)
			LOG(INFO) << "\t" << it->first;
		}
		LOG(INFO) << "unknown actions :" << name;
		return mouseActionMap[ERROR_HANDLER_NAME];

	}

}
//

//


//







/*********************************************************************************/

extern int mouse_quit(void);
RegisterMouseAction(q,mouse_quit);

extern int error_handler(void);
RegisterMouseAction(liygcheng, error_handler);



int mouse_quit(void){

	std::cout << "quit function" << std::endl;
	exit(-1);
	return 0;
}


int error_handler(void){
	//nothing to do...
	return 0;
}




#endif

