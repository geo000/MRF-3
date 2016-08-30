#ifndef _CLC_MOUSE_H_
#define _CLC_MOUSE_H_

#include"Utility.h"

typedef int(*UserAction)(void);
typedef UserAction MouseAction;

typedef std::map<std::string, UserAction> key_user_action;
typedef std::map<int, MouseAction> mouse_user_action;

 key_user_action userAction;
 mouse_user_action mouseAction;

#define RegisterKeyAction(key,fun)					\
	namespace{										\
		class _register_user_key_##fun{				\
		public:	_register_user_key_##fun(){			\
		mouseAction[key] = &(fun);					\
							}						\
					};								\
	_register_user_key_##fun key_action_##fun; \
}													\

extern int quit(void);
RegisterKeyAction(0, quit);



int quit(void){

	std::cout << "" << std::endl;
	exit(-1);
	return 0;
}

#endif

