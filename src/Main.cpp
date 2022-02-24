#include "Engine.hpp"

int main(int argc, char* argv[])
{
	PeteEngine e;
	e.init();
	e.run();
	e.cleanup();
}
