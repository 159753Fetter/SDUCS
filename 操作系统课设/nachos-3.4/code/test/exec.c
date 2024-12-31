#include "syscall.h"

int
main()
{
    Exec("../test/halt.noff");
    //Yield();
    Exit(0);
    /* not reached */
}
