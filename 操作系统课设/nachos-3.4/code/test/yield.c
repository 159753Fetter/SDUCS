#include "syscall.h"

int
main()
{
    SpaceId pid = Exec("../test/halt.noff");
    Yield();
    Exit(0);
    /* not reached */
}
