#include "syscall.h"

int
main()
{
    SpaceId pid = Exec("../test/halt.noff");
    Join(pid);
    Exit(0);
    /* not reached */
}
