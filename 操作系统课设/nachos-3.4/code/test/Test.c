#include "syscall.h"

int 
main(){
    OpenFileId fd;
    char buffer[50];
    int size;

    Create("Temp");

    fd = Open("Temp");

    Write("hello world\n", 11, fd);

    size = Read(buffer, 11 , fd);
    Close(fd);
    Exit(0);
}