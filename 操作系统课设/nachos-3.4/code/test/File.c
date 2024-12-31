#include "syscall.h"
int main(){
    OpenFileId fp;
    char buffer[50];
    int size;
    Create("Test");

    fp = Open("Test");

    Write("About File_STUB system call\n",27,fp);

    size = Read(buffer, 16, fp);

    Close(fp);
    Exit(0);
}