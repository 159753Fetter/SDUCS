// exception.cc 
//	Entry point into the Nachos kernel from user programs.
//	There are two kinds of things that can cause control to
//	transfer back to here from user code:
//
//	syscall -- The user code explicitly requests to call a procedure
//	in the Nachos kernel.  Right now, the only function we support is
//	"Halt".
//
//	exceptions -- The user code does something that the CPU can't handle.
//	For instance, accessing memory that doesn't exist, arithmetic errors,
//	etc.  
//
//	Interrupts (which can also cause control to transfer from user
//	code into the Nachos kernel) are handled elsewhere.
//
// For now, this only handles the Halt() system call.
// Everything else core dumps.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "system.h"
#include "syscall.h"
#include "openfile.h"

extern void ThreadTest(void), Copy(char *unixFile, char *nachosFile);
extern void Append(char *unixFile, char *nachosFile, int half);
extern void NAppend(char *nachosFileFrom, char *nachosFileTo);
extern void Print(char *file), PerformanceTest(void);
extern void StartProcess(char *file), ConsoleTest(char *in, char *out);
extern void MailTest(int networkID);

void StartProcess(int spaceId)
{
    AddrSpace *space = AddrSpaces[spaceId]; // 分配地址空间
    space->InitRegisters(); // 设置初始的寄存器值
    space->RestoreState(); // 加载页表寄存器
    machine->Run();			// jump to the user progam 运行用户程序
    ASSERT(FALSE);			// machine->Run never returns;
					// the address space exits
					// by doing the syscall "exit"
}

//-------------------------------------------------------------------------
// delete all char 'c' in str 
//
//-------------------------------------------------------------------------
char buf[128];
char *EraseStrChar(char *str,char c){ 
    int i = 0;
    while (*str != '\0'){
        if (*str != c){
            buf[i] = *str;
            str++;
            i++;
        } //if
        else
            str++;
    } //while
    buf[i] = '\0';
    return buf;
}

char istr[10] = "";
char* itostr(int num){
    int n;
    char ns[10];
    int k = 0;
    while (true){
        n = num % 10;
        ns[k] = 0x30 + n; 
        k++;
        num = num /10;
        if (num == 0)
            break;
    }
    for (int i = 0; i < k; i++)
        istr[i] = ns[k-i-1] ; 
    //printf("str=%s\n",str);
    return istr;
}

void  AdvancePC() { 
    machine->WriteRegister(PCReg, machine->ReadRegister(PCReg) + 4); 
    machine->WriteRegister(NextPCReg, machine->ReadRegister(NextPCReg) + 4); 
} 


//----------------------------------------------------------------------
// ExceptionHandler
// 	Entry point into the Nachos kernel.  Called when a user program
//	is executing, and either does a syscall, or generates an addressing
//	or arithmetic exception.
//
// 	For system calls, the following is the calling convention:
//
// 	system call code -- r2
//		arg1 -- r4
//		arg2 -- r5
//		arg3 -- r6
//		arg4 -- r7
//
//	The result of the system call, if any, must be put back into r2. 
//
// And don't forget to increment the pc before returning. (Or else you'll
// loop making the same system call forever!
//
//	"which" is the kind of exception.  The list of possible exceptions 
//	are in machine.h.
//----------------------------------------------------------------------
// 在此处完成系统调用，$4-$7（4到7号寄存器）传递函数的前四个参数给子程序，参数多于4个时，其余的利用堆栈进行传递
void
ExceptionHandler(ExceptionType which)
{
    int type = machine->ReadRegister(2);

    if ((which == SyscallException)) {
        switch(type){
            case SC_Halt:{
                printf("CurrentThreadId: %d Name: %s, Execute system call of Halt() \n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                DEBUG('a', "Shutdown, initiated by user program.\n");
   	            interrupt->Halt();
                break;
            }
            case SC_Exec:{
                printf("CurrentThreadId: %d Name: %s, Execute system call of Exec() \n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                char filename[128]; 
                int addr = machine->ReadRegister(4); 
                int ii = 0;
                do{
                    machine->ReadMem(addr + ii,1,(int *)&filename[ii]);
                } while (filename[ii++] != '\0');

                if (filename[0] == 'l' && filename[1] == 's'){ //ls
                    printf("File(s) on Nachos DISK:\n");
                    fileSystem->List();
                    machine->WriteRegister(2,127); //
                    AdvancePC(); 
                    break; 
                }
                
                else if (filename[0] == 'r' && filename[1] == 'm'){ //rm file
                    char fn[128];
                    strncpy(fn,filename,strlen(filename) - 5);
                    fn[strlen(filename) - 5] = '\0';
                    fn[0] = ' ';
                    fn[1] = ' ';
                    char *file = EraseStrChar(fn,' ');
                    if (file != NULL && strlen(file) >0){
                        fileSystem->Remove(file);
                        machine->WriteRegister(2,127);
                    }
                    else{
                        printf("Remove: invalid file name.\n");
                        machine->WriteRegister(2,-1);
                    }
                    AdvancePC(); 
                    break; 
                } //rm
                else if (filename[0] == 'c' && filename[1] == 'a' && filename[2] == 't'){//cat file
                    char fn[128];
                    strncpy(fn,filename,strlen(filename) - 5);
                    fn[strlen(filename) - 5] = '\0';
                    fn[0] = ' ';
                    fn[1] = ' ';
                    fn[2] = ' ';
                    char *file = EraseStrChar(fn,' ');
                    if(file != NULL && strlen(file) >0){
                        Print(file);
                        machine->WriteRegister(2,127);
                    }
                    else{
                        printf("Cat: file not exists.\n");
                        machine->WriteRegister(2,-1);
                    }
                    AdvancePC(); 
                    break; 
                }

                else if (filename[0] == 'c' && filename[1] == 'f') //create a nachos file
                { //create file
                    char fn[128];
                    strncpy(fn,filename,strlen(filename) - 5);
                    fn[strlen(filename) - 5] = '\0';
                    fn[0] = ' ';
                    fn[1] = ' ';
 
                    char *file = EraseStrChar(fn,' ');
                    if (file != NULL && strlen(file) >0){
                        fileSystem->Create(file,0);
                        machine->WriteRegister(2,127);
                    }
                    else{
                        printf("Create: file already exists.\n");
                        machine->WriteRegister(2,-1);
                    }
                    AdvancePC(); 
                    break;
                }

                else if (filename[0] == 'c' && filename[1] == 'p'){//cp source dest
                    char fn[128];
                    strncpy(fn,filename,strlen(filename) - 5);
                    fn[strlen(filename) - 5] = '\0';
                    fn[0] = '#';
                    fn[1] = '#';
                    fn[2] = '#';
                    char source[128];
                    char dest[128];
                    int startPos = 3;
                    int j = 0;
                    int k = 0;
                    for (int i = startPos; i < 128;i++)
                        if (fn[i] != ' ')
                            source[j++] = fn[i];
                        else 
                             break;
                    source[j] = '\0';
                    j++; 
 
                    for (int i = startPos + j; i < 128;i++)
                        if (fn[i] != ' ')
                            dest[k++] = fn[i];
                        else 
                            break; 
                    dest[k] = '\0';
 
                    if (source == NULL || strlen(source) <= 0){
                        printf("cp: Source file not exists.\n");
                        machine->WriteRegister(2,-1);
                        AdvancePC(); 
                        break; 
                    }
                    if (dest != NULL && strlen(dest) > 0){
                        NAppend(source, dest);
                        machine->WriteRegister(2,127);
                    }
                    else{
                        printf("cp: Missing dest file.\n");
                        machine->WriteRegister(2,-1);
                    }
                    AdvancePC();
                    break;
                }

                else if ((filename[0] == 'u' && filename[1] == 'a' && filename[2] == 'p') 
                || (filename[0] == 'u' && filename[1] == 'c' && filename[2] == 'p')){
                    char fn[128];
                    strncpy(fn,filename,strlen(filename) - 5);
                    fn[strlen(filename) - 5] = '\0';
                    fn[0] = '#';
                    fn[1] = '#';
                    fn[2] = '#';
                    fn[3] = '#';
                    char source[128];
                    char dest[128];
                    int startPos = 4;
                    int j = 0;
                    int k = 0;
                    for (int i = startPos; i < 128;i++)
                        if (fn[i] != ' ')
                            source[j++] = fn[i];
                        else 
                            break;
                    source[j] = '\0';
                    j++; 
                    for (int i = startPos + j; i < 128;i++)
                        if (fn[i] != ' ')
                            dest[k++] = fn[i];
                        else 
                             break; 
                    dest[k] = '\0';
 
                    if (source == NULL || strlen(source) <= 0){
                        printf("uap or ucp: Source file not exists.\n");
                        machine->WriteRegister(2,-1);
                        AdvancePC(); 
                        break; 
                    }
                    if (dest != NULL && strlen(dest) > 0){
                        if (filename[0] == 'u' && filename[1] == 'c' && filename[2] == 'p')
                            Append(source, dest,0); //append dest file at the end of source file
                        else
                            Copy(source, dest); //ucp
                        machine->WriteRegister(2,127);
                    }
                    else{
                        printf("uap or ucp: Missing dest file.\n");
                        machine->WriteRegister(2,-1);
                    }
                    AdvancePC(); 
                    break; 
                }

                //nap source dest
                else if (filename[0] == 'n' && filename[1] == 'a' && filename[2] == 'p'){
                    char fn[128];
                    strncpy(fn,filename,strlen(filename) - 5);
                    fn[strlen(filename) - 5] = '\0';
                    fn[0] = '#';
                    fn[1] = '#';
                    fn[2] = '#';
                    fn[3] = '#';
                    char source[128];
                    char dest[128];
                    int j = 0;
                    int k = 0;
                    int startPos = 4;
                    for (int i = startPos; i < 128; i++)
                        if (fn[i] != ' ')
                            source[j++] = fn[i];
                        else 
                            break;
                    source[j] = '\0';
                    j++; 
 
                    for (int i = startPos + j; i < 128; i++)
                        if (fn[i] != ' ')
                            dest[k++] = fn[i];
                        else 
                            break; 
                    dest[k] = '\0';
 
                    if (source == NULL || strlen(source) <= 0){
                        printf("nap: Source file not exists.\n");
                        machine->WriteRegister(2,-1);
                        AdvancePC(); 
                        break; 
                    }
                    if (dest != NULL && strlen(dest) > 0){
                        NAppend(source, dest);
                        machine->WriteRegister(2,127);
                    }
                    else{
                        printf("ap: Missing dest file.\n");
                    machine->WriteRegister(2,-1);
                    }
                    AdvancePC(); 
                    break; 
                }

                else if (strstr(filename,"format") != NULL){//format
                    printf("strstr(filename,\"format\"=%s \n",strstr(filename,"format"));
                    printf("WARNING: Format Nachos DISK will erase all the data on it.\n");
                    printf("Do you want to continue (y/n)? ");
                    char ch;
                    while (true){
                        ch = getchar();
                        if (ch == 'Y' || ch == 'y' || ch == 'n' || ch == 'N')
                            break;
                    } //while
                    if (ch == 'n' || ch == 'N'){
                        machine->WriteRegister(2,127); //
                        AdvancePC(); 
                        break; 
                    }
                    printf("Format the DISK and create a file system on it.\n");
                    fileSystem->FormatDisk(true);
                    machine->WriteRegister(2,127); //
                    AdvancePC(); 
                    break; 
                } 

                else if (strstr(filename,"fdisk") != NULL){ //fdisk 
                    fileSystem->Print();
                    machine->WriteRegister(2,127); //
                    AdvancePC(); 
                    break; 
                } 

                else if (strstr(filename,"perf") != NULL){ //Performance
                    PerformanceTest();
                    machine->WriteRegister(2,127); //
                    AdvancePC(); 
                    break; 
                } 
                else if (strstr(filename,"help") != NULL){ //fdisk
                    printf("Commands and Usage:\n");
                    printf("ls : list files on DISK.\n");
                    printf("fdisk : display DISK information.\n");
                    printf("format : format DISK with creating a file system on it.\n");
                    printf("performence : test DISK performence.\n");
                    printf("cf name : create a file \"name\" on DISK.\n");
                    printf("cp source dest : copy Nachos file \"source\" to \"dest\".\n");
                    printf("nap source dest : Append Nachos file \"source\" to \"dest\"\n");
                    printf("ucp source dest : Copy Unix file \"source\" to Nachos file \"dest\"\n");
                    printf("uap source dest : Append Unix file \"source\" to Nachos file \"dest\"\n");
                    printf("cat name : print content of file \"name\".\n");
                    printf("rm name : remove file \"name\".\n"); 
                    //-----------------------------------------------------------
                    machine->WriteRegister(2,127); //
                    AdvancePC(); 
                    break; 
                }

                else {//check if the parameter of exec(file), i.e file, is valid
                    if (strchr(filename,' ') != NULL || strstr(filename,".noff") == NULL){
                        //not an inner command, and not a valid Nachos executable, then return 
                        machine->WriteRegister(2,-1); 
                        AdvancePC(); 
                        break;
                    }
                }

                OpenFile *executable = fileSystem->Open(filename); 
                if (executable == NULL) {
                    DEBUG('f',"\nUnable to open file %s\n", filename);
                    machine->WriteRegister(2,-1); 
                    AdvancePC(); 
                    break;
                    //return;
                }
 
                //new address space
                AddrSpace* space = new AddrSpace(executable); 
                delete executable; // close file
 
                DEBUG('H',"Execute system call Exec(\"%s\"), it's SpaceId(pid) = %d \n",filename,space->GetSpaceId());
                //new and fork thread
                char *forkedThreadName = filename;
 
                //------------------------------------------------
                char *fname=strrchr(filename,'/');
                if (fname != NULL) // there exists "/" in filename 
                    forkedThreadName=strtok(fname,"/"); 
                //----------------------------------------------- 
                Thread* thread = new Thread(forkedThreadName);
                //printf("exec -- new thread pid =%d\n",space->getSpaceID());
                thread->Fork(StartProcess, space->GetSpaceId());
                thread->space = space;
                printf("user process \"%s(%d)\" map to kernel thread \" %s \"\n",filename,space->GetSpaceId(),forkedThreadName);
 
                machine->WriteRegister(2,space->GetSpaceId());
                currentThread->Yield();
                AdvancePC(); 
                break;
            }
            case SC_Join:{
                printf("CurrentThreadId: %d Name: %s, Execute system call of Join() \n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int SpaceID=machine->ReadRegister(4);//读取子进程的ID
                currentThread->Join(SpaceID);//等待子进程结束
                machine->WriteRegister(2,currentThread->waitProcessExitCode);//返回子进程的返回值
                AdvancePC();
                break;
            }
            case SC_Exit:{
                printf("CurrentThreadId: %d Name: %s, Execute system call of Exit() \n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int ExitStatus = machine->ReadRegister(4); // 读取返回值
                currentThread->setExitCode(ExitStatus); // 设置返回值
                if(ExitStatus == 99){
                    List *terminatedList=scheduler->getTerminatedList();
                    scheduler->emptyList(terminatedList);
                }
                delete currentThread->space; // 删除地址空间
                currentThread->Finish();
                AdvancePC();
                break;
            }
            case SC_Yield:{
                printf("CurrentThreadId: %d Name: %s, Execute system call of Yield() \n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                currentThread->Yield();
                AdvancePC();
                break;
            }
#ifdef FILESYS_STUB
            case SC_Create:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_STUB_SC_Create()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4);
                int value;
                int count = 0;
                char *FileName = new char[128];
                do{
                    machine->ReadMem(base + count, 1 ,&value);
                    FileName[count] = (char)value;
                    count++;
                }while((char)value != '\0' && count < 128);
                int fileDescriptor = OpenForWrite(FileName);
                if(fileDescriptor == -1){
                    printf("create file %s failed!\n",FileName);
                }
                else{
                    printf("create file %s succeed!,the file id is %d\n", FileName, fileDescriptor);
                }
                Close(fileDescriptor);
                AdvancePC();
                break;
            }
            case SC_Open:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_STUB_SC_Open()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4);
                int value;
                int count = 0;
                char *FileName = new char[128];
                do{
                    machine->ReadMem(base + count, 1, &value);
                    FileName[count] = (char)value;
                    count++;
                }while(count < 128 && (char)value != '\0');
                int fileDescriptor = OpenForReadWrite(FileName,FALSE);
                if(fileDescriptor==-1){
                    printf("Open file %s failed!\n",FileName);
                }
                else{
                    printf("Open file %s succeed!, the file id is %d\n",FileName,fileDescriptor); 
                }
                machine->WriteRegister(2,fileDescriptor);
                AdvancePC();
                break;
            }
            case SC_Write:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_STUB_SC_Write()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4);    // buffer
                int size = machine->ReadRegister(5);    // bytes written to file
                int fileId = machine->ReadRegister(6);  // fd
                int value;
                int count = 0;
                OpenFile *openfile = new OpenFile(fileId);
                ASSERT(openfile != NULL);
                char *buffer = new char[128];
                do{
                    machine->ReadMem(base + count, 1, &value);
                    buffer[count] = (char)value;
                    count++;
                }while((char)value != '\0' && count < size);
                buffer[size] = '\0';
                int WritePostion;
                if (fileId == 1){
                    WritePostion = 0;
                }
                else{
                    WritePostion = openfile->Length();
                }
                int writtenBytes = openfile->WriteAt(buffer,size,WritePostion);
                if(writtenBytes == 0){
                    printf("write file failed!\n");
                }
                else{
                    printf("\"%s\" has wrote in file %d succeed!\n",buffer,fileId);
                }
                AdvancePC();
                break;
            }
            case SC_Read:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_STUB_SC_Read()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4);
                int size = machine->ReadRegister(5);
                int fileId = machine->ReadRegister(6);
                OpenFile *openfile = new OpenFile(fileId);
                char buffer[size];
                int readnum = 0;
                readnum = openfile->Read(buffer,size);
                for(int i = 0;i < size; i++){
                    if(!machine->WriteMem(base,1,buffer[i])) 
                        printf("This is something wrong.\n"); 
                }
                buffer[size]='\0';
                printf("read succeed!The content is \"%s\",the length is %d\n",buffer,size);
                machine->WriteRegister(2,readnum);
                AdvancePC();
                break;
            }
            case SC_Close:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_STUB_SC_Close()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int fileId = machine->ReadRegister(4);
                Close(fileId);
                printf("File %d closed succeed!\n", fileId);
                AdvancePC();
                break;
            }
#else
            case SC_Create: {
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_SC_Create()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4);
                int value;
                int count = 0;
                char *FileName = new char[128];
                do{
                    machine->ReadMem(base + count, 1, &value);
                    FileName[count] = (char)value;
                    count++;
                }while(count < 128 && (char)value != '\0');

                //when calling Create(), thread go to sleep, waked up when I/O finish
                if(!fileSystem->Create(FileName,0)) //call Create() in FILESYS,see filesys.h
                    printf("create file %s failed!\n",FileName);
                else
                    DEBUG('f',"create file %s succeed!\n",FileName); 
                AdvancePC();
                break;
            }
            case SC_Open:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_SC_Open()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4);
                int value;
                int count = 0;
                char *FileName = new char[128];
                do{
                    machine->ReadMem(base + count, 1, &value);
                    FileName[count] = (char)value;
                    count++;
                }while(count < 128 && (char)value != '\0');
                int fileid;
                //call Open() in FILESYS,see filesys.h,Nachos Open()
                OpenFile* openfile = fileSystem->Open(FileName); 
                if(openfile == NULL ) { //file not existes, not found
                    printf("File \"%s\" not Exists, could not open it.\n",FileName);
                    fileid = -1;
                }
                else{ //file found
                        //set the opened file id in AddrSpace, which wiil be used in Read() and Write()
                    fileid = currentThread->space->getFileDescriptor(openfile); 
                    if (fileid < 0)
                        printf("Too many files opened!\n");
                    else 
                        DEBUG('f',"file :%s open secceed! the file id is %d\n",FileName,fileid);
                }
                machine->WriteRegister(2,fileid);
                AdvancePC();
                break;
            }
            case SC_Write:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_SC_Write()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4); //buffer
                int size = machine->ReadRegister(5); //bytes written to file 
                int fileId = machine->ReadRegister(6); //fd 
                int value;
                int count = 0;
                // printf("base=%d, size=%d, fileId=%d \n",base,size,fileId );
                OpenFile* openfile = new OpenFile (fileId);
                ASSERT(openfile != NULL);
 
                char* buffer = new char[128];

                do{
                    machine->ReadMem(base + count, 1, &value);
                    buffer[count] = (char)value;
                    count++;
                }while(count < 128 && (char)value != '\0');
                buffer[size]='\0';
 
                openfile = currentThread->space->getFileId(fileId); 
                //printf("openfile =%d\n",openfile);
                if (openfile == NULL)
                {
                    printf("Failed to Open file \"%d\" .\n",fileId); 
                    AdvancePC();
                    break;
                } 
 
                if (fileId ==1 || fileId ==2)
                {
                    openfile->WriteStdout(buffer,size); 
                    delete [] buffer;
                    AdvancePC();
                    break;
                } 
 
                int WritePosition = openfile->Length();
 
                openfile->Seek(WritePosition); //append write
                //openfile->Seek(0); //write form begining
 
                int writtenBytes;
                //writtenBytes = openfile->AppendWriteAt(buffer,size,WritePosition);
                writtenBytes = openfile->Write(buffer,size);
                if((writtenBytes) == 0)
                    DEBUG('f',"\nWrite file failed!\n");
                else
                {
                    if (fileId != 1 && fileId != 2)
                    {
                        DEBUG('f',"\n\"%s\" has wrote in file %d succeed!\n",buffer,fileId);
                        DEBUG('H',"\n\"%s\" has wrote in file %d succeed!\n",buffer,fileId);
                        DEBUG('J',"\n\"%s\" has wrote in file %d succeed!\n",buffer,fileId);
                    }
                //printf("\n\"%s\" has wrote in file %d succeed!\n",buffer,fileId); 
                }
 
                //delete openfile;
                delete [] buffer;
                AdvancePC();
                break;
            }
            case SC_Read:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_SC_Read()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int base = machine->ReadRegister(4);
                int size = machine->ReadRegister(5);
                int fileId = machine->ReadRegister(6);
                OpenFile* openfile = currentThread->space->getFileId(fileId); 
 
                char buffer[size];
                int readnum = 0;
                if (fileId == 0) //stdin
                    readnum = openfile->ReadStdin(buffer,size);
                 else
                    readnum = openfile->Read(buffer,size);
 
                for(int i = 0;i < readnum; i++)
                    machine->WriteMem(base,1,buffer[i]);
                buffer[readnum]='\0';
 
                for(int i = 0;i < readnum; i++)
                    if (buffer[i] >=0 && buffer[i] <= 9)
                        buffer[i] = buffer[i] + 0x30; 
 
                char *buf = buffer;
                if (readnum > 0)
                {
                    if (fileId != 0)
                    {
                        DEBUG('f',"Read file (%d) succeed! the content is \"%s\" , the length is %d\n",fileId,buf,readnum);
                        DEBUG('H',"Read file (%d) succeed! the content is \"%s\" , the length is %d\n",fileId,buf,readnum);
                        DEBUG('J',"Read file (%d) succeed! the content is \"%s\" , the length is %d\n",fileId,buf,readnum);
                    } 
                }
                else
                    printf("\nRead file failed!\n");
                machine->WriteRegister(2,readnum);
                AdvancePC();
                break;
            }
            case SC_Close:{
                printf("CurrentThreadId: %d Name: %s ,Execute system call of FILESYS_SC_Close()\n",(currentThread->space)->GetSpaceId(),currentThread->getName());
                int fileId = machine->ReadRegister(4);
                OpenFile* openfile = currentThread->space->getFileId(fileId);
                if (openfile != NULL)
                {
                    openfile->WriteBack(); // write file header back to DISK
                    delete openfile; // close file 
                    currentThread->space->releaseFileDescriptor(fileId);
 
                    DEBUG('f',"File %d closed succeed!\n",fileId);
                    DEBUG('H',"File %d closed succeed!\n",fileId);
                    DEBUG('J',"File %d closed succeed!\n",fileId);
                }
                else
                    printf("Failded to Close File %d.\n",fileId);
                AdvancePC();
                break;
            }
#endif
            default:{
                printf("Unexpected user mode exception %d %d\n", which, type);
                ASSERT(FALSE);
                break;
            }
        }
    } else {
	    printf("Unexpected user mode exception %d %d\n", which, type);
	    ASSERT(FALSE);
    }
}
