// directory.cc 
//	Routines to manage a directory of file names.
//
//	The directory is a table of fixed length entries; each
//	entry represents a single file, and contains the file name,
//	and the location of the file header on disk.  The fixed size
//	of each directory entry means that we have the restriction
//	of a fixed maximum size for file names.
//
//	The constructor initializes an empty directory of a certain size;
//	we use ReadFrom/WriteBack to fetch the contents of the directory
//	from disk, and to write back any modifications back to disk.
//
//	Also, this implementation has the restriction that the size
//	of the directory cannot expand.  In other words, once all the
//	entries in the directory are used, no more files can be created.
//	Fixing this is one of the parts to the assignment.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "utility.h"
#include "filehdr.h"
#include "directory.h"
#include <iostream>
#include <fstream>

using namespace std;
//----------------------------------------------------------------------
// Directory::Directory
// 	Initialize a directory; initially, the directory is completely
//	empty.  If the disk is being formatted, an empty directory
//	is all we need, but otherwise, we need to call FetchFrom in order
//	to initialize it from disk.
//
//	"size" is the number of entries in the directory
//----------------------------------------------------------------------

Directory::Directory(int size)
{   
    table = new DirectoryEntry[size];
    tableSize = size;
    for (int i = 0; i < tableSize; i++){
        table[i].inUse = 0;
        table[i].Parent = 0;
        table[i].ChildSize = 0;
        table[i].Child = table[i].Slibing = -1;// -1表示没有孩子和兄弟
    }
    table[0].sector = 1;
    table[0].Parent = table[0].Slibing = -1; // 默认table[0]为根节点
    table[0].inUse = 1; 
    table[0].name[0]='/';
}

//----------------------------------------------------------------------
// Directory::~Directory
// 	De-allocate directory data structure.
//----------------------------------------------------------------------

Directory::~Directory()
{ 
    delete [] table;
} 

//----------------------------------------------------------------------
// Directory::FetchFrom
// 	Read the contents of the directory from disk.
//
//	"file" -- file containing the directory contents
//----------------------------------------------------------------------

void
Directory::FetchFrom(OpenFile *file)
{
    (void) file->ReadAt((char *)table, tableSize * sizeof(DirectoryEntry), 0);
}

//----------------------------------------------------------------------
// Directory::WriteBack
// 	Write any modifications to the directory back to disk
//
//	"file" -- file to contain the new directory contents
//----------------------------------------------------------------------

void
Directory::WriteBack(OpenFile *file)
{
    (void) file->WriteAt((char *)table, tableSize * sizeof(DirectoryEntry), 0);
}

vector<string>
Directory::Parse(string Path){
    vector<string> ans;
    int locate = 0;
    for (int i = 0; i < Path.length(); i++) {
        if (Path[i] == '/') {
            ans.push_back(Path.substr(locate, i - locate)); // 碰到另一个/就说明两个/之间是一个目录
            locate = i + 1;
        }
    }
    ans.push_back(Path.substr(locate)); // 最后压入的就是文件名
    return ans;
}

//----------------------------------------------------------------------
// Directory::FindIndex
// 	Look up file name in directory, and return its location in the table of
//	directory entries.  Return -1 if the name isn't in the directory.
//
//	"name" -- the file name to look up
//----------------------------------------------------------------------

int
Directory::FindIndex(char *name)
{
    // 接下来根据文件名分割路径    
    vector<string> Path = Parse(name);

    // 判断给出的路径是否合法
    if(Path.size() == 0){
        return tableSize + 1;
    }

    int now = 0, findIndex;
    for(int i = 0; i < Path.size(); i++){
        int childNo = table[now].Child;
        findIndex = -1;
        if(childNo == -1){
            // 说明它没有孩子
            return -1;
        }
        else{
            now = childNo;
            if(table[now].name == Path[i]){ // 判断是否是它的第一个孩子
                findIndex = now;
                continue;
            }
            else{ // 如果不是，找它的兄弟结点
                while(table[now].Slibing !=-1){
                    now = table[now].Slibing;
                    if(table[now].name == Path[i]){
                        findIndex = now;
                        break;
                    }
                }
            }
        }
        if(findIndex == -1){
            return -1;
        }
    }
    // 说明找到了
    if(findIndex != -1){
        return now;
    }
}

//----------------------------------------------------------------------
// Directory::Find
// 	Look up file name in directory, and return the disk sector number
//	where the file's header is stored. Return -1 if the name isn't 
//	in the directory.
//
//	"name" -- the file name to look up
//----------------------------------------------------------------------

int
Directory::Find(char *name)
{
    int i = FindIndex(name);

    if (i != -1)
	return table[i].sector;
    return -1;
}

//----------------------------------------------------------------------
// Directory::Add
// 	Add a file into the directory.  Return TRUE if successful;
//	return FALSE if the file name is already in the directory, or if
//	the directory is completely full, and has no more space for
//	additional file names.
//
//	"name" -- the name of the file being added
//	"newSector" -- the disk sector containing the added file's header
//----------------------------------------------------------------------

bool
Directory::Add(char *name, int newSector)
{
    // 说明文件存在
    int vis = FindIndex(name);
    if (vis != -1){
        cout << "The file is exist" << endl;
    	return FALSE;
    }

    // 说明给出的路径不对（指要添加的文件开头不是/）
    if(vis == tableSize + 1){
        printf("The file path is incorrect\n");
        return FALSE;
    }

    string copy_name = name;
    int pos = -1, ParentIndex = 0;

    for(int i = 0; i < copy_name.length(); i++){
        if(copy_name[i] == '/')    
            pos = i;
    }

    string ParentPath = "";
    string ChildPath = "";
    for(int i = 0; i < pos; i++)
        ParentPath += copy_name[i];
    for(int i = pos + 1; i < copy_name.length(); i++)
        ChildPath += copy_name[i];
    if(pos == -1)
        ParentIndex = 0;
    else{
        char copyParentPath[100];
        strcpy(copyParentPath, ParentPath.c_str());
        ParentIndex = FindIndex(copyParentPath);
    }
    printf("ParentIndex:%d\n",ParentIndex);
    for (int i = 1; i < tableSize; i++){
        // find a empty DirectoryEntry
        if(!table[i].inUse){
            table[i].inUse = 1;
            table[i].sector = newSector;
            table[i].Parent = ParentIndex;
            table[i].ChildSize = 0;
            table[i].isDir = 0;
            table[i].Slibing = table[i].Child = -1;
            for(int j = 0; j < copy_name.size(); j++)
                table[i].name[j] = ChildPath[j];

            table[ParentIndex].ChildSize++;
            int temp = table[ParentIndex].Child;
            // 说明一个孩子也没有
            if(temp == -1)
                table[ParentIndex].Child = i;
            else{
                while(table[temp].Slibing != -1){
                    temp = table[temp].Slibing;
                }
                table[temp].Slibing = i;
            }
            Mark();
            return TRUE;
        }
    }
    return FALSE;	// no space.  Fix when we have extensible files.
}

//----------------------------------------------------------------------
// Directory::Remove
// 	Remove a file name from the directory.  Return TRUE if successful;
//	return FALSE if the file isn't in the directory. 
//
//	"name" -- the file name to be removed
//----------------------------------------------------------------------

bool
Directory::Remove(char *name)
{ 
    int i = FindIndex(name);

    if (i == -1)
	return FALSE; 		// name not in directory

    // 说明要删除的目录存在
    table[i].inUse = FALSE;
    int parent = table[i].Parent; // 先处理它的父目录,将它从它父目录的孩子中删除，由于我们用兄弟指针来存孩子
    // 所以要处理兄弟指针
    if(parent != -1){
        int tep = table[parent].Child;
        table[parent].ChildSize--;
        if(tep == i){ // 说明它是孩子指针的第一个
            if(table[i].Slibing != -1){ // 说明它有兄弟，让父节点的孩子指针指向自己的兄弟
                table[parent].Child = table[i].Slibing;
            }
            else{ // 且没有兄弟,删除以后就没有孩子
                table[parent].Child = -1;
            }
        }
        else{ // 说明它不是孩子指针的第一个
            while(table[tep].Slibing != i){ // 循环结束后tep的Slibing指向的是i
                tep = table[tep].Slibing;
            }
            table[tep].Slibing = table[i].Slibing;
        }
    }

    // 接下来处理它的孩子,先将它所有孩子的序号记下来
    vector<int> vec;
    int temp = table[i].Child;
    for(int j = 0; j < table[i].ChildSize; j++){
        vec.push_back(temp);
        temp = table[temp].Slibing;
    }
    // 然后依次删除它的孩子
    for(int j = 0; j < vec.size(); j++){
        int index = vec[j];
        if(table[index].inUse){
            table[index].inUse = 0;
            deletechild(index);
            table[index].ChildSize = 0;
            table[index].Child = -1;
        }
    }
    return TRUE;	
}

void 
Directory::deletechild(int index){
    vector<int> vec;
    int temp = table[index].Child;
     for(int j = 0; j < table[index].ChildSize; j++){
        vec.push_back(temp);
        temp = table[temp].Slibing;
    }
    // 然后依次删除它的孩子
    for(int j = 0; j < vec.size(); j++){
        int i = vec[j];
        if(table[i].inUse){
            table[i].inUse = 0;
            table[i].ChildSize = 0;
            deletechild(i);
            table[index].ChildSize = 0;
            table[index].Child = -1;
        }
    }
}

void
Directory::Mark(){
    for(int i = 0; i < tableSize; i++){
        if(table[i].inUse){
            if(table[i].ChildSize != 0)
                table[i].isDir = 1;
        }
    }
}

//----------------------------------------------------------------------
// Directory::List
// 	List all the file names in the directory. 
//----------------------------------------------------------------------

void
Directory::List()
{
    printf("Directory Contents:\n");
    // 根节点一定存在，单独处理
    printf("/root\n");
    // 从第1个开始，因为0存的是根节点
    for(int i = 1; i < tableSize; i++){
        if(table[i].inUse){
            vector<string> Path;
            int cur = i;
            // 如果不为-1说明有父节点，将父节点名称先加入数组中
            while(table[cur].Parent != -1){
                string tep = table[cur].name;
                Path.push_back(tep);
                cur = table[cur].Parent;
            }
            printf("/root");
            for(int k = Path.size() - 1; k >= 0; k--){
                cout <<"/"<< Path[k];
            }
            printf("\n");
        }
    }
    VisualTree();
}

//----------------------------------------------------------------------
// Directory::Print
// 	List all the file names in the directory, their FileHeader locations,
//	and the contents of each file.  For debugging.
//----------------------------------------------------------------------

void
Directory::Print()
{ 
    FileHeader *hdr = new FileHeader;

    printf("Directory contents:\n");
    for (int i = 0; i < tableSize; i++){
	    if (table[i].inUse) {
            vector<string> Path;
            int index = i;

            while(table[index].Parent != -1){
                string tep = table[index].name;
                Path.push_back(tep);
                index = table[index].Parent;
            }

            cout  << "Name:/root";
            for(int k = Path.size() - 1; k >= 0; k--)
                cout<<"/" << Path[k];
            cout << endl;
	        printf("Sector: %d\n", table[i].sector);
	        hdr->FetchFrom(table[i].sector);
	        hdr->Print();
	    }
    }
    printf("\n");
    delete hdr;
}

void 
Directory::VisualTree(){
    ofstream out;
    out.open("VisualCataLogTree.dot");
    out << "strict digraph s{\n";
    out << "root[color=green];\n";
    for(int i = 1; i < tableSize; i++){
        if(table[i].inUse){
            int ParentIndex = table[i].Parent;
            if(ParentIndex != 0)
                out << table[ParentIndex].name << " -> " << table[i].name << ";\n";
            else
                out << "root" << " -> " << table[i].name << ";\n";
        }
    }
    out<<"}\n";
    out.close();
    system("dot -Tpng VisualCataLogTree.dot -o VisualCataLogTree.png");
    return ;
}

vector<int>
Directory::GetChild(char *name){
    vector<int> vec;
    int i = FindIndex(name);

    if (i == -1)
	    return vec; 		// name not in directory
    else{
        int ChildIndex = table[i].Child;
        while(ChildIndex != -1){
            vec.push_back(ChildIndex);
            ChildIndex = table[ChildIndex].Slibing;
        }
        return vec;
    }
}

string
Directory::GetName(int Index){
    return table[Index].name;
}

int 
Directory::GetIndex(int Index){
    return table[Index].Parent;
}