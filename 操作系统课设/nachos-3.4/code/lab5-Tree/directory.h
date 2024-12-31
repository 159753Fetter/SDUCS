// directory.h 
//	Data structures to manage a UNIX-like directory of file names.
// 
//      A directory is a table of pairs: <file name, sector #>,
//	giving the name of each file in the directory, and 
//	where to find its file header (the data structure describing
//	where to find the file's data blocks) on disk.
//
//      We assume mutual exclusion is provided by the caller.
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"

#ifndef DIRECTORY_H
#define DIRECTORY_H

#include <string>
#include <vector>
#include <iostream>
#include "openfile.h"
using std::string;
using std::vector;

#define FileNameMaxLen 		19	// for simplicity, we assume 
					// file names are <= 9 characters long

// The following class defines a "directory entry", representing a file
// in the directory.  Each entry gives the name of the file, and where
// the file's header is to be found on disk.
//
// Internal data structures kept public so that Directory operations can
// access them directly.

class DirectoryEntry {
  public:
    bool inUse;				// Is this directory entry in use?
    int sector;				// Location on disk to find the 
					//   FileHeader for this file 
    char name[FileNameMaxLen + 1];	// Text name for file, with +1 for 
					// the trailing '\0'

    bool isDir;     // 判断是否是目录，0为文件，1为目录
    int ChildSize;  // 该目录下文件个数（目录也算）
    int Parent;     // 该目录的父节点
    int Child;      // 该目录下子目录
    int Slibing;    // 与该目录处于同一目录下的文件
};

// The following class defines a UNIX-like "directory".  Each entry in
// the directory describes a file, and where to find it on disk.
//
// The directory data structure can be stored in memory, or on disk.
// When it is on disk, it is stored as a regular Nachos file.
//
// The constructor initializes a directory structure in memory; the
// FetchFrom/WriteBack operations shuffle the directory information
// from/to disk. 

class Directory {
  public:
    Directory(int size); 		// Initialize an empty directory
					// with space for "size" files
    ~Directory();			// De-allocate the directory

    void FetchFrom(OpenFile *file);  	// Init directory contents from disk
    void WriteBack(OpenFile *file);	// Write modifications to 
					// directory contents back to disk

    int Find(char *name);		// Find the sector number of the 
					// FileHeader for file: "name"

    bool Add(char *name, int newSector);  // Add a file name into the directory

    bool Remove(char *name);		// Remove a file from the directory

    void List();			// Print the names of all the files
					//  in the directory
    void Print();			// Verbose print of the contents
					//  of the directory -- all the file
					//  names and their contents.
    void Mark(); // 该函数的作用是将所有目录项中是目录的标记,依据是是否有孩子

    void deletechild(int index); // 递归删除孩子

    vector<string> Parse(string Path); // 根据文件名分割路径

    vector<int> GetChild(char *name); // 获取给定文件孩子

    string GetName(int Index); // 获取给定索引名字

    int GetIndex(int Index); // 获取给定索引父节点

    void VisualTree(); // 将目录树可视化
  private:
    int tableSize;			// Number of directory entries
    DirectoryEntry *table;		// Table of pairs: 
					// <file name, file header location> 

    int FindIndex(char *name);		// Find the index into the directory 
					//  table corresponding to "name"
};

#endif // DIRECTORY_H