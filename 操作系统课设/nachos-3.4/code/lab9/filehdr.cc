// filehdr.cc 
//	Routines for managing the disk file header (in UNIX, this
//	would be called the i-node).
//
//	The file header is used to locate where on disk the 
//	file's data is stored.  We implement this as a fixed size
//	table of pointers -- each entry in the table points to the 
//	disk sector containing that portion of the file data
//	(in other words, there are no indirect or doubly indirect 
//	blocks). The table size is chosen so that the file header
//	will be just big enough to fit in one disk sector, 
//
//      Unlike in a real system, we do not keep track of file permissions, 
//	ownership, last modification date, etc., in the file header. 
//
//	A file header can be initialized in two ways:
//	   for a new file, by modifying the in-memory data structure
//	     to point to the newly allocated data blocks
//	   for a file already on disk, by reading the file header from disk
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"

#include "system.h"
#include "filehdr.h"

//----------------------------------------------------------------------
// FileHeader::Allocate
// 	Initialize a fresh file header for a newly created file.
//	Allocate data blocks for the file out of the map of free disk blocks.
//	Return FALSE if there are not enough free blocks to accomodate
//	the new file.
//
//	"freeMap" is the bit map of free disk sectors
//	"fileSize" is the bit map of free disk sectors
//----------------------------------------------------------------------

bool
FileHeader::Allocate(BitMap *freeMap, int fileSize)
{ 
    numBytes = fileSize;
    numSectors  = divRoundUp(fileSize, SectorSize);
    if (freeMap->NumClear() < numSectors)
	return FALSE;		// not enough space

    for (int i = 0; i < numSectors; i++)
	dataSectors[i] = freeMap->Find();
    return TRUE;
}

//重载Allocate函数，在原函数的基础上运行重新分配扇区块
bool
FileHeader::Allocate(BitMap *freeMap, int fileSize, int incrementBytes){
    if(numSectors > 30)     //限定每个文件最多可分配30个扇区（(128 - 8)/4 =30）
    return false;           //超出限定文件的大小的话就无法继续分配
    if((fileSize == 0) && (incrementBytes > 0)){ //fileSize为0说明这是一个空文件
        if(freeMap->NumClear() < 1) // 查找位示图，计算看有无空闲扇区块
            return false;
        // 先为要添加的数据分配一个空闲磁盘块，并更新文件头信息
        dataSectors[0] = freeMap->Find();
        numSectors = 1;
        numBytes = 0;
    }
    numBytes = fileSize;
    int offset = numSectors * SectorSize - numBytes; // numBytes * SectorSize是原文件总大小
    int newSectorBytes = incrementBytes - offset; // offset则是剩余数据量
    // 最后一个扇区的空闲空间足够
    if(newSectorBytes <= 0){
        numBytes = numBytes + incrementBytes;
        return true; 
    }
    // 最后一个扇区的空闲空间不够,需要重新分配磁盘块
    int moreSectors = divRoundUp(newSectorBytes, SectorSize); //需要新加的扇区块数
    if(numSectors + moreSectors > 30)
        return false;   //文件过大，超出30个磁盘块
    if(freeMap->NumClear() < moreSectors)
        return false;
    // 没有超过文件大小的限制，并且磁盘有足够的空闲块
    for(int i = numSectors; i < numSectors + moreSectors; i++)
        dataSectors[i] = freeMap->Find();
    numBytes = numBytes + incrementBytes;   // 更新文件大小
    numSectors = numSectors + moreSectors;  // 更新文件扇区块数
    return true;
}

//----------------------------------------------------------------------
// FileHeader::Deallocate
// 	De-allocate all the space allocated for data blocks for this file.
//
//	"freeMap" is the bit map of free disk sectors
//----------------------------------------------------------------------

void 
FileHeader::Deallocate(BitMap *freeMap)
{
    for (int i = 0; i < numSectors; i++) {
	ASSERT(freeMap->Test((int) dataSectors[i]));  // ought to be marked!
	freeMap->Clear((int) dataSectors[i]);
    }
}

//----------------------------------------------------------------------
// FileHeader::FetchFrom
// 	Fetch contents of file header from disk. 
//
//	"sector" is the disk sector containing the file header
//----------------------------------------------------------------------

void
FileHeader::FetchFrom(int sector)
{
    synchDisk->ReadSector(sector, (char *)this);
}

//----------------------------------------------------------------------
// FileHeader::WriteBack
// 	Write the modified contents of the file header back to disk. 
//
//	"sector" is the disk sector to contain the file header
//----------------------------------------------------------------------

void
FileHeader::WriteBack(int sector)
{
    synchDisk->WriteSector(sector, (char *)this); 
}

//----------------------------------------------------------------------
// FileHeader::ByteToSector
// 	Return which disk sector is storing a particular byte within the file.
//      This is essentially a translation from a virtual address (the
//	offset in the file) to a physical address (the sector where the
//	data at the offset is stored).
//
//	"offset" is the location within the file of the byte in question
//----------------------------------------------------------------------

int
FileHeader::ByteToSector(int offset)
{
    return(dataSectors[offset / SectorSize]);
}

//----------------------------------------------------------------------
// FileHeader::FileLength
// 	Return the number of bytes in the file.
//----------------------------------------------------------------------

int
FileHeader::FileLength()
{
    return numBytes;
}

//----------------------------------------------------------------------
// FileHeader::FileHeader
// 	对文件头的无用内容进行清除
//----------------------------------------------------------------------

FileHeader::FileHeader(){
    numBytes = 0;
    numSectors = 0;
    for(int i = 0; i < NumDirect; i++){
        dataSectors[i] = 0;
    }
}

//----------------------------------------------------------------------
// FileHeader::Print
// 	Print the contents of the file header, and the contents of all
//	the data blocks pointed to by the file header.
//----------------------------------------------------------------------

void
FileHeader::Print()
{
    int i, j, k;
    char *data = new char[SectorSize];

    printf("FileHeader contents.  File size: %d.  File blocks:\n", numBytes);
    for (i = 0; i < numSectors; i++)
	printf("%d ", dataSectors[i]);
    printf("\nFile contents:\n");
    for (i = k = 0; i < numSectors; i++) {
	synchDisk->ReadSector(dataSectors[i], data);
        for (j = 0; (j < SectorSize) && (k < numBytes); j++, k++) {
	    if ('\040' <= data[j] && data[j] <= '\176')   // isprint(data[j])
		printf("%c", data[j]);
            else
		printf("\\%x", (unsigned char)data[j]);
	}
        printf("\n"); 
    }
    delete [] data;
}
