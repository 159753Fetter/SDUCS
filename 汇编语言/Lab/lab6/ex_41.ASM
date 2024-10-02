DATA			SEGMENT
PGSIZE			DW				?
BUF_SIZE		DB				80
S_BUF			DB				?
BUF				DB				200 DUP(?)
CUR				DW				?
HANDLE			DW				?
MESS_GETNAME	DB				0DH, 0AH, "    Please input filename: $"
MESS_ERR1		DB				0AH, 0DH, "    Illegal filename! $"
MESS_ERR2		DB				0AH, 0DH, "    File not found! $"
MESS_ERR3		DB				0AH, 0DH, "    File read error! $"
MESS_PSIZE		DB				0AH, 0DH, "    Page size : $"
CRLF			DB				0AH, 0DH, "$"
MESS_STAR		DB				0AH, 0DH, "*********************"
                DB				0AH, 0DH, "$"
DATA			ENDS

CODE			SEGMENT
                ASSUME			DS:DATA, CS:CODE
MAIN			PROC			FAR
START:
                PUSH			DS
                SUB				AX, AX
                PUSH			AX
                MOV				AX, DATA
                MOV				DS, AX
                
                MOV				PGSIZE, 12				;each page 12 lines
                MOV				CUR, 200				;file data buffer is empty
                CALL			GETLINE					;get file name
                CALL			OPENF					;open the file, (ax)=0 means no such file
                OR				AX, AX
                JNZ				DISPLAY					;成功打开文件，跳转到display
                MOV				DX, OFFSET MESS_ERR2
                MOV				AH, 09H
                INT				21H						;(ax)=0:no such file
                
                JMP				FILE_ENDING
DISPLAY:												;BUG原因：文件名也放在了文件内容的缓冲区（BUF）内，需要重新定向文件内容指针
                                                        ;使其从文件真正内容开始读，不将文件名字也视作内容，之后只要读到0，说明文件内容结束
                                                        ;正常退出即可。
                MOV				CX, PGSIZE
SHOW_PAGE:
                CALL			READ_BLOCK				;read a line from HANDLE to buf
                OR				AX, AX					;正常读完了，则跳转到next2
                JNZ				NEXT2
                
                MOV				DX, OFFSET MESS_ERR3
                MOV				AH, 09H
                INT				21H						;(ax)=0;error in read
                JMP				FILE_ENDING
NEXT2:
                CALL			SHOW_BLOCK				;DISPLAY a line in buff,(bx)return 0
                										;means that the file reach its end
                OR				BX, BX
                JZ				FILE_ENDING				;(bx)=0:at the end of file
                CMP				CX, 0					;CX=0，是当前页最后一行了
                JNZ				SHOW_PAGE				;(cx)<>0:not the last line of page
                MOV				DX, OFFSET MESS_STAR
                MOV				AH, 09H
                INT				21H						;at the end of page,print a line of *
                                                        ;the current page has been on screen,and followed by a line of stars.
                                                        ;the following part get the command from keyboard:
WAIT_SPACE:
                MOV				AH, 1
                INT				21H
                CMP				AL, " "
                JNZ				PSIZE
                JMP				DISPLAY					;是空格，重新读取下一页
PSIZE:
                CMP				AL, "p"
                JNZ				WAIT_SPACE
                CALL			CHANGE_PSIZE
HERE:
                MOV				AH, 1
                INT				21H
                CMP				AL, " "
                JNZ				HERE					;stick HERE to wait for space
                JMP				DISPLAY
FILE_ENDING:
                RET
MAIN			ENDP
;**************************************************************************

;**************************************************************************
CHANGE_PSIZE	PROC			NEAR
                PUSH			AX
                PUSH			BX
                PUSH			CX
                PUSH			DX
                MOV				DX, OFFSET MESS_PSIZE
                MOV				AH, 09H
                INT				21H						;print the promt line
                
                MOV				AH, 01
                INT				21H						;get the new num of page size
                CMP				AL, 0DH
                JZ				ILLEG
                SUB				AL, "0"
                MOV				CL, AL					;CL存放着更改后的page size的真值
GETP:
                MOV				AH, 1
                INT				21H
                CMP				AL, 0DH					;如果是回车则结束
                JZ				PGOT					;否则则说明输入的数字不只一位，进行转换成真值
                SUB				AL, "0"
                MOV				DL, AL
                MOV				AL, CL
                MOV				CL, DL					;exchange al and cl
                
                MOV				BL, 10
                MUL				BL
                ADD				CL, AL					;最后的行数真值放入CL
                JMP				GETP
PGOT:
                MOV				DL, 0AH
                MOV				AH, 2
                INT				21H						;output 0ah to complete the return
                
                CMP				CX, 0
                JLE				ILLEG					;不大于0或者大于24，则非法
                CMP				CX, 24
                JG				ILLEG
                MOV				PGSIZE, CX				;be sure the new page size in (0..24)region
ILLEG:
                MOV				DL, 0AH
                MOV				AH, 2
                INT				21H						;output 0ah to complete the return
                POP				DX
                POP				CX
                POP				BX
                POP				AX
                RET
CHANGE_PSIZE	ENDP
;**************************************************************************

;**************************************************************************
OPENF			PROC			NEAR
                PUSH			BX
                PUSH			CX
                PUSH			DX
                MOV				DX, OFFSET BUF
                MOV				AL, 0					;读文件形式，打开文件
                MOV				AH, 3DH
                INT				21H
                MOV				HANDLE, AX				;若成功AX=文件代号
                MOV				AX, 1
                JNC				OK
                MOV				AX, 0
OK:
                POP				DX
                POP				CX
                POP				BX
                RET
OPENF			ENDP
;**************************************************************************

;**************************************************************************
GETLINE			PROC			NEAR					;输入文件名
                PUSH			AX
                PUSH			BX
                PUSH			CX
                PUSH			DX
                MOV				DX, OFFSET MESS_GETNAME
                MOV				AH, 09H
                INT				21H						;promt user to input file name
                
                MOV				DX, OFFSET BUF_SIZE
                MOV				AH, 0AH
                INT				21H						;function call of buffer input
                
                MOV				DX, OFFSET CRLF
                MOV				AH, 09H
                INT				21H						;return
                
                MOV				BL, S_BUF
                MOV				BH, 0
                MOV				[BUF+BX], 0				;put 0 into the end of file name
                
                POP				DX
                POP				CX
                POP				BX
                POP				AX
                RET
GETLINE			ENDP
;**************************************************************************

;**************************************************************************
READ_BLOCK		PROC			NEAR
                PUSH			BX
                PUSH			CX
                PUSH			DX
                CMP				CUR, 200
                JNZ				BACK
                										;首先先修改CUR的上限，如果名字长度超出上限则直接退出
                MOV				SI, 3
                MOV				AH, 0
                MOV				AL, [SI]
                
                MOV				CX, 200					;修改可读字节数到BUF中（因为文件名字也占了一定长度）
                
                SUB				CX, AX
                CMP				CX, 0
                JL				BACK
                
                MOV				BX, HANDLE
                MOV				DX, OFFSET BUF
                ADD				DX, AX
                
                MOV				AH, 3FH
                INT				21H						;从HANDLE（存着文件名）中打开文件，读取CX（200-文件名所占字节）个字节到BUF中
                MOV				CUR, 0
                MOV				AX ,1
                JNC				BACK
                MOV				CUR, 200
                MOV				AX, 0
BACK:
                POP				DX
                POP				CX
                POP				BX
                RET
READ_BLOCK		ENDP
;**************************************************************************

;**************************************************************************
SHOW_BLOCK		PROC			NEAR
                PUSH			AX
                PUSH			DX
                MOV				BX, CUR
                										;修改BX指针位置，使其从文件的真正内容开始读
                MOV				SI, 3
                MOV				AH, 0
                MOV				AL, [SI]
                ADD				BX, AX
LOOP1:
                CMP				BX, 200
                JL				LP
                JMP				EXITING					;if BUFis empty then return
LP:
                MOV				DL, BUF[BX]				;else show the current char
                CMP				DL, 00					;search the file end
                JZ				EXIT_EOF				;是文件尾，则退出
                INC				BX
                INC				CUR
                MOV				AH, 02					;显示DL，即当前读出的字符
                INT				21H
                CMP				DL, 0DH
                JZ				EXIT_LN					;if the char shown is return
                										;then exit.A line has been on screen
                JMP				LOOP1
EXIT_EOF:
                MOV				BX, 0
EXIT_LN:
                DEC				CX						;读到换行符，CX（剩余行数）自减
EXITING:
                POP				DX
                POP				AX
                RET
SHOW_BLOCK		ENDP
;**************************************************************************
CODE			ENDS
                END				START