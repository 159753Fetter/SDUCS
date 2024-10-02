;PROGRAM TITLE GOES HERE -- HANOI
;Solves tower of HANOI puzzle. Printout sequence of moves
; of N discs from initial spindle X to final spindle Z
; Using spindle Y for temporery storage
;****************************************************************************
DATAREA			SEGMENT												;define data segment
BUF_SIZE		DB				80
S_BUF			DB				?
BUF				DB				200 DUP(?)
NAMES			DB				20 DUP(?)
NAMES2			DB				20 DUP(?)
HANDLE			DW				?
HANDLE2			DW				?
OUTING			DW				?
MESS_GETNAME	DB				0DH, 0AH, "    Please input input_filename: $"
MESS_GETNAME2	DB				0DH, 0AH, "    Please input output_filename: $"
MESS_ERR1		DB				0AH, 0DH, "    Illegal filename! $"
MESS_ERR2		DB				0AH, 0DH, "    File not found! $"
MESS_ERR3		DB				0AH, 0DH, "    File read error! $"
MESS			DB				0AH, 0DH, "    Successfully done! $"
FLAG			DW				0
CONSTANT		DW				10000, 1000, 100, 10, 1
DATAREA			ENDS
;****************************************************************************
PROGNAM			SEGMENT												;define code segment
;----------------------------------------------------------------------------
MAIN			PROC			FAR
                ASSUME			CS:PROGNAM, DS:DATAREA
START:
                                                                	;set up stack for return
                PUSH			DS
                SUB				AX, AX
                PUSH			AX
                                                                	;set DS register to current data segment
                MOV				AX, DATAREA
                MOV				DS, AX
                                                					;MAIN PART OF PROGRAM GOES HERE
                CALL			GETING
                CALL			OPENF
                OR				AX, AX
                JNZ				NEXTING								;如果出错，则退出，否则则读数据
                MOV				DX, OFFSET MESS_ERR2
                MOV				AH, 09H
                INT				21H
                JMP				EXITING
                
NEXTING:
                CALL			GET_NUM								;此时将所有文件内容已读到BUF中
                OR				AX, AX
                JNZ				NEXT2
                MOV				DX, OFFSET MESS_ERR3
                MOV				AH, 09H
                INT				21H									;error in read
                JMP				EXITING
                
NEXT2:
                CALL			READING								;读取N的值和X,Y,Z
                CMP				BX, 0
                JZ				EXITING
                PUSH			BX
                PUSH			CX
                CALL			OUTPUT_FILE							;打开输出文件
                OR				AX, AX
                JNZ				CALLING								;如果出错，则退出，否则则写数据
                POP				CX
                POP				BX
                MOV				DX, OFFSET MESS_ERR2
                MOV				AH, 09H
                INT				21H
                JMP				EXITING
                
CALLING:
                POP				CX
                POP				BX
                CALL			HANOI								;CALL HANOI(N,X,Y,Z)
                MOV				DX, OFFSET MESS
                MOV				AH, 09H
                INT				21H
EXITING:
                RET
MAIN			ENDP
;----------------------------------------------------------------------------
READING			PROC			NEAR								;读取N,X,Y,Z
                
                MOV				BX, 0
                MOV				SI, 2
NEWCHAR_OF_N:
                MOV				AL, [SI]
                CMP				AL, ' '
                JE				GETING_X							;如果是空格，说明读完N了
                SUB				AL, 30H								;转换成真实数字，如果不是数字，则退出，数据不合法
                JL				EXIT1
                CMP				AL, 9D
                JG				EXIT1
                CBW													;在AL的字节转换成AX中的字
                INC				SI									;下一个字符的下标
                
                XCHG			AX, BX								;如果是多位数，则转换成多位数形式
                MOV				CX, 10D
                MUL				CX
                XCHG			AX, BX
                
                ADD				BX, AX								;获取总和，总和在BX
                JMP				NEWCHAR_OF_N                        ;获取下一个字符
GETING_X:
                INC				SI									;开始获取X,Y,Z
                MOV				CL, [SI]
                INC				SI
                INC				SI
                
                MOV				DI, SI
                
                MOV				SI, [SI]
                AND				SI, 00FFH
                INC				DI
                INC				DI
                MOV				DI, [DI]
                AND				DI, 00FFH
EXIT1:           
                RET
READING			ENDP
;----------------------------------------------------------------------------
GETING			PROC			NEAR								;获取文件名
                PUSH			AX
                PUSH			BX
                PUSH			CX
                PUSH			DX
                
                MOV				DX, OFFSET MESS_GETNAME
                MOV				AH, 09H
                INT				21H
                
                MOV				DX, OFFSET BUF_SIZE
                MOV				AH, 0AH
                INT				21H
                
                MOV				BL, S_BUF
                MOV				BH, 0
                MOV				NAMES[BX], 0						;insert 0 to form the ascii string
NAME_MOVE:
                DEC				BX
                MOV				AL, BUF[BX]
                MOV				NAMES[BX], AL						;move the line got into name string
                JNZ				NAME_MOVE
                
                POP				DX
                POP				CX
                POP				BX
                POP				AX
                RET
GETING			ENDP
;----------------------------------------------------------------------------
OPENF			PROC			NEAR								;打开文件
                PUSH			BX
                PUSH			CX
                PUSH			DX
                MOV				DX, OFFSET NAMES
                MOV				AL, 0								;打开NAMES对应的文件
                MOV				AH, 3DH
                INT				21H
                MOV				HANDLE, AX
                MOV				AX, 1
                JNC				QUTING
                MOV				AX, 0
QUTING:
                POP				DX
                POP				CX
                POP				BX
                RET
OPENF			ENDP
;----------------------------------------------------------------------------
GET_NUM			PROC			NEAR								;将文件内容读到BUF中
                PUSH			BX
                PUSH			CX
                PUSH			DX
                MOV				CX, 200
                MOV				BX, HANDLE
                MOV				DX, OFFSET BUF
                MOV				AH, 3FH
                INT				21H									;读到BUF数组中
                MOV				AX, 1
BACK:
                POP				DX
                POP				CX
                POP				BX
                RET
GET_NUM			ENDP
;----------------------------------------------------------------------------
HANOI			PROC			NEAR
                                                                    ;Solves tower of HANOI puzzle.
                                                                    ;Argement:(BX) = N, (CX) = X, (SI) = Y, (DI) = Z
                CMP				BX, 1								;if N = 1,execute basis
                JE				BASIS
                CALL			SAVE                                ;save N, X, Y, Z
                DEC				BX
                XCHG			SI, DI
                CALL			HANOI								;call HANOI(N-1,X,Z,Y)
                CALL			RESTOR                              ;restore N,X,Y,Z
                CALL			PRINT                               ;print XNZ
                DEC				BX
                XCHG			CX, SI
                CALL			HANOI								;call HANOI(N-1,Y,X,Z)
                JMP				RETURN
BASIS:
                CALL			PRINT								;PRINT X1Z
RETURN:
                RET
HANOI			ENDP
;----------------------------------------------------------------------------
PRINT			PROC			NEAR
                PUSH			BX
                MOV				BX, HANDLE2
                MOV				DX, CX								;PRINT X
                MOV				OUTING, DX							;outing为DX的值，输出一位
                LEA				DX, OUTING
                PUSH			CX									;暂时储存CX
                MOV				CX, 1
                MOV				AH, 40H
                INT				21H
                POP				CX
                
                POP				BX
                CALL			BINIDEC								;PRINT N
                
                MOV				DX, DI								;PRINT Z
                MOV				OUTING, DX							;outing为DX的值，输出一位
                LEA				DX, OUTING
                PUSH			CX									;暂时储存CX
                PUSH			BX
                MOV				BX, HANDLE2
                MOV				CX, 1
                MOV				AH, 40H
                INT				21H
                POP				BX
                POP				CX
                CALL			CRLF								;skip to next line
                RET
PRINT			ENDP
;----------------------------------------------------------------------------
SAVE			PROC			NEAR
                													;push N,X,Y,Z onto stack
                POP				BP
                PUSH			BX
                PUSH			CX
                PUSH			SI
                PUSH			DI
                PUSH			BP
                RET
SAVE			ENDP
;----------------------------------------------------------------------------
RESTOR			PROC			NEAR
                                									;pop Z,Y,X,N from stack
                POP				BP
                POP				DI
                POP				SI
                POP				CX
                POP				BX
                PUSH			BP
                RET
RESTOR			ENDP
;----------------------------------------------------------------------------
DECIBIN			PROC			NEAR
                													;procedure to convert decimal on keybd to bianry
                                                                	;result is left in BX register
                MOV				BX, 0
                                                                	;get digit from keyboard, convert to binary
NEWCHAR:
                MOV				AH, 1								;keyboard input
                INT				21H									;call DOS
                SUB				AL, 30H								;ASCII to binary
                JL				EXIT2								;jump if < 0
                CMP				AL, 9D								;is it > 9d?
                JG				EXIT2								;yes, not dec digit
                CBW													;BYTE in AL to WORD in AX
                
                                                                    ;(digit is now in AX)
                                                                	;multiply number in BX by 10 decimal
                
                XCHG			AX, BX								;trade digit & number
                MOV				CX, 10D                             ;put 10 dec in CX
                MUL				CX                                	;number times 10
                XCHG			AX, BX                              ;trade number & digit
                
                                                                	;add digit in AX to number in BX
                ADD				BX, AX								;add digit to number
                JMP				NEWCHAR                             ;get next digit
EXIT2:
                RET
DECIBIN			ENDP
;----------------------------------------------------------------------------
BINIDEC			PROC			NEAR
                                                                	;procedure to convert binary number in BX to decimal
                                                                    ; on console screen
                PUSH			BX
                PUSH			CX
                PUSH			SI
                PUSH			DI
                MOV				FLAG, 0
                MOV				CX, 5
                LEA				SI, CONSTANT
DEC_DIV:
                MOV				AX, BX								;number high half
                MOV				DX, 0                				;zero out low half
                DIV				WORD PTR [SI]                		;divide by contant
                MOV				BX, DX                              ;remainder into BX
                MOV				DL, AL                              ;quotient into DL
                
                CMP				FLAG, 0								;have not leading zero
                JNZ				PRINT1
                CMP				DL, 0
                JE				SKIP
                MOV				FLAG, 1
                                                                	;PRINT the contents of DL on screent
PRINT1:
                ADD				DL, 30H								;convert to ASCII
                MOV				DH, 0
                MOV				OUTING, DX
                LEA				DX, OUTING
                PUSH			CX
                PUSH			BX
                MOV				CX, 1
                MOV				BX, HANDLE2
                MOV				AH, 40H                				;display function
                INT				21H
                POP				BX
                POP				CX
SKIP:
                ADD				SI, 2
                LOOP			DEC_DIV
                POP				DI
                POP				SI
                POP				CX
                POP				BX
                RET
BINIDEC			ENDP
;----------------------------------------------------------------------------
CRLF			PROC			NEAR
                PUSH			CX
                PUSH			BX
                PUSH			DX
                
                MOV				BX, HANDLE2
                MOV				CX, 1
                MOV				DH, 00H
                MOV				DL, 0AH
                MOV				OUTING, DX
                LEA				DX, OUTING
                MOV				AH, 40H
                INT				21H
                
                POP				DX
                POP				BX
                POP				CX
                RET
CRLF			ENDP
;----------------------------------------------------------------------------
OUTPUT_FILE		PROC			NEAR
                MOV				DX, OFFSET MESS_GETNAME2
                MOV				AH, 09H
                INT				21H
                
                MOV				DX, OFFSET BUF_SIZE
                MOV				AH, 0AH
                INT				21H
                
                MOV				BL, S_BUF
                MOV				BH, 0
                MOV				NAMES2[BX], 0						;insert 0 to form the ascii string
NAME_MOVING:
                DEC				BX
                MOV				AL, BUF[BX]
                MOV				NAMES2[BX], AL						;move the line got into name string
                JNZ				NAME_MOVING
                
                MOV				DX, OFFSET NAMES2
                MOV				AL, 1								
                MOV				AH, 3DH								;打开NAMES2对应的文件
                INT				21H
                MOV				HANDLE2, AX
                MOV				AX, 1
                JNC				OKING
                MOV				AX, 0
OKING:
                RET
                
                RET
OUTPUT_FILE		ENDP
;----------------------------------------------------------------------------
PROGNAM			ENDS
                END				START