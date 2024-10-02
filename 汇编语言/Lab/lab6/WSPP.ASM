;title wspp -- program of word process function
; for insert, left and right
;-------------------------------------------------------------------------
DSEG			SEGMENT									;define data segment
 KBD_BUF		DB				96 DUP(' ')				;input buffer
 CNTL			DB				16 DUP(0)				;char number of rows
 BUFPT			DW				0						;buffer head pointer
 BUFTL			DW				0						;buffer tail pointeer
 COLPT			DB				0                       ;current col pointer
 ROWPT			DB				0                		;current row pointer
 ROWMX			DW				0                		;maxium row number
DSEG			ENDS
;-------------------------------------------------------------------------
CURS			MACRO			ROW, COL                ;position cursor macro
                MOV				DH, ROW					;传入参数：ROW和COL
                MOV				DL, COL
                MOV				BH, 0
                MOV				AH, 2					;功能：置光标位置，DH = 行,DL = 列,BH = 页号 
                INT				10H
                ENDM
;-------------------------------------------------------------------------
CSEG			SEGMENT									;define code segment
MAIN			PROC			FAR
                ASSUME			CS:CSEG, DS:DSEG, ES:DSEG
START:
                PUSH			DS                		;save for return to DOS
                SUB				AX, AX
                PUSH			AX
                MOV				AX, DSEG                ;dseg addr into ds, es
                MOV				DS, AX
                MOV				ES, AX
                
                MOV				BUFTL, 0                ;initialize pointers
                MOV				COLPT, 0
                MOV				ROWPT, 0
                MOV				BUFPT, 0
                MOV				ROWMX, 0
                MOV				CX, LENGTH CNTL			;initialize cntl area（CX = 16）
                MOV				AL, 0
                LEA				DI, CNTL
                CLD
                REP				STOSB
                
                MOV				AH, 6					;清空屏幕
                MOV				AL, 0
                MOV				CX, 0
                MOV				DH, 24
                MOV				DL, 79
                MOV				BH, 07
                INT				10H
                CURS			0, 0					;place cursor(0,0)
READ_K:
                MOV				AH, 0					;read char from kbd
                INT				16H						;call ROM routine			ASCII读到AL中
                CMP				AL, 1BH					;is Escape?
                JNZ				ARROW
                RET										;yes, return to DOS
ARROW:
                CMP				AH, 4BH					;is left arrow?				判断是不是左右移动
                JE				LEFT					;yes,moving cursor
                CMP				AH, 4DH					;is right arrow?
                JZ				RIGHT					;yes
;-------------------------------------------------------------------------
INST:													;insert a character
                JMP				INS_K					;正常插入
LEFT:
                JMP				LEFT_K					;是左方向键
RIGHT:
                JMP				RIGHT_K					;是右方向键
;-------------------------------------------------------------------------
INS_K:
                MOV				BX, BUFPT
                MOV				CX, BUFTL
                CMP				BX, CX					;判断buffer尾指针和头指针是否重合
                JE				KM						;重合了，直接插入
                LEA				DI, KBD_BUF				;没重合，则从尾指针前一位到头指针的所有内容向后移动一位
                ADD				DI, CX
                MOV				SI, DI
                DEC				SI
                SUB				CX, BX
                STD
                REP				MOVSB
KM:
                MOV				KBD_BUF[BX], AL			;插入字符
                INC				BUFPT                	;inc head pointer
                INC				BUFTL                	;inc tail pointer
                CMP				AL, 0DH                	;是插入回车吗
                JNZ				KN						;no
                LEA				SI, CNTL				;yes, move the count
                ADD				SI, ROWMX	            ;     of each row
                INC				SI						;		backward
                MOV				DI, SI
                INC				DI
                MOV				CX, ROWMX
                SUB				CL, ROWPT
                STD
                REP				MOVSB
                
                MOV				BL, ROWPT				;adjust the counts
                XOR				BH, BH	                ; of current row
                MOV				CL, COLPT				; and next row
                MOV				CH, CNTL[BX]
                SUB				CH, COLPT
                MOV				CNTL[BX], CL
                MOV				CNTL[BX + 1], CH
                
                MOV				AX, ROWMX				;clear displaying row
                MOV				BH, 07					; use scroll function
                MOV				CH, ROWPT
                MOV				DH, 24
                MOV				CL, 0
                MOV				DL, 79
                MOV				AH, 6
                INT				10H
                
                INC				ROWPT                	;POINT to next row
                INC				ROWMX                	;inc max row count
                MOV				COLPT, 0                ;POINT to 0 column
                JMP				SHORT KP
KN:
                MOV				BL, ROWPT
                XOR				BH, BH
                INC				CNTL[BX]                ;inc current row count
                INC				COLPT                	;POINT to next column
KP:
                CALL			DISPBF                	;display input buffer
                CURS			ROWPT, COLPT            ;position the cursor
                JMP				READ_K
LEFT_K:
                CMP				COLPT, 0                ;is at 0 column
                JNZ				K2                		; no
                CMP				ROWPT, 0                ;is at 0 row
                JZ				LRET                	;yes,cursor is unmove
                DEC				ROWPT                	;POINT to upper row
                MOV				AL, ROWPT
                LEA				BX, CNTL
                MOV				COLPT, AL               ;POINT to tail of row
                JMP				K3
K2:
                DEC				COLPT                	;dec column pointer
K3:                
                DEC				BUFPT                	; dec buffer point
                CURS			ROWPT, COLPT            ;position cursor
LRET:
                JMP				READ_K
RIGHT_K:
                MOV				BX, BUFPT                ;is at tail of file?
                CMP				BX, BUFTL
                JE				RRET                	 ;yes, cursor unmoved
                INC				COLPT                	 ;POINT to next column
                CMP				KBD_BUF[BX], 0DH         ;is CR ?
                JNZ				K4                       ;no
                INC				ROWPT					 ;yes, POINT to next row
                MOV				COLPT, 0                 ;and 0 column
K4:
                INC				BUFPT                    ;adjust buffer pointer
                CURS			ROWPT, COLPT			 ;position cursor
RRET:
                JMP				READ_K
;-------------------------------------------------------------------------
DISPBF			PROC			NEAR					  ;display char of buffer
                MOV				BX, 0
                MOV				CX, 96
                CURS			0, 0
DISP:
                MOV				AL, KBD_BUF[BX]
                PUSH			BX
                MOV				BX, 0700
                MOV				AH, 0EH
                INT				10H							;call ROM routine
                POP				BX
                CMP				AL, 0DH						;is CR?
                JNZ				KK
                MOV				AL, 0AH						;yes, display LF
                MOV				AH, 0EH
                INT				10H							;video call
KK:
                INC				BX
                LOOP			DISP
                RET
DISPBF			ENDP
;-------------------------------------------------------------------------
MAIN			ENDP										;end MAIN part of program
;-------------------------------------------------------------------------
CSEG			ENDS
                END				START