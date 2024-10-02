;Display a message based on an error code in AX
;All registers are preserved
;***********************************************************
DSEG				SEGMENT			PARA'DATA'
CR					EQU				13
LF					EQU				10
EOM					EQU				'$'
NUMIN				LABEL			BYTE
NOUSE				DB				3
ACT					DB				?					;储存输入位数
NUM					DB				3 DUP(?)
;
MESS				DB				'INPUT THE NUM',CR,LF,EOM
OUT_OF_RANGE		DB				'Error code is not in valid range(1-83)',CR,LF,EOM
RESERVED			DB				'Error code is reserved(36-79)',CR,LF,EOM
ER1					DB				'Invalid function number',CR,LF,EOM
ER2					DB				'File not found',CR,LF,EOM
ER3					DB				'Path not found',CR,LF,EOM
ER4					DB				'Too many open files',CR,LF,EOM
ER5					DB				'Access denied',CR,LF,EOM
ER6					DB				'Invalid handle',CR,LF,EOM
ER7					DB				'Memory control blocks destroyed',CR,LF,EOM
ER8					DB				'Insufficient memory',CR,LF,EOM
ER9					DB				'Invalid memory block address',CR,LF,EOM
ER10				DB				'Invalid environment',CR,LF,EOM
ER11				DB				'Invalid format',CR,LF,EOM
ER12				DB				'Invalid access code',CR,LF,EOM
ER13				DB				'Invalid data',CR,LF,EOM
ER14				DB				'No such message',CR,LF,EOM
ER15				DB				'Invalid drive was specified',CR,LF,EOM
ER16				DB				'Attempted to remove th current directory',CR,LF,EOM
ER17				DB				'Not same device',CR,LF,EOM
ER18				DB				'No more files',CR,LF,EOM
ER19				DB				'Disk is write protected',CR,LF,EOM
ER20				DB				'Unknown unit',CR,LF,EOM
ER21				DB				'Drive not ready',CR,LF,EOM
ER22				DB				'Unknown command',CR,LF,EOM
ER23				DB				'Data error(CRC)',CR,LF,EOM
ER24				DB				'Bad request structure length',CR,LF,EOM
ER25				DB				'Seek error',CR,LF,EOM
ER26				DB				'Unknown media type',CR,LF,EOM
ER27				DB				'Sector not found',CR,LF,EOM
ER28				DB				'Printer out of paper',CR,LF,EOM
ER29				DB				'Write fault',CR,LF,EOM
ER30				DB				'Read fault',CR,LF,EOM
ER31				DB				'General failure',CR,LF,EOM
ER32				DB				'Sharing violation',CR,LF,EOM
ER33				DB				'Lock violation',CR,LF,EOM
ER34				DB				'Invalid disk change',CR,LF,EOM
ER35				DB				'FCB unavailable',CR,LF,EOM
ER80				DB				'File exists',CR,LF,EOM
ER81				DB				'Reserved',CR,LF,EOM
ER82				DB				'Cannot make',CR,LF,EOM
ER83				DB				'Fail on INT 24',CR,LF,EOM

ERTAB1				DW				ER1,ER2,ER3,ER4,ER5,ER6,ER7,ER8,ER9,ER10
					DW				ER11,ER12,ER13,ER14,ER15,ER16,ER17,ER18
					DW				ER19,ER20,ER21,ER22,ER23,ER24,ER25,ER26
					DW				ER27,ER28,ER29,ER30,ER31,ER32,ER33,ER34
					DW				ER35
ERTAB2				DW				ER80,ER81,ER82,ER83

DSEG				ENDS
;********************************************************************************
CSEG				SEGMENT 		PARA 					'CODE'
					ASSUME			CS:CSEG,DS:DSEG
SHOW_ERR 			PROC 			FAR
					PUSH			DS
					SUB				BX,BX
					PUSH			BX
;
					MOV				BX,DSEG				;initialize DS
					MOV				DS,BX
;
					LEA				DX,MESS
					MOV				AH,09
					INT				21H
					
;					INPUT
					LEA				DX,NUMIN			;输入到缓冲区
					MOV				AH,0AH
					INT				21H
					
					MOV				DL,13
					MOV				AH,02
					INT				21H
					MOV				DL,10
					MOV				AH,02
					INT				21H
					
														;NUM中存着数字
					SUB				BX,BX
					CMP				ACT,2				;如果两位数，就将高位放入BH，低位放入BL
					JE				TWO
					MOV				AH,30H				;一位数，高位放30H（0），数字放AL
					MOV				AL,NUM
					XOR				AX,3030H
					JMP				CONV
					
TWO:				
					MOV				BH,10				;两位数的话，就高位转换成数字，乘10再加低位数字
					MOV				AL,NUM
					XOR				AL,30H
					MUL				BH
					MOV				BL,NUM+1
					MOV				BH,30H
					XOR				BX,3030H
					ADD				AX,BX
					;将输入的数字转换后，都放入AX
CONV:
					
					CMP				AX,83				;check for error code in range
					JG				O_O_R
					CMP				AX,0
					JG				IN_RANGE
					
O_O_R:
					LEA				DX,OUT_OF_RANGE
					JMP				DISP_MSG
					
;Error code is valid,determine with table to use
IN_RANGE:
					CMP				AX,35				;error code 1--35?
					JG				TRY79
					LEA				BX,ERTAB1			;yes,point to ERTAB1
					DEC				AX
					JMP				FORM_ADDR
TRY79:
					CMP				AX,79				;error code 36--79?
					JG				LAST_4
					LEA				DX,RESERVED			;yes display message
					JMP				DISP_MSG
LAST_4:
					LEA				BX,ERTAB2			;error code 80--83
					AND				AX,3
FORM_ADDR:
					SHL				AX,1				;POINT to correct offset
					ADD				BX,AX
					MOV				DX,[BX]				;put message addr into DX
DISP_MSG:
					MOV				AH,09				;display message string
					INT				21H
ET:
					RET									;return to calling program		
SHOW_ERR			ENDP
;
CSEG				ENDS
;*******************************************************************************************
					END				SHOW_ERR