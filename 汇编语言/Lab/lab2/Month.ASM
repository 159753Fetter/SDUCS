;PROGRAM TITLE GOES HERE--DIRECT
;direct(com)Direct table access
;******************************************************************
DATASG		 SEGMENT		PARA		'data'
THREE		 DB				3					;在数组定位中，每3个字节是一个月
MESS		 DB				'month?',13,10,'$'	;定义程序提示信息
MONIN		 LABEL			BYTE 
MAX			 DB				3					;输入字符数最多为3
ACT			 DB				?					;实际输入长度
MON			 DB				3 dup(?)			;存储输入的数字，最多2个数字+1个回车
;
ALFMON		 DB				'???',13,10,'$'		;作为缓冲区使用
MONTAB		 DB				'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'
;
DATASG	 	ENDS
;******************************************************************
CODESG		SEGMENT			PARA		'code'
			assume CS:CODESG,DS:DATASG,ES:DATASG
MAIN		PROC 			FAR
			PUSH		DS
			SUB			AX,AX
			PUSH		AX
			;
			MOV			AX,DATASG
			MOV			DS,AX
			MOV			ES,AX
												;	INPUT MONTH:
												;	-----------
START:
			LEA			DX,MESS
			MOV 		AH,09
			INT 		21H
			LEA 		DX,MONIN
			MOV			AH,0AH
			INT 		21H
			MOV 		DL,13
			MOV 		AH,02
			INT 		21H
			MOV			DL,10
			MOV 		AH,02
			INT 		21H
			CMP			ACT,0
			JE			EXIT
												;	CONVERT ASCII TO BINARY;
												;	------------------------
			MOV 		AH,30H					;	Set up month
			CMP			ACT,2
			JE			TWO
			MOV			AL,MON
			JMP			CONV
TWO:
			MOV			AL,MON+1
			MOV			AH,MON
CONV:
			XOR			AX,3030H				;Clear ASCII 3's
			CMP			AH,0					;MOnth 01-09?
			JZ			LOC						; yes--bypass
			SUB			AH,AH					; no--clear AH
			ADD			AL,10					; correct for binary
												;	locate month in table:
												;	----------------------
LOC:
			LEA			SI,MONTAB
			DEC			AL						;Correct for table
			MUL			THREE					;Mult AL by 3
			ADD			SI,AX
			MOV			CX,03					;Init'ze 3-char move
			CLD	
			LEA			DI,ALFMON
			REP			MOVSB					;Move 3 chars
												;	Display alpha month:
												;	--------------------
			LEA			DX,ALFMON
			MOV			AH,09
			INT			21H
			JMP			START
;
EXIT:		RET
MAIN		ENDP
;----------------------------------------------------------------------
CODESG		ENDS
;**********************************************************************
			END			MAIN