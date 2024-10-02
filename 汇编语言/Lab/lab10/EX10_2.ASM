DATAREG			SEGMENT
ROW				DB				11
DATAREG			ENDS
;****************************************************************************
CODEREG			SEGMENT
				ASSUME			CS:CODEREG, DS:DATAREG
MAIN			PROC			FAR
START:
				PUSH			DS
				SUB				AX, AX
				PUSH			AX
				
				MOV				AX, DATAREG
				MOV				DS, AX
				MOV				DL, 21H
				
RESTART:
				MOV				CX, 20			;每行循环次数
				
OUTPUT:
				MOV				AH, 2			;输出字符
                INT				21H
				PUSH			DX
                MOV				DL, 0			;输出空格
                MOV				AH, 2
                INT				21H	
                POP				DX
				INC				DL
				
				LOOP	 		OUTPUT 			;循环输出每行的16个ASCII元素
				
NEXT:
      			PUSH 			DX				;将下一行的首元素推入堆栈
      			MOV  			DL, 0DH    
      			MOV  			AH, 02H                  
      			INT  			21H
     			MOV  			AH, 02H
      			MOV  			DL, 0AH
      			INT  			21H 			;输出换行

      			POP 			DX 				;将栈顶元素推出
				
                DEC				ROW
                CMP				ROW, 0
                JNE				RESTART
                JE				EXITING
EXITING:
				RET
;----------------------------------------------------------------------------
CODEREG			ENDS
				END				START
