DATAREA			SEGMENT												;define data segment
COUNTER			DB				105
				DB				?
				DB				105 DUP(0)
L_NUM			DB				0
U_NUM			DB				0
N_NUM			DB				0
CRLF   			DB  			0AH, 0DH,'$'  
TEP				DB				10 DUP('$')
DATAREA			ENDS
;****************************************************************************
CODEREA			SEGMENT												;define code segment
                ASSUME			CS:CODEREA, DS:DATAREA
MAIN			PROC			FAR
START:
				PUSH			DS
				SUB				AX,AX
				PUSH			AX
				
				MOV				AX,DATAREA
				MOV				DS,AX
				
				LEA				DX, COUNTER
				MOV				AH, 0AH
                INT				21H
                MOV 			AL, COUNTER + 1                     ;对字符串进行处理
        		ADD 			AL, 2
        		MOV 			AH, 0
        		MOV 			SI, AX
        		MOV 			COUNTER[SI], '$'
                
                LEA				DX, CRLF                         	;另取一行                  
        		MOV 			AH, 09H							 
        		INT 			21H
                
           		MOV				CH, 0
                MOV				CL, COUNTER + 1  
                MOV				BX, 2
                
COUNT:
				MOV				AL,[COUNTER + BX]
				CMP				AL,65
				JB				COUNT_NUM
				CMP				AL,97
				JB				COUNT_LOW
				INC				U_NUM
				JMP				NEXT
				
COUNT_NUM:
				INC				N_NUM
				JMP				NEXT
		
COUNT_LOW:
				INC				L_NUM
				JMP				NEXT
				
NEXT:
				INC 			BX
				LOOP			COUNT

OUTPUTING:
				MOV				AL, U_NUM
                CALL			OUTPUT
                MOV				AL, L_NUM
                CALL			OUTPUT
                MOV				AL, N_NUM
                CALL			OUTPUT				
                RET
MAIN			ENDP

;----------------------------------------------------------------------------
OUTPUT			PROC			NEAR
                CMP				AL, 100
                JE				TWO
                CMP				AL, 10
                JB				ONE
                
                MOV				BL, 10
                MOV				BH, 0
                MOV				AH, 0
                DIV				BL
                MOV				BL, AH
                ADD				BL, 30H
                MOV				[TEP + 1], BL
                ADD				AL, 30H
                MOV				[TEP + 0], AL
                MOV				[TEP + 2], 0AH
                MOV				[TEP + 3], 0DH
                MOV				[TEP + 4], '$'
                LEA				DX, TEP
                MOV				AH, 09H
                INT				21H
                JMP				EXITING
                
TWO:
                MOV				[TEP + 0], 31H
                MOV				[TEP + 1], 30H
                MOV				[TEP + 2], 30H
                MOV				[TEP + 3], 0AH
                MOV				[TEP + 4], 0DH
                MOV				[TEP + 5], '$'
                LEA				DX, TEP
                MOV				AH, 09H
                INT				21H
                JMP				EXITING
                
ONE:
                ADD				AL, 30H
                MOV				[TEP + 0], AL
                MOV				[TEP + 1], 0AH
                MOV				[TEP + 2], 0DH
                MOV				[TEP + 3], '$'
                LEA				DX, TEP
                MOV				AH, 09H
                INT				21H
                JMP				EXITING
                
EXITING:
                RET
                
OUTPUT			ENDP

;----------------------------------------------------------------------------
CODEREA			ENDS
                END				START