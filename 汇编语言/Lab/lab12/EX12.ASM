DATAREG				SEGMENT
SEMIBREVE			DW				262,294,330,262,262,294,330,262
     				DW				330,349,392,330,349,392
     				DW				392,440,392,349,330,262,392,440,392,349,330,262
     				DW				294,196,262,294,196,262
				
METER				DW				2500,2500,2500,2500,2500,2500,2500,2500
                	DW				2500,2500,5000,2500,2500,5000
               	 	DW				1200,1200,1200,1200,2500,2500,1200,1200,1200,1200,2500,2500
                	DW				2500,2500,5000,2500,2500,5000
                	
DATAREG				ENDS
;----------------------------------------------------------------------------
CODEREG			SEGMENT
MAIN			PROC		FAR
				ASSUME CS:CODEREG,DS:DATAREG
START:
				PUSH		DS
				XOR			AX,AX
				PUSH		AX
					
				MOV			AX,DATAREG
				MOV			DS,AX
				LEA			DI, SEMIBREVE
                LEA			BX, METER
				MOV			CX,32
					
NEW_SHOT:
				PUSH 		CX
				CALL		SHOT
				POP			CX
				ADD			DI, 2
				ADD			BX, 2
					
STOP:
				LOOP		NEW_SHOT
               	MOV			AH, 4CH
				INT			21H
				RET
MAIN			ENDP	
;----------------------------------------------------------------------------
SHOT			PROC		NEAR
                IN			AL, 61H				;get port 61
               	AND			AL, 11111100B		;and off bits 0,1
SOUND:
                XOR			AL, 2				;toggle bit #1 in AL
                OUT			61H, AL				;output to port 61
                PUSH		AX
                PUSH		BX
                MOV			AX, 2801
                MOV			BX, 50
                MUL			BX
                DIV			WORD PTR [DI]
                MOV			DX, AX				;rotate
                POP			BX
                POP			AX
                MOV			CX, DX				;put in CX
WAITING:
                LOOP		WAITING				;wait
;made noise long enough?
                DEC			WORD PTR [BX]		;done enough?
                JNZ			SOUND				;jump if not
;turn off sound
                AND			AL, 00000010B		;and off bits 0,1
                OUT			61H, AL				;turn off bits 0,1
                RET
SHOT			ENDP
;----------------------------------------------------------------------------
CODEREG			ENDS
				END 	START