TITLE GUN --- MAKES MACHINE GUN SOUND
;			  FIRES FIXED NUMBER OF SHOTS
;***********************************************************************
PROGNAM			SEGMENT								;define code segment
;-----------------------------------------------------------------------
MAIN			PROC			FAR
                ASSUME			CS:PROGNAM
                ORG				100H
START:
                MOV				CX, 25D				;枪声次数
NEW_SHOT:
                PUSH			CX
                CALL			SHOOT
                MOV				CX,9FFFH			;枪声间隔时间				更改
SILENT:
                LOOP			SILENT
                POP				CX
                LOOP			NEW_SHOT
                MOV				AL, 48H
                OUT				61H, AL
                MOV				AH, 4CH
                INT				21H
MAIN			ENDP
;-----------------------------------------------------------------------
;subroutine to make brief noise
SHOOT			PROC			NEAR
                MOV				DX, 9130H			;initial value of wait（枪声频率）
                MOV				BX, 200H			;set count（单词枪响发声次数）更改
                IN				AL, 61H				;get port 61
                AND				AL, 11111100B		;and off bits 0,1
SOUND:
                XOR				AL, 2				;toggle bit #1 in AL
                OUT				61H, AL				;output to port 61
                ADD				DX, 8FF1H			;枪声频率					更改
                MOV				CL, 3				;set to rotate 3 bits
                ROR				DX, CL				;rotate
                MOV				CX, DX				;put in CX
                AND				CX, 1FFH			;mask of upper 7 bits
                OR				CX, 10				;ensure not too short
WAITING:
                LOOP			WAITING				;wait
;made noise long enough?
                DEC				BX					;done enough?
                JNZ				SOUND				;jump if not
;turn off sound
                AND				AL, 00000010B		;and off bits 0,1
                OUT				61H, AL				;turn off bits 0,1
                RET									;return from subr
SHOOT			ENDP
;-----------------------------------------------------------------------
PROGNAM			ENDS
                END				START