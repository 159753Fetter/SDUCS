DATAREG			SEGMENT
COUNT			DB				0

NAMING 			LABEL			BYTE									;定义字符串记录姓名
N_MAX_LEN		DB				21
N_LEN			DB				?
THE_NAME		DB				21 DUP(' ')

PHONING			LABEL			BYTE									;定义字符串记录电话
P_MAX_LEN		DB				9
P_LEN			DB				?
THE_PHONE		DB				9 DUP(' ')

MESSAGE1		DB				'Input name:','$'						;给出相应的提示信息
MESSAGE2		DB				'Input a telephone number:','$'
MESSAGE3		DB				'Do you want a telephone number?(Y/N)','$'
MESSAGE4		DB				'name?','$'
MESSAGE5		DB				'name                tel.','$'

THE_CRLF		DB				13, 10, '$'
THE_NUM			DB				20 DUP(28 DUP(' '))						;定义二维数组记录姓名以及对应电话
THE_TMP			DB				28 DUP(' '),'$'
ASK_PHONE		DB				20 DUP(' ')
THE_SORT		DB				28 DUP(' ')

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
				MOV				ES, AX									;为了使用REP指令，让ES寄存器也指向数据段
				MOV				DS, AX
				
				MOV				AH, 01H									;根据题目要求多次调用
                INT				21H
                SUB				AL, 30H
				MOV				COUNT,AL
				
				MOV				AH, 01H
                INT				21H
                CMP				AL, 13
                JE				OVER
                
                SUB				AL, 30H
                ADD				COUNT, AL
                ADD				COUNT, 9
                
                MOV				AX, 0
                MOV				AH, 01H
                INT				21H
                CMP				AL, 13
                JE				OVER
OVER:
                MOV				CL, COUNT							
                MOV				CH, 0
                LEA				DI, THE_NUM
				
INPUTING:
				CALL			CLEAR									;这是为了先把原来存在的数据清空
				CALL			INPUT									;这是输入部分
				CALL			STORE
                LOOP			INPUTING

ASKING:
				LEA				DX,MESSAGE3
				MOV				AH, 09H
				INT				21H
				MOV				AH, 01H
				INT				21H
				CMP				AL, 'Y'
				JE				FINDING
				CALL			CRLF									;如果不相等说明为N，则输出后直接退出即可
				CALL			OUTPUT
				JMP				EXITING
FINDING:
				CALL			CRLF
                LEA				DX,MESSAGE4
                MOV				AH, 09H
                INT				21H
                
                LEA				DX, NAMING								;输入姓名
                MOV				AH, 0AH
                INT				21H
                MOV				CH, 0
                MOV				CL, N_LEN
                LEA				SI, NAMING
                ADD				SI, CX
                INC				SI
                INC				SI
                MOV				BYTE PTR[SI], 32
                
				CALL			CRLF
				CALL			FIND						   			;在二维数组里面进行查找，找到了就输出信息即可
				CALL			CLEAR1
				LEA				DX, MESSAGE5
                MOV				AH, 09H
                INT				21H
                CALL			CRLF
                LEA				DX, THE_TMP
                MOV				AH, 09H
                INT				21H
                CALL			CRLF
                JMP				ASKING
EXITING:
				MOV				AH,4CH
      			INT				21H
MAIN			ENDP        

;----------------------------------------------------------------------------
CLEAR			PROC			NEAR									;每次清空输入时候的两个字符串
                PUSH			CX
                PUSH			AX
                PUSH			DI
                LEA				DI, NAMING
                MOV				AL, 32
                MOV				CX, 20
                REP				STOSB
                LEA				DI, PHONING
                MOV				AL, 32
            	MOV				CX, 9
            	REP				STOSB
                POP				DI
                POP				AX
                POP				CX             
                RET
CLEAR			ENDP

;----------------------------------------------------------------------------
CLEAR1			PROC			NEAR									;每次清空输出时候的两个字符串
                PUSH			CX
                PUSH			AX
                PUSH			DI
                LEA				DI, NAMING
                MOV				AL, 32
                MOV				CX, 20
                REP				STOSB
                LEA				DI, ASK_PHONE
                MOV				AL, 32
            	MOV				CX, 20
            	REP				STOSB
                POP				DI
                POP				AX
                POP				CX
                RET
CLEAR1			ENDP

;----------------------------------------------------------------------------
OUTPUT			PROC			NEAR
				CALL			SORT
				MOV				CL, COUNT
                MOV				CH, 0
                LEA				BX, THE_NUM
                
OUTPUTING:
                PUSH			CX
                MOV				SI, BX
                LEA				DI, THE_TMP
                MOV				CX, 14
                REP				MOVSW
                ADD				BX, 28
                LEA				DX, THE_TMP
                MOV				AH, 09H
                INT				21H
                CALL			CRLF
                POP				CX
                LOOP			OUTPUTING
                RET
OUTPUT			ENDP

;----------------------------------------------------------------------------
CRLF			PROC			NEAR
                LEA				DX, THE_CRLF
                MOV				AH, 09H
                INT				21H
                RET
CRLF			ENDP

;----------------------------------------------------------------------------
STORE			PROC			NEAR
                PUSH			CX
                LEA				SI, THE_NAME
                MOV				CX, 20
                REP				MOVSB
                LEA				SI, THE_PHONE
                MOV				CX, 8
                REP				MOVSB
                POP				CX
                RET
STORE			ENDP

;----------------------------------------------------------------------------
FIND			PROC			NEAR
    			PUSH			CX
                
                MOV				CL, NAMING + 1
                MOV				CH, 0
                LEA				SI, NAMING
                INC				SI
                INC				SI
                LEA				DI, ASK_PHONE
                REP				MOVSB
                
                LEA				SI, ASK_PHONE
                LEA				DI, THE_NUM
                MOV				BX, DI
                MOV				CL, COUNT
                MOV				CH, 0
I3:
                PUSH			CX
                MOV				CH, 0
                MOV				CL, 20
                REPZ			CMPSB
                JE				FIND_EXIT
                POP				CX
                ADD				BX, 28
                MOV				DI, BX
                LEA				SI, ASK_PHONE
                LOOP			I3
                JMP				TO_EXIT
FIND_EXIT:
                POP				CX
                MOV				SI, BX
                LEA				DI, THE_TMP
                MOV				CX, 14
                CLD
                REP				MOVSW
TO_EXIT:
                POP				CX        
                RET
FIND			ENDP

;----------------------------------------------------------------------------				
INPUT			PROC			NEAR
				PUSH			CX
				PUSH			SI
                LEA				DX, MESSAGE1
                MOV				AH, 09H
                INT				21H
                LEA				DX, NAMING
				MOV				AH, 0AH
                INT				21H
                
                MOV				CH, 0
                MOV				CL, N_LEN
                LEA				SI, NAMING								;输入姓名
                ADD				SI, CX
                INC				SI
                INC				SI
                MOV				BYTE PTR[SI], 32
        		
        		CALL			CRLF
        		LEA				DX, MESSAGE2
                MOV				AH, 09H
                INT				21H
        		LEA				DX, PHONING
				MOV				AH, 0AH
                INT				21H
                
                MOV				CH, 0									;输入电话
                MOV				CL, P_LEN
                LEA				SI, PHONING
                ADD				SI, CX
                INC				SI
                INC				SI
                MOV				BYTE PTR[SI], 32
     
        		CALL			CRLF
                POP				SI
                POP				CX
                RET
INPUT			ENDP

;----------------------------------------------------------------------------
SORT			PROC			NEAR
                MOV				CH, 0								;CH = i
OUTER_FOR:															;for(int i=1;i<n;++i)
                INC				CH
                CMP				CH, COUNT
                JNB				BREAKING
                LEA				SI, THE_NUM
                LEA				DI, THE_NUM + 28					;DI = a[j+1], SI = a[j]
                MOV				CL, 0								;CL = j
INNER_FOR:
                MOV				BL, COUNT
                SUB				BL, CH
                CMP				CL, BL								;j < COUNT - i - 1
                JNB				OUTER_FOR							;for(int j=0;j<COUNT - i - 1;++j)
                PUSH			CX
                MOV				CX, 20
                MOV				BX, DI
                MOV				AX, SI
                REPE			CMPSB
                JBE				NEXT_FOR							;a[j] > a[j + 1]----->CHANGE
                CALL			CHANGE								;a[j] <= a[j+1]------>next i
NEXT_FOR:
                POP				CX
                MOV				DI, BX
                MOV				SI, AX
                ADD				SI, 28
                ADD				DI ,28
                INC				CL
                JMP				INNER_FOR
BREAKING:
                RET
SORT			ENDP

;----------------------------------------------------------------------------
CHANGE			PROC			NEAR
                MOV				SI, BX						;temp = a[j+1]
                LEA				DI, THE_SORT
                MOV				CX, 14
                REP				MOVSW
                
                MOV				DI, BX						;a[j+1] = a[j]
                MOV				SI, AX
                MOV				CX, 14
                REP				MOVSW
                
                MOV				DI, AX						;a[j] = temp
                LEA				SI, THE_SORT
                MOV				CX, 14
                REP				MOVSW
                RET
CHANGE			ENDP
;----------------------------------------------------------------------------
CODEREG			ENDS
				END				START
