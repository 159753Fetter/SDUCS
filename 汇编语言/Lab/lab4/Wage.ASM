;PROGRAM TITLE GOES HERE--SCREMP
;Enter hours&rate,display wage
;******************************************************************
STACKSG				SEGMENT				PARA 						STACK'stack'
					DW					32 DUP(?)
STACKSG				ENDS
;******************************************************************
DATASG				SEGMENT				PARA 						'data'
HRSPAR				LABEL				BYTE						;Hours parameter list
MAXHLEN				DB					6
ACTHLEN				DB					?
HRSFLD				DB					6 DUP (?)

RATEPAR				LABEL				BYTE						;Rate parameter list
MAXRLEN				DB					6
ACCLEN				DB					?
RATETID				DB					6 DUP (?)

MESSG1				DB					'Hours worked?','$'
MESSG2				DB					'Rate of pay?','$'
MESSG3				DB					'Wage =='
ASCWAGE				DB					14 DUP(30H),13,10,'$'

MESSG4				DB					13,10,'Overflow!',13,10,'$'
ADJUST				DW					?
BINVAL				DW					0
BINHRS				DW					0
BINRATE				DW					0
COL					DB					0
DECIND				DB					0
MULT10				DW					01
NODEC				DW					0
ROW					DB					0
SHIFT				DW					?
TENWD				DW					10
TEMPDX				DW					?
TEMPAX				DW					?
DATASG				ENDS
;******************************************************************
CODESG				SEGMENT				PARA						'code'
;-------------------------------------------------------------------
BEGIN				PROC				FAR							;main part of program
					ASSUME				CS:CODESG, DS:DATASG, SS:STACKSG, ES:DATASG
					
					PUSH				DS							;save old data segment
					SUB					AX, AX						;put zero in AX
					PUSH				AX							;save it on stack
					
					MOV					AX, DATASG					;data segment addr
					MOV					DS, AX						;into DS register
					MOV					ES, AX						;into ES register
					
					MOV					AX, 0600H
					CALL				Q10SCR						;clear screent
					CALL				Q20CURS						;set curosr
A20LOOP:
					CALL				B10INPT						;accept hours & rate
					CMP					ACTHLEN, 0					;End of input?
					JE					A30							;如果输入0，则清空屏幕退出
					CALL				D10HOUR						;convert hours to binary
					CALL				E10RATE						;convert rate to binary
					CALL				F10MULT						;calc wage, round
					CALL				G10WAGE						;convert wage to ASCII
					CALL				K10DISP						;display wage
					JMP					A20LOOP
					
A30:
					MOV					AX, 0600H
					CALL				Q10SCR						;Clear screen
					RET												;return to DOS
BEGIN				ENDP											;end of main of program
;----------------------------------------------------------------------------
;					Input hours & rate
B10INPT				PROC				NEAR
					LEA					DX, MESSG1					;Prompt for hours
					MOV					AH, 09H
					INT					21H
					LEA					DX, HRSPAR					;accepthours
					MOV					AH, 0AH
					INT					21H							;输入工作小时数
					CMP					ACTHLEN, 0					;如果工作小时未输入，退出
					JNE					B20							;否则跳转到B20部分
					RET
B20:
					MOV					COL, 25						;列设为25
					CALL				Q20CURS						;重置光标
					LEA					DX, MESSG2
					MOV					AH, 09H
					INT					21H
					LEA					DX, RATEPAR					;输入rate值
					MOV					AH, 0AH
					INT					21H
					RET
B10INPT				ENDP
;----------------------------------------------------------------------------
D10HOUR				PROC				NEAR
					MOV					NODEC, 0
					MOV					CL, ACTHLEN					;把位数放入CL
					SUB					CH, CH
					LEA					SI, HRSFLD-1				;输入的数字的的前一位地址赋值到SI中
					ADD					SI, CX						;定位到最后一位数字
					CALL				M10ASBI
					MOV					AX, BINVAL
					MOV					BINHRS, AX					;把得到的最终结果放入BINHRS中
					RET
D10HOUR				ENDP
;----------------------------------------------------------------------------
E10RATE				PROC				NEAR
					MOV					CL, ACTHLEN
					SUB					CH, CH
					LEA					SI, RATETID - 1				;整体思路与D10HOUR差不多
					ADD					SI, CX
					CALL				M10ASBI
					MOV					AX, BINVAL
					MOV					BINRATE, AX					;把得到的最终结果放入BINHRS中
					RET
E10RATE				ENDP
;----------------------------------------------------------------------------
F10MULT				PROC				NEAR
					MOV					CX, 07
					LEA					DI, ASCWAGE
					MOV					AX, 3030H
					CLD
					REP					STOSW
					MOV					SHIFT, 10
					MOV					ADJUST, 0
					MOV					CX, NODEC
					CMP					CL, 06
					JA					F40							;如果小数位数超过6位，则溢出，显示0
					DEC					CX
					DEC					CX
					JLE					F30							;如果小数位数少于两位，则转向F30
					MOV					NODEC, 02					;否则小数点视为2位
					MOV					AX, 01
F20:
					MUL					TENWD
					LOOP				F20
					MOV					SHIFT, AX					;移位因子SHIFT = 10^(NODEC-2)
					SHR					AX, 1
					MOV					ADJUST, AX					;舍入 = SHIFT / 2
F30:
					MOV					AX, BINHRS
					MUL					BINRATE						;AX此时为product低位
					ADD					AX, ADJUST
					ADC					DX, 0						;带进位加法，加上高位
					MOV					TEMPDX, DX					;product高位
					MOV					TEMPAX, AX					;product低位
					
					CMP					ADJUST, 0
					JZ					F50
					
					MOV					AX, DX						;分别将高位低位除以shift得到最终答案
					MOV					DX, 0
					DIV					SHIFT
					MOV					TEMPDX, AX
					MOV					AX, TEMPAX
					DIV					SHIFT
					MOV					DX, TEMPDX
					MOV					TEMPAX, AX
					JMP					F50
F40:
					MOV					AX, 0
					MOV					DX, 0
F50:
					RET
F10MULT				ENDP
;----------------------------------------------------------------------------
;convert to ASCII
G10WAGE				PROC				NEAR
					LEA					SI, ASCWAGE+11
					MOV					BYTE PTR[SI],'.'
					ADD					SI, NODEC						;set right start pos'n
G30:
					CMP					BYTE PTR[SI],'.'
					JNE					G35								;不是两位小数，跳G35进行调整到两位小数
					DEC					SI
G35:
					CMP					DX, 0
					JNZ					G40
					CMP					AX, 0010
					JB					G50								;只有低位且小于10，跳转G50
G40:
					MOV					AX, DX
					MOV					DX, 0
					DIV					TENWD
					MOV					TEMPDX, AX
					MOV					AX, TEMPAX
					DIV					TENWD
					MOV					TEMPAX, AX						;数字部分除以10，以调整小数点位数
					OR					DL, 30H
					MOV					[SI], DL
					DEC					SI
					MOV					DX, TEMPDX
					JMP					G30
G50:																	;转换成ASCII码，重新放回对应位置
					OR					AL, 30H
					MOV					[SI], AL
					RET
G10WAGE				ENDP
;----------------------------------------------------------------------------
;----------------------------------------------------------------------------
;display wage
K10DISP				PROC				NEAR
					MOV					COL, 50							;Set column
					CALL				Q20CURS
					MOV					CX, 10
					LEA					SI, ASCWAGE
K20:																	;Clear leading zeros
					CMP					BYTE PTR[SI], 30H
					JNE					K30								;to blanks
					MOV					BYTE PTR[SI], 20H
					INC					SI
					LOOP				K20
K30:
					LEA					DX, MESSG3						;Display
					MOV					AH, 09
					INT					21H
					CMP					ROW, 20							;Bottom of screen?
					JAE					K80
					INC					ROW								;no--increment row
					JMP					K90
K80:
					MOV					AX, 0601H						;yes--
					CALL				Q10SCR							;scroll &
					MOV					COL, 0							;set cursor
					CALL				Q20CURS
K90:
					RET
K10DISP				ENDP
;----------------------------------------------------------------------------
;convert ASCII to binary

M10ASBI				PROC				NEAR
					MOV					MULT10, 01
					MOV					BINVAL, 0
					MOV					DECIND, 0
					SUB					BX, BX
M20:
					MOV					AL,[SI]								;将该位的值放入AL
					CMP					AL, '.'
					JNE					M40									;如果不是'.'就去M40
					MOV					DECIND, 01							;反之去M90,DECIND设为1
					JMP					M90
M40:
					AND					AX, 000FH							;取后四位（此时将ASCII码转换为了数字，因为30H代表0，以此类推）
					MUL					MULT10								;乘MULT10（即该数中每个字符代表的权重，因为是10进制所以为10）
					JC					OVERFLOWING							;进位说明溢出
					ADD					BINVAL, AX							;将AX值加入BINVAL（最终结果）中
					JC					OVERFLOWING							;进位说明溢出
					MOV					AX, MULT10
					MUL					TENWD								;MULT10乘上TENWD，再回归原位，TENWD是进制（自然运算为10进制）
					MOV					MULT10, AX							;MULT10代表每个数字所代表的位数权重
					CMP					DECIND, 0							;如果DECIND为0，BX自增（储存小数位数），反之去M90
					JNZ					M90
					INC					BX
M90:
					DEC					SI									;取前一位数字
					LOOP				M20									;重新进行处理判断是否为点
					CMP					DECIND, 0
					JZ					M100								;如果重复出现小数点，则退出
					ADD					NODEC, BX							;说明是小数部分，将记录的小数位数（BX）加入NODEC中
					JMP					M100
OVERFLOWING:
					MOV					BINVAL, 0
M100:
					RET
M10ASBI				ENDP
;----------------------------------------------------------------------------
;scroll screen

Q10SCR				PROC				NEAR							;AX set on entry(AH = 06)
					MOV					BH, 07							;color，卷入行属性
					SUB					CX, CX							;左上角行列号为0
					MOV					DX, 184FH						;右下角行号为18H，列号为4FH
					INT					10H								;全屏幕空白
					RET
Q10SCR				ENDP
;----------------------------------------------------------------------------
;set cursor

Q20CURS				PROC				NEAR
					MOV					AH, 2
					SUB					BH, BH
					MOV					DH, ROW
					MOV					DL, COL
					INT					10H								;将光标位置放到第0页（BH），ROW行，COL列
					RET
Q20CURS				ENDP
;----------------------------------------------------------------------------
CODESG				ENDS
;******************************************************************
					END				