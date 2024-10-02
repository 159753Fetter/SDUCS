;PROGRAM TITLE GOES HERE -- Compare string
;***********************************************************
DATAREA SEGMENT    ;define data SEGMENT
string1		DB  'Move the cursor backward.'
string2		DB  'Move the cursor backward.'
;
mess1		DB	'Match.',13,10,'$'
mess2		DB	'No match!',13,10,'$'

DATAREA		ENDS
;***********************************************************

PROGNAM         SEGMENT         ;define code segment
;-----------------------------------------------------------

main	proc	far
		ASSUME		CS:PROGNAM,DS:DATAREA,ES:DATAREA
START:				;starting execution address

;set up stack for return
	PUSH	DS	;save old data segment 压栈，保持当前状态
	SUB		AX,AX	;put zero in AX
	PUSH	AX		;save it on stack
;set DS	register to current data segment
	MOV AX,DATAREA; datarea segment addr
	MOV DS,AX	;into DS register 
	MOV	ES,AX	;into ES register
;MAIN PART OF PROGRAM GOES HERE
	LEA		SI,string1	;将string1的偏移地址放到SI中，使SI指向string1
	LEA		DI,string2	;将string2的偏移地址放到DI中，使DI指向string2
	CLD					;将标志寄存器Flag的方向标志位DF清零。
	MOV		CX,25		;为后面循环做准备，CX=25，循环25次判断字符串是否一致
	REPZ	CMPSB		
	JZ		MATCH
	LEA		DX,mess2
	JMP		SHORT DISP
MATCH:
	LEA		DX,mess1
DISP:
	MOV		AH,09
	int		21H
	RET										;return to DS
	
main	ENDP								;end of main part of program
;-----------------------------------------------------------
PROGNAM         ENDS						;end of code segment
;***********************************************************
                END             START		;end assembly