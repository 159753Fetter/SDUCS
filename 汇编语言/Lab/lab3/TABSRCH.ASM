;PROGRAM TITLE GOES HERE--tabsrch
;Table search
;****************************************************************
datasg			segment		para		'data'
mess1			db			'stock nember?',13,10,'$'
;
STOKNIN			LABEL		BYTE
  MAX			DB			3
  ACT			DB			?
  STOKN			DB			3 DUP(?)
;
STOKTAB			DB			'05',' EXCAVATORS '
				DB			'08',' LIFTERS    '
				DB			'09',' PRESSES    '
				DB			'12',' VALVES     '
				DB			'23',' PROCESSORS '
				DB			'27',' PUMPS      '
;
DESCRN			DB			14 DUP(20H),13,10,'$'
MESS			DB			'NOT IN TABLE!','$'
DATASG			ENDS
;****************************************************************
CODESG			SEGMENT		PARA	'CODE'
				ASSUME		CS:CODESG,DS:DATASG,ES:DATASG
;----------------------------------------------------------------
MAIN			PROC		FAR
				PUSH		DS					;save old data segment
				SUB 		AX,AX				;put zero in AX
				PUSH		AX					;save it on stack
				MOV			AX,datasg			;data segment addr
				MOV			DS,AX				;into DS register
				MOV			ES,AX				;into ES register
;MAIN PART OF PROGRAM GOES HERE
START:
				LEA			DX,MESS1			;PROMPT FOR STOCK NUMBER
				MOV			AH,09
				INT			21H
				LEA			DX,STOKNIN
				MOV			AH,0AH
				INT			21H
				CMP			ACT,0
				JE			ET
				MOV			AL,STOKN			;GET STOCK
				MOV			AH,STOKN+1
				MOV			CX,06				;NO. OF ENTRIES
				LEA			SI,STOKTAB			;INIT'ZE TABLE ADDRESS
A20:		
				
				CMP			AX,WORD PTR[SI]
				JE			A30					;EQUAL - EXIT
				ADD			SI,14				;NOT EQUAL - INCREMENT
				LOOP		A20
				LEA			DX,MESS				;NOT IN TABLE
				MOV			AH,09
				INT			21H
				JMP			ET
A30:
				MOV			CX,07
				LEA			DI,DESCRN			;LENGTH OF DESCR'N
				REP			MOVSW
;
				LEA			DX,DESCRN
				MOV			AH,09
				INT			21H
				JMP			START
ET:
				RET								;return to DOS
MAIN			ENDP
;-----------------------------------------------------------------------
CODESG			ENDS							;end of code segment
				END				MAIN			;end assembly
				