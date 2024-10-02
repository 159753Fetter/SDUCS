TITLE TYPE_EX --- TEST TIME FOR TYPING EXERCISE
; ---------------------------------------------------------------
STACK           SEGMENT         PARA  STACK  'STACK'
                DB              256	DUP(0)
TOP             LABEL           WORD  
STACK           ENDS
; ---------------------------------------------------------------
DATA            SEGMENT         PARA  PUBLIC  'DATA'
BUFFER          DB              16H DUP(0)
BUFPT1          DW              0
BUFPT2          DW              0
; BUFPT1=BUFPT2, the buffer is empty
KBFLAG          DB              0
PROMPT          DB              '     * PLEASE PRACTISE TYPING *', 0DH, 0AH, '$'
SCANTAB         DB              0, 0, '1234567890-=', 8, 0
                DB              'qwertyuiop[]', 0DH, 0
                DB              'asdfghjkl;', 0, 0, 0, 0
                DB              'zxcvbnm,./', 0, 0, 0
                DB              ' ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                DB              '789-456+1230.'
EVEN
OLDCS9          DW              ?
OLDIP9          DW              ?
; ----------------------------------------------------------------
STR1            DB              'abcd efgh ijkl mnop qrst uvwx yz.'
                DB              0DH, 0AH, '$'
STR2            DB              'christmas is a time of joy and love.'
                DB              0DH, 0AH, '$'
STR3            DB              'store windows hold togs and gifts.'
                DB              0DH, 0AH, '$'
STR4            DB              'people send chirstmas cards and gifts.'
                DB              0DH, 0AH, '$'
STR5            DB              'santa wish all people peace on earth.'
CRLF            DB              0DH, 0AH, '$'
COLON           DB              ':', '$'
EVEN
SADDR           DW              STR1, STR2, STR3, STR4, STR5
COUNT           DW              0
SEC             DW              0
MIN             DW              0
HOURS           DW              0
SAVE_LC         DW              2  DUP(?)
DATA            ENDS
; ------------------------------------------------------------------
CODE            SEGMENT
                ASSUME  CS:CODE, DS:DATA, ES:DATA, SS:STACK
MAIN            PROC            FAR
START:
                MOV             AX, STACK                   ; set up stack			 
                MOV             SS, AX
                MOV             SP, OFFSET TOP

                PUSH            DS		                    ; save ds:0 for return	       
                SUB             AX, AX
                PUSH            AX
                MOV             AX, DATA		            ; set DS to data segment		
                MOV             DS, AX
                MOV             ES, AX

                MOV             AH, 35H                     ; save interrupt vector
                MOV             AL, 09H                     ; of keyboard
                INT             21H
                MOV             OLDCS9, ES
                MOV             OLDIP9, BX

                PUSH            DS                          ; set interrupt vector
                MOV             DX, SEG KBINT               ; of kbint 
                MOV             DS, DX
                MOV             DX, OFFSET KBINT
                MOV             AL, 09H
                MOV             AH, 25H
                INT             21H
                POP             DS

                MOV             AH, 35H                     ; save interrupt vector 
                MOV             AL, 1CH                     ; of timer
                INT             21H
                MOV             SAVE_LC, BX
                MOV             SAVE_LC+2, ES

                PUSH            DS                          ; set interrupt vector
                MOV             DX, SEG CLINT               ; of clint
                MOV             DS, DX
                MOV             DX, OFFSET CLINT
                MOV             AL, 1CH
                MOV             AH, 25H
                INT             21H
                POP             DS
	
                IN              AL, 21H                     ; clear kbd & timer
                AND             AL, 11111100B               ; mask bit 
                OUT             21H, AL        
FIRST:   
                MOV             AH, 0                       ; set video mode
                MOV             AL, 3                       ; 80x25 color text
                INT             10H

                MOV             DX, OFFSET PROMPT
                MOV             AH, 9                       ; display kbd message
                INT             21H
	
                MOV             SI, 0
NEXT:   
                MOV             DX, SADDR[SI]               ; display sentence
                MOV             AH, 09H
                INT             21H

                MOV             COUNT, 0	                ; set initial value		 
                MOV             SEC, 0
                MOV             MIN, 0
                MOV             HOURS, 0

                STI                                         ; set IF flag
FOREVER:    
                CALL            KBGET                       ; wait enter a key 
                TEST            KBFLAG, 80H
                JNZ             ENDINT
                PUSH            AX
                CALL            DISPCHAR                    ; display the character
                POP             AX
                CMP             AL, 0DH
                JNZ             FOREVER
                MOV             AL, 0AH
                CALL            DISPCHAR                    ; display CR/LF

                CALL            DISPTIME                    ; display typping time 

                LEA             DX, CRLF                    ; display CR/LF
                MOV             AH, 09H
                INT             21H

                ADD             SI, 2                       ; update pointer
                CMP             SI, 5*2                     ; end of sentences ?
                JNE             NEXT                        ; no, display next 
                JMP             FIRST                       ; yes, display first 

ENDINT: 
                CLI                                         ; end of typing 
                PUSH            DS
                MOV             DX, SAVE_LC
                MOV             AX, SAVE_LC+2
                MOV             DS, AX
                MOV             AL, 1CH                     ; reset interrupt vector
                MOV             AH, 25H                     ; of type 1ch
                INT             21H
                POP             DS

                PUSH            DS
                MOV             DX, OLDIP9
                MOV             AX, OLDCS9
                MOV             DS, AX
                MOV             AL, 09H                     ; reset interrupt vector 
                MOV             AH, 25H                     ; of type 09h
                INT             21H
                POP             DS
	
                STI
                RET                                         ; return to DOS 
MAIN            ENDP
; -----------------------------------------------------------------------------------
CLINT           PROC            NEAR                        ; timer int routine 
                PUSH            DS                          ; save ROM data area
                MOV             BX, DATA                    ; set data segment
                MOV             DS, BX

                LEA             BX, COUNT
                INC             WORD PTR[BX]                ; increment count
                CMP             WORD PTR[BX], 18            ; 1 sec = 18 count
                JNE             RETURN
                CALL            INCT                        ; update sec and min 
ADJ:  
                CMP             HOURS, 12                   ; update hours 
                JLE             RETURN
                SUB             HOURS, 12
RETURN:
                POP             DS
                STI
                IRET                                        ; interrupt return 
CLINT           ENDP
; --------------------------------------------------------------------------------------
INCT            PROC            NEAR                        ; update sec and min
                MOV             WORD PTR[BX], 0
                ADD             BX, 2
                INC             WORD PTR[BX]
                CMP             WORD PTR[BX], 60
                JNE             EXIT
                CALL            INCT
EXIT:  
                RET                                         ; return to clint
INCT            ENDP
; ---------------------------------------------------------------------------------------
DISPTIME        PROC            NEAR
; subroutime to display typping time for min:sec:msec
                MOV             AX, MIN
                CALL            BINDEC                      ; display min 
	
                MOV             BX, 0                       ; display':'
                MOV             AL, ':'
                MOV             AH, 0EH
                INT             10H
                MOV             AX, SEC                     ; display sec 
                CALL            BINDEC
	
                MOV             BX, 0                       ; display':'
                MOV             AL, ':'
                MOV             AH, 0EH
                INT             10H
	
                MOV             BX, COUNT                   ; count convert to ms 
                MOV             AL, 55D
                MUL             BL
                CALL            BINDEC                      ; display ms 
	
                RET                                         ; return to main 
DISPTIME        ENDP     
; ----------------------------------------------------------------------------------------
BINDEC          PROC            NEAR
; subroutime to convert binary in AX to decimal
                MOV             CX, 100D
                CALL            DECDIV
                MOV             CX, 10D
                CALL            DECDIV
                MOV             CX, 1
                CALL            DECDIV
                RET                                         ; return to disptime 
BINDEC          ENDP
; -----------------------------------------------------------------------------------------
DECDIV          PROC            NEAR
; sub_subroutine divide number in AX by CX 
                MOV             DX, 0                       ; number hight half
                DIV             CX

                MOV             BX, 0
                ADD             AL, 30H                     ; convert to ASCII 
                MOV             AH, 0EH                     ; display function
                INT             10H

                MOV             AX, DX
                RET                                         ; return to bindec
DECDIV          ENDP
; ********************************************************************************************
KBGET           PROC            NEAR                        ; kbd interrupt routine
                PUSH            BX
                CLI                                         ; interrupt back off                  
                MOV             BX, BUFPT1                  ; get pointer to head
                CMP             BX, BUFPT2                  ; test empty of buffer 
                JNZ             KBGET2                      ; no, fetch a character
                CMP             KBFLAG, 0 
                JNZ             KBGET3
                STI                                         ; allow an interrupt to occur
                POP             BX
                JMP             KBGET                       ; loop until something in buff
KBGET2: 
                MOV             AL, [BUFFER+BX]             ; get ascii code 
                INC             BX                          ; inc a buffer pointer    
                CMP             BX, 16                      ; at end of buffer ?          
                JC              KBGET3                      ; no, continue
                MOV             BX, 0                       ; reset to buf beginning
KBGET3: 
                MOV             BUFPT1, BX                  ; store value in variable
                POP             BX
                RET                                         ; return to main
KBGET           ENDP                                        
; --------------------------------------------------------------------------------------------
KBINT           PROC            FAR                         ; keyboard interrupt routine
                PUSH            BX                          
                PUSH            AX

                IN              AL, 60H                     ; read in the character
                PUSH            AX                          ; save it    
                IN              AL, 61H                     ; get the control port            
                OR              AL, 80H                     ; set acknowledge bit for kbd
                OUT             61H, AL
                AND             AL, 7FH                     ; reset acknowledge bit 
                OUT             61H, AL

                POP             AX                          ; recover scan code 
                TEST            AL, 80H                     ; is press or release code?
                JNZ             KBINT2                      ; is release code, return 
                MOV             BX, OFFSET SCANTAB
                XLAT            SCANTAB                     ; ascii code to AL 
                CMP             AL, 0   
                JNZ             KBINT4
                MOV             KBFLAG, 80H
                JMP             KBINT2
KBINT4: 
                MOV             BX, BUFPT2                  ; buffer tail pointer 
                MOV             [BUFFER+BX], AL             ; ASCII fill in buffer
                INC             BX
                CMP             BX, 16                      ; is end of buffer?    
                JC              KBINT3                      ; no         
                MOV             BX, 0                       ; reset to buf beginning                  
KBINT3: 
                CMP             BX, BUFPT1                  ; is buffer full?               
                JZ              KBINT2                      ; yes, lose character                   
                MOV             BUFPT2, BX                  ; save buf tail pointer
KBINT2:
                CLI
                MOV             AL, 20H                     ; end of interrupt 
                OUT             20H, AL
                POP             AX 
                POP             BX
                STI
                IRET                                        ; interrupt return 
KBINT           ENDP
; --------------------------------------------------------------------------------------------
DISPCHAR        PROC            NEAR                        ; (AL)=displaying char.	 
                PUSH            BX				
                MOV             BX, 0
                MOV             AH, 0EH                     ; display function
                INT             10H                         ; call video routine
                POP             BX
                RET
DISPCHAR        ENDP
; ---------------------------------------------------------------------------------------------
CODE            ENDS                                        ; end of code segment
                END             START