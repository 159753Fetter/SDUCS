#include "syscall.h"
int
main()
{
    SpaceId newProc;
    OpenFileId input = ConsoleInput;
    OpenFileId output = ConsoleOutput;
    char prompt[7], ch, buffer[60];
    char Hbuffer[80];
    char Cbuffer[80];
    int i,j,k,h,m; 
    char c;
 
    prompt[0] = 'N';
    prompt[1] = 'a';
    prompt[2] = 'c';
    prompt[3] = 'h';
    prompt[4] = 'o';
    prompt[5] = 's';
    prompt[6] = '$';
    prompt[7] = ':';
 
    while( 1 )
    {
        Write(prompt, 8, output);
        i = 0; 
        k = 0;
        h = 0;
        m = 0;
        do { 
            Read(&c, 1, input);
            buffer[i] = c;
            Hbuffer[h++] = buffer[i]; 
            k++;
        } while( buffer[i++] != '\n' );
        Hbuffer[--h] = '\0'; 
        for (m=0;j < 80;j++)
        {
            Cbuffer[m] = buffer[m];
        }
        buffer[--i] = '.';
        buffer[i++] = '.';
        buffer[i++] = 'n';
        buffer[i++] = 'o';
        buffer[i++] = 'f';
        buffer[i++] = 'f';
        buffer[i] = '\0';
 
        if (k == 1)
            continue;
 
        if( h > 0 ) {
            newProc = Exec(buffer);
 
            if (newProc == -1)
                //Write("Invalid Command, Enter again.", 30, output);
                Write("Invalid Command, Enter again, or try \"help\"\n", 44, output);
            else if (newProc != 127)
            {
                Join(newProc);
                Write("Command \"", 9, output);
                Write(Hbuffer,k, output);
                Write("\" Execute Completely.\n", 22, output); 
            } // if (newProc == -1)
        } //if( i > 0 ) {
        i = 0;
    } // while( 1 )
}