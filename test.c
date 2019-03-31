#include<stdio.h>
#include<stdlib.h>

int main()
{
    int array[6][6];
    int rows,columns;
    int random,i;
    int randvalues[36],m=0;
    int t,j;


    for(i=0;i<36;i++)     //assigning values 1 to 36 
         randvalues[i]=i+1;

    for(i=0;i<36;i++)      //shuffle logic
    {
         j=i+rand()/(RAND_MAX/(36-i) + 1);
         t=randvalues[j];
         randvalues[j] = randvalues[i];
         randvalues[i] = t;
    }

    for(rows=0;rows<6;rows++) //conversion from 1-D to 2-D array and printning
    {
        for(columns=0;columns<6;columns++)
        {
            array[rows][columns] = randvalues[m++];
            printf("%d " , array[rows][columns]);
        }
        printf("\n");
    }
    return 0;
}