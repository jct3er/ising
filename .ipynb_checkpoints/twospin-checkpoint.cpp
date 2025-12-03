/* 
******************************************************************************
* twospin.c: Uses Metropolis algorithm to generate thermal ensemble for a    *
*	     two coupled spins in a magnetic field h..                	     *
*  								             *
******************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
 


/* update_spin (*s, env)
** -----------
** Do a metropolis update on a spin *s whose "environment" is env.
** The environment should be set beforehand so that the dependence
** of the total beta*energy on the value of the selected spin is
** given by
**               (selected spin)*env .
**
*/
// update state by flipping spin, accept new state if E_new<E_old
// or rand[0,0]< P(new)/P(old)
void update_spin(int *s, double env) {
  int spin = *s;
  int newspin = ( drand48() < 0.5 ? 1 : -1 );
  double DeltaBetaE = -(newspin-spin)*env;       /* beta*(E(new) - E(old)) */
  if ( DeltaBetaE <= 0 || drand48() < exp(-DeltaBetaE) ) *s = newspin;   // Metropolis method
}

/*
** sweep (s1, s2, beta, h)
** -----
** Sweep once through both spins, trying an update for each with inverse
** temperature beta and external magnetic field parameter h.
*/
// sweep through spins of each particle
// here the "environment" for each particle is given by the
// sign of its neighbor + the external field and temperature parameter (h = \beta H)
void sweep(int *s1, int *s2, double beta, double h) {
  int spin2 = *s2;
  update_spin(s1, beta*spin2 + h);     // update first spin
  int spin1 = *s1;
  update_spin(s2, beta*spin1 + h);     // update second spin 
}



int main(int argc, char *argv[]) {
   int nsweep;
   int spin1, spin2;
   double beta, h;
   int nupup=0, nupdown=0, ndownup=0, ndowndown=0;
   int nmag=0, ncorr=0, ntotal=0;

   printf("Program generates a thermal ensemble for two coupled spins.\n\n");

   srand48((long)time(0));               // seed the number generator

   if (argc==4){
    nsweep=atoi(argv[1]);
    h=atof(argv[2]);
    beta=atof(argv[3]);
   }
   else{
     printf("Enter total number of spin configurations (sweeps) generated:\n");
     scanf("%d", &nsweep); 

     printf("Enter value of magnetic field parameter h=H/(k_bT):\n");
     scanf("%lf", &h);

     printf("Enter temperature parameter beta (= 1/kT):\n");
     scanf("%lf", &beta);
   }
   
   //  Initialize spins with a "hot" start
   spin1 = spin2 = -1;
   if(drand48()<0.5) spin1 = 1; 
   if(drand48()<0.5) spin2 = 1;

   for(int n=0; n<nsweep; n++) {
     sweep(&spin1, &spin2, beta, h);
 
     // accumulate magnetization 
     nmag += spin1; ntotal++;
     nmag += spin2; ntotal++;

     // count number of times each state occurs
     if (spin1== 1 && spin2== 1) nupup++;
     if (spin1== 1 && spin2==-1) nupdown++;
     if (spin1==-1 && spin2== 1) ndownup++;
     if (spin1==-1 && spin2==-1) ndowndown++;
  
     // compute spin-spin correlation function
     ncorr += spin1*spin2;
   }
  
   printf("\n"); 
   printf("State Probabilities:\n");
   printf("P(++) = %lf\tP(+-) = %lf\nP(-+) = %lf\tP(--) = %lf\n",
	  (double)nupup/nsweep, (double)nupdown/nsweep,
	  (double)ndownup/nsweep, (double)ndowndown/nsweep);
   printf("\n"); 
   printf("Magnetization: <s> = %lf\n", (double)nmag/(2*nsweep));
   printf("Spin-Spin Correlation Function:\n");
   printf("<s1 s2> = %lf\n", (double)ncorr/nsweep);

   return(0);
}
