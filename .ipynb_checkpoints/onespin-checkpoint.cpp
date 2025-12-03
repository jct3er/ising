/* 
********************************************************************************
* onespin.c: Uses Metropolis algorithm to generate thermal "ensemble" for a    *
*	     single spin.                			               *
*  								               *
********************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
 
// update state by flipping spin, accept new state if E_new<E_old
// or rand[0,1)< P(new)/P(old)
// s: is the inputs spin
// h: is the environment (= \beta H)
void update(int *s, double h) {
  int spin = *s;
  int newspin = -spin;                         // trial spin flip
  double DeltaBetaE = -(newspin-spin)*h;       // beta*(E(new) - E(old))
  // if the new state is at lower energy, accept it
  // if not, accept it with a probability decreasing w/ exp(-DeltaBetaE)
  if ( DeltaBetaE <= 0 || drand48() < exp(-DeltaBetaE) ) *s = newspin;  // Metropolis method
}

int main(int argc, char *argv[]) {
  int nsweep;
  double h;

  printf("Program generates a thermal ensemble for states with one spin.\n\n");

  srand48((long)(time(0)));

  if (argc==3){
    nsweep=atoi(argv[1]);
    h=atof(argv[2]);
  }
  else{
    printf("Enter total number of spin configurations (sweeps) generated:\n");
    scanf("%d", &nsweep); 

    // h > 1 means that effect of H field is larger that temperature effects
    printf("Enter value of magnetic field parameter h=H/(k_bT):\n");
    scanf("%lf", &h);
  }
  
  // our initial state (arbitrary value, could be -1 as well)
  int spin = 1;

  // Metropolis update loop
  int nplus=0;
  int nminus=0;
  for(int n=0; n<nsweep; n++) {
    update(&spin,h);           // try a Metropolis update
    if (spin == 1) nplus++;                        
    if (spin == -1) nminus++;
  }

  printf("Calculations %d %d\n",nplus,nminus);
  printf("P(+ state) = %13.10lf\n", (double)nplus/(nplus+nminus) );
  printf("P(- state) = %13.10lf\n", (double)nminus/(nplus+nminus) );

  // Write <magnetization> 
  printf("<sigma> = %13.10lf\n", (double)(nplus-nminus)/(nplus+nminus) );
  
  /* Write theoretical prediction for comparison  */
  printf("\nTheory prediction: sigma = tanh(h) = %12.10lf\n", tanh(h));   
  
  return(0);
}
