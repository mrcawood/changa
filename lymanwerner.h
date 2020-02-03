/// @file lymanwerner.h
/// Declarations for Lyman Werner calculation.
#include <math.h>

/* inline function used in feedback.C and starform.C to calculate the Lyman Werner luminosity for a star particle of a given age and mass*/

typedef struct lwDataStruct {
    float **lwLuminosity;
} LWDATA;

LWDATA *LymanWernerTableInit( );
void LymanWernerTableFinalize(LWDATA *lwd);

double calcLogStochLymanWerner(double dAgelog, SFEvent *sfEvent, LWDATA *LWData);

inline double calcLogSSPLymanWerner(double dAgelog, double dMassLog)
{
 /* Variables below are to a polynomial fit for data representing a SSP generated by Starburst99*/    
    double a0 = -84550.812,
      a1 =  54346.066,
      a2 = -13934.144,
      a3 =  1782.1741,
      a4 = -113.68717,
      a5 =  2.8930795;
    return a0
      + a1*dAgelog
      + a2*dAgelog*dAgelog
      + a3*dAgelog*dAgelog*dAgelog
      + a4*dAgelog*dAgelog*dAgelog*dAgelog
      + a5*dAgelog*dAgelog*dAgelog*dAgelog*dAgelog + dMassLog;
}

inline double calcLogMax8LymanWerner(double dAgelog, double loglownorm)
{
/* The SSP portion is a polynomial fit for data generated by Starburst99 (t>10^6.5 yr), with a maximum stellar mass
 * of 8 Msol. The high mass portion was generated from a series of Starburst99 "SSPs" consisting of
 * only a single mass. A lookup table is used to estimate the LW feedback for these stars
 *
 * Recall, lownorm (dLowNorm) is the normalization for the continuous part of the stochastic IMF, or the mass
 * the whole star particle would have if scaled by the same amount as the low mass portion*/

    double a0 = -39793.789,
      a1 = 26397.519,
      a2 = -6970.825,
      a3 = 916.934,
      a4 = -60.068,
      a5 = 1.567;
    double lowLW = a0
      + a1*dAgelog
      + a2*dAgelog*dAgelog
      + a3*dAgelog*dAgelog*dAgelog
      + a4*dAgelog*dAgelog*dAgelog*dAgelog
      + a5*dAgelog*dAgelog*dAgelog*dAgelog*dAgelog + loglownorm;
}
