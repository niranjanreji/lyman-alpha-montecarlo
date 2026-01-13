#ifndef PLUTO_INTERFACE_H
#define PLUTO_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void RadiativeTransfer(
    double dt,
    const double* rho,
    const double* vr,
    const double* r,
    const double* pr,
    const double* vol,
    int n,
    double* out_force);

#ifdef __cplusplus
}
#endif

/* Global radiative transfer force array - set by SplitSource, used by RightHandSide */
extern double *la_radForce;
extern int     la_radForceSize;

#endif
