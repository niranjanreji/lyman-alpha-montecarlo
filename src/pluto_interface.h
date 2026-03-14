/* pluto_interface.h — C-linkage interface between PLUTO and the
 * lyman-alpha RT module. included by both C (split_source.c, rhs.c)
 * and C++ (pluto_interface.cpp) translation units.
 *
 * Niranjan Reji, Raman Research Institute, March 2026
 * assisted by Claude (Anthropic) */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;                   /* number of PLUTO cells (NX1_TOT) */
    int nghost;              /* number of ghost cells per side (PLUTO grid->nghost) */
    double dt;               /* hydrodynamics (and RT) time-step */
    double* rho;             /* hydro grid density */
    double* vr;              /* hydro grid velocity */
    double* r;               /* hydro grid cell centers */
    double* pr;              /* hydro grid pressure */
    double  gamma_gas;       /* adiabatic index for thermal energy conversion */
    double* dV;              /* hydro grid cell volumes (CGS) */
    double* out_force;       /* OUTPUT: force / volume */
    double* out_energy;      /* OUTPUT: energy rate / volume */
    double* photon_count;    /* OUTPUT: num of photons in each cell */
    double* photon_energy;   /* OUTPUT: sum(weight * h * nu / c) in each cell */
    long long num_scatters;  /* OUTPUT: total number of scatters */
} LyaData;

void LyaRadiativeTransfer(LyaData* lya);
double LyaTemperatureFromHydro(double rho, double pr);
double LyaNeutralFractionFromTemperature(double T);
double LyaNeutralHydrogenNumberDensity(double rho, double pr);


/* arrays set by SplitSource.c, used in rhs.c 
 * or used in diagnostics in init.c:Analysis() */
extern double *g_lya_radForce;
extern int     g_lya_radForceSize;
extern double *g_lya_radEnergy;
extern int     g_lya_radEnergySize;
extern double *g_lya_photonCount;
extern int     g_lya_photonCountSize;
extern double *g_lya_photonEnergy;
extern int     g_lya_photonEnergySize;

extern double g_lya_dt_force;
extern double g_lya_cum_mass_flux;
extern double g_lya_cum_energy_flux;
extern double g_lya_cum_momentum_flux;
extern double g_lya_cum_photon_energy_injected;
extern double g_lya_cum_photon_energy_escaped;

#ifdef __cplusplus
}
#endif

