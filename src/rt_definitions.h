/* ================================================================
   rt_definitions.h - compile-time configuration for the RT module.

   For PLUTO-coupled builds, this is separate from PLUTO's
   definitions.h. Tests override this file by placing their own
   rt_definitions.h in the test directory (-I test/<name>/ -I src/).

   Physical setup (density, temperature, velocity, sources) is
   defined in user_setup.cpp, not here.
   ================================================================ */

#ifndef LYART_RT_DEFINITIONS_H
#define LYART_RT_DEFINITIONS_H

#ifndef TRUE
#define TRUE   1
#endif
#ifndef FALSE
#define FALSE  0
#endif

/* enum-like constants for compile-time flags */
#define SMITH2015       1
#define HUMLICEK1982    2
#define TASITSIOMI2006  3

#define DIPOLE          1
#define ISOTROPIC       2

#define FULL_BOX        1
#define SLAB            2

#define FDV             1
#define DIRECT          2

/* ---- Physics ---- */

/* Couples RT force/energy with Hydro */
#define COUPLE_LYA_RT   TRUE

/* Phase function for scattering angle.
     DIPOLE     - RASCAS-style dipole (default)
     ISOTROPIC  - isotropic scattering                              */
#define PHASE_FUNCTION   DIPOLE

/* Voigt profile approximation.
     SMITH2015      - continued-fraction (default)
     HUMLICEK1982   - rational approximation, accurate in wings
     TASITSIOMI2006 - analytic fitting                              */
#define VOIGT_FUNCTION   SMITH2015

/* How heating rates are passed to PLUTO.
     FDV    - RT passes momentum only; PLUTO computes F dot v
     DIRECT - RT tracks h*delta_nu per scatter, passes heating rate */
#define ENERGY_DEPOSIT   FDV

/* Turns scatter-recoil on or off.                                  */
#define RECOIL   TRUE

/* assume fully neutral hydrogen (mu = 1).
     TRUE  - density_fn returns n_HI directly, T = p * m_p / (rho * k)
     FALSE - density_fn returns n_H, x_HI(T) computed from table,
             T solved iteratively with mu(x_HI)                      */
#define FULLY_NEUTRAL   FALSE

/* Turns core-skipping (acceleration scheme) on or off.
     TRUE - scatterer velocities are biased to minimize scatter counts
            using the algorithm used by COLT, RASCAS, etc. 
            INCOMPATIBLE WITH COUPLE_LYA_RT.
     FALSE - scatters are treated normally. Use FALSE if you 
             want accurate momentum deposition.  */
#define CORE_SKIPPING    FALSE

/* Turns destruction mechanisms on or off. 
     TRUE - 2p-2s, H2, dust destruction enabled. 
            (NOT IMPLEMENTED YET!) 
     FALSE - Lya photons never destroyed. */
#define DESTRUCTION      FALSE

/* ---- Domain & grid ---- */

/* Boundary condition geometry.
     FULL_BOX - all boundaries are escape boundaries
     SLAB     - periodic in x/y, escape in z                       */
#define RTGEOMETRY       FULL_BOX

#define NX               256       /* grid cells in x */
#define NY               256       /* grid cells in y */
#define NZ               256       /* grid cells in z */

#define LX               6.2e18    /* domain size x [cm] */
#define LY               6.2e18    /* domain size y [cm] */
#define LZ               6.2e18    /* domain size z [cm] */

/* ---- Photons ---- */

/* Photon emission control.
   When DT_PHOTONS > 0: the actual number of packets emitted scales as
     n_emit = max(1, N_PHOTONS * dt / DT_PHOTONS)
   so that packet weight stays constant across varying timesteps.
   When DT_PHOTONS <= 0: scaling is disabled and exactly N_PHOTONS
   packets are emitted every timestep.                               */
#define N_PHOTONS        100      /* reference packet count */
#define DT_PHOTONS       1e8      /* reference timestep [s]; set <= 0 to disable scaling */

/* ---- Output & diagnostics ---- */

#define VERBOSE_OUTPUT       FALSE  /* more detailed MC progress output */
#define SPECTRUM_OUTPUT      TRUE   /* write escaped frequencies */
#define MOMENTUM_OUTPUT      FALSE  /* write momentum deposited for each RT cell */
#define POSITION_OUTPUT      FALSE  /* write photon position snapshots */
#define POSITION_INTERVAL    500    /* write position every 500th scatter */
#define SMOOTHING            FALSE  /* momentum smoothing (experimental) */

/* ---- Build ---- */

/* this flag is not needed when coupled - use the PLUTO parallel flag */
/* REMOVE the #define statement below for serial builds */
#ifndef PARALLEL
#define PARALLEL                     /* multi-node MPI support */
#endif
#define OMP_NUM_THREADS      2

#endif /* LYART_RT_DEFINITIONS_H */
