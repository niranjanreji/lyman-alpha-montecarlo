#include "pluto.h"
#include "src/pluto_interface.h"
#ifdef PARALLEL
    #include <mpi.h>
#endif

/* global quantities for lya radiative transfer force */
double *g_lya_radForce = NULL;
int     g_lya_radForceSize = 0;
double *g_lya_radEnergy = NULL;
int     g_lya_radEnergySize = 0;
double *g_lya_photonCount = NULL;
int     g_lya_photonCountSize = 0;
double *g_lya_photonEnergy = NULL;
int     g_lya_photonEnergySize = 0;
double  g_lya_dt_force = 1.e38;
double  g_lya_cum_mass_flux = 0.0;
double  g_lya_cum_energy_flux = 0.0;
double  g_lya_cum_momentum_flux = 0.0;
/* g_lya_cum_photon_energy_{injected,escaped} defined in monte_carlo.cpp */

/* **************************************************************************  */
void SplitSource (const Data *d, double dt, Time_Step *Dts, Grid *grid)
/*
 *
 * PURPOSE
 *
 *   Main driver for handling source terms as a separate
 *   step using operator splitting.
 *
 *   Source terms may be:
 *
 *    - optically thin radiative losses (cooling)
 *    - Diffusion operators: 
 *       * resistivity 
 *       * Thermal conduction
 *       * Viscosity
 *
 *
 *
 ***************************************************************************** */
{
    int i, j, k, nv;
    static real **v;
    real t_save;

    /* Get unit conversions from PLUTO 4.0 parameters (CGS units) */

    double unit_length   = g_unitLength;
    double unit_density  = g_unitDensity;
    double unit_velocity = g_unitVelocity;

    /* LYMAN ALPHA RADIATIVE TRANSFER ===========
    * PREP + CALL (GATHER PLUTO FIELDS AND DATA)
    * SET PLUTO TIMESTEP BASED ON FORCE ======== */

    int n_glob = grid[IDIR].np_tot_glob;
    int nghost = grid[IDIR].nghost;

    static LyaData lya = {0};
    static double *global_rho = NULL;
    static double *global_vr  = NULL;
    static double *global_pr  = NULL;
    static double *global_dV  = NULL;

    /* Allocate global arrays (on all ranks for simplicity) */
    if (lya.rho == NULL) {
        lya.rho       = malloc(n_glob * sizeof(double));
        lya.vr        = malloc(n_glob * sizeof(double));
        lya.pr        = malloc(n_glob * sizeof(double));
        lya.r         = malloc(n_glob * sizeof(double));
        lya.dV        = malloc(n_glob * sizeof(double));
        lya.out_force = malloc(n_glob * sizeof(double));
        lya.out_energy = malloc(n_glob * sizeof(double));
        lya.photon_count = malloc(n_glob * sizeof(double));
        lya.photon_energy = malloc(n_glob * sizeof(double));
        global_rho    = malloc(n_glob * sizeof(double));
        global_vr     = malloc(n_glob * sizeof(double));
        global_pr     = malloc(n_glob * sizeof(double));
        global_dV     = malloc(n_glob * sizeof(double));
    }

    lya.n      = n_glob;
    lya.dt     = dt * (unit_length / unit_velocity);
    lya.nghost = nghost;
    lya.gamma_gas = g_gamma;

    /* Fill radial positions from global array (available on all ranks) */
    for (i = 0; i < n_glob; i++) {
        lya.r[i] = grid[IDIR].x_glob[i] * unit_length;
    }

    #ifdef PARALLEL
        /* --- MPI: Gather local Vc data to rank 0 --- */
        int local_n = grid[IDIR].np_tot; /* local total including ghosts */

        /* prepare local data (full local array including ghosts) */
        double *local_rho = malloc(local_n * sizeof(double));
        double *local_vr  = malloc(local_n * sizeof(double));
        double *local_pr  = malloc(local_n * sizeof(double));
        double *local_dV  = malloc(local_n * sizeof(double));

        for (i = 0; i < local_n; ++i) {
            local_rho[i] = d->Vc[RHO][0][0][i];
            local_vr[i]  = d->Vc[VX1][0][0][i];
            local_pr[i]  = d->Vc[PRS][0][0][i];
            local_dV[i]  = grid[IDIR].dV[i];
        }

        /* get counts and displacements for MPI_Gatherv */
        int nprocs, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int *recvcounts = malloc(nprocs * sizeof(int));  /* to store how many non-ghost cells each rank owns */
        int *displs     = malloc(nprocs * sizeof(int));  /* displs[i] = ptr to rank i's cells in global array */

        /* each rank sends its interior (non-ghost) cell count */
        int local_int = grid[IDIR].np_int;
        MPI_Allgather(&local_int, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

        /* compute displacements (offset by nghost for ghost cells at start) */
        displs[0] = nghost;
        for (i = 1; i < nprocs; i++) {
            /* each subsequent ranks data = index where i-1
             * starts + number of non-ghost cells in i-1 */
            displs[i] = displs[i-1] + recvcounts[i-1];
        }

        /* gather interior cells only (indices IBEG to IEND in local array) 
         * displs are absolute offsets into global_rho 
         * places each rank's data at indices specified by 
         * displs with recvcounts elements each */
        MPI_Gatherv(local_rho + nghost, local_int, MPI_DOUBLE,
                    global_rho, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_vr + nghost, local_int, MPI_DOUBLE,
                    global_vr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_pr + nghost, local_int, MPI_DOUBLE,
                    global_pr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_dV + nghost, local_int, MPI_DOUBLE,
                    global_dV, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* the boundary ghost cells are ignored by pluto_interface.cpp, no change needed 
         * now broadcast global arrays to all ranks, since Gatherv only happens on rank 0 */

        MPI_Bcast(global_rho, n_glob, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(global_vr,  n_glob, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(global_pr,  n_glob, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(global_dV,  n_glob, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* convert to CGS and fill lya struct */
        for (i = 0; i < n_glob; i++) {
            lya.rho[i] = global_rho[i] * unit_density;
            lya.vr[i]  = global_vr[i]  * unit_velocity;
            lya.pr[i]  = global_pr[i]  * unit_density * unit_velocity * unit_velocity;
            lya.dV[i]  = global_dV[i]  * unit_length * unit_length * unit_length;
        }

        /* all ranks call RT */
        LyaRadiativeTransfer(&lya);

        free(local_rho);
        free(local_vr);
        free(local_pr);
        free(local_dV);
        free(recvcounts);
        free(displs);

        /* change force, energy deposition density to PLUTO units */
        double f_unit = unit_density * unit_velocity * unit_velocity / unit_length;
        double e_unit = unit_density * unit_velocity * unit_velocity;

        for (i = 0; i < n_glob; i++) {
            lya.out_force[i] /= f_unit;
            lya.out_energy[i] /= e_unit;
        }

        MPI_Allreduce(MPI_IN_PLACE, lya.out_force,    n_glob, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, lya.out_energy,    n_glob, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, lya.photon_count,  n_glob, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, lya.photon_energy, n_glob, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &lya.num_scatters, 1,   MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

        g_lya_radForce = lya.out_force;
        g_lya_radForceSize = n_glob;
        g_lya_radEnergy = lya.out_energy;
        g_lya_radEnergySize = n_glob;
        g_lya_photonCount = lya.photon_count;
        g_lya_photonCountSize = n_glob;
        g_lya_photonEnergy = lya.photon_energy;
        g_lya_photonEnergySize = n_glob;
    #else
        for (i = 0; i < n_glob; ++i) {
            lya.rho[i] = d->Vc[RHO][0][0][i] * unit_density;
            lya.vr[i]  = d->Vc[VX1][0][0][i] * unit_velocity;
            lya.pr[i]  = d->Vc[PRS][0][0][i] * unit_density * unit_velocity * unit_velocity;
            lya.dV[i]  = grid[IDIR].dV[i] * 4 * CONST_PI * unit_length * unit_length * unit_length;
        }

        LyaRadiativeTransfer(&lya);

        /* change force, energy deposition density to PLUTO units */
        double f_unit = unit_density * unit_velocity * unit_velocity / unit_length;
        double e_unit = unit_density * unit_velocity * unit_velocity;

        for (i = 0; i < n_glob; i++) {
            lya.out_force[i] /= f_unit;
            lya.out_energy[i] /= e_unit;
        }

        g_lya_radForce = lya.out_force;
        g_lya_radForceSize = n_glob;
        g_lya_radEnergy = lya.out_energy;
        g_lya_radEnergySize = n_glob;
        g_lya_photonCount = lya.photon_count;
        g_lya_photonCountSize = n_glob;
        g_lya_photonEnergy = lya.photon_energy;
        g_lya_photonEnergySize = n_glob;
    #endif

    /* Force-based timestep constraint:
       The Lya cross section in the Doppler core goes as sigma(x) ~ exp(-x^2),
       where x = (nu - nu_0) / delta_nu_D is the dimensionless frequency.
       A bulk velocity change dv shifts x by dx = dv / v_th.
       We require that the cross section not change by more than a factor
       tau_drop in one step:
         exp(-(dv/v_th)^2) >= 1/tau_drop
         => dv_max = v_th * sqrt(ln(tau_drop))
       With tau_drop = 1e4, dv_max ~ 3 * v_th.
       The timestep is then dt = dv_max / a, where a = F/rho is the
       radiation force acceleration. */

    double dv_over_vth_max = sqrt(log(1.e4)); /* larger value - more inaccurate coupling */
    double dt_force = 1.e38;
    if (lya.num_scatters > 100) {
        for (i = IBEG; i <= IEND; i++) {
            double force = fabs(g_lya_radForce[i]);
            if (force < 1.e-30) continue;

            double rho = MAX(d->Vc[RHO][0][0][i], g_smallDensity);
            double prs = MAX(d->Vc[PRS][0][0][i], g_smallPressure);
            double dR  = grid[IDIR].dx[i];

            /* thermal speed proxy in code units: v_th^2 - 2*p/rho */
            double v_th   = sqrt(2.0 * prs / rho);
            double dv_max = MAX(v_th * dv_over_vth_max, 1.e-50);

            /* acceleration */
            double a = MAX(force / rho, 1.e-50);
            double v_over_a = dv_max / a;
            /* displacement condition: dt from quadratic x = v*t + 0.5*a*t^2 = CFL*dR */
            dt_force  = (-v_over_a + sqrt(v_over_a * v_over_a + 2.0 * Dts->cfl * dR / a)) / 2.0;
        }
    }

    g_lya_dt_force = dt_force;

    /*  ---- GLM source term treated in main ----  */

    /*
    #ifdef GLM_MHD
    GLM_SOURCE (d->Vc, dt, grid);
    #endif
    */

    /*  ---------------------------------------------
                Cooling/Heating losses
        ---------------------------------------------  */

    #if COOLING != NO
    #if COOLING == POWER_LAW  /* -- solve exactly -- */
        PowerLawCooling (d->Vc, dt, Dts, grid);
    #else
        CoolingSource (d, dt, Dts, grid);
    #endif
    #endif

    /* ----------------------------------------------
        Parabolic terms using STS:

        - resistivity 
        - thermal conduction
        - viscosity 
    ---------------------------------------------- */

    #if (PARABOLIC_FLUX & SUPER_TIME_STEPPING)
        STS (d, Dts, grid);
    #endif

    #if (PARABOLIC_FLUX & RK_CHEBYSHEV)
        RKC (d, Dts, grid);
    #endif
}