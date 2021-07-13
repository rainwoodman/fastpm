#include <stdio.h>
#include <string.h>
#include <alloca.h>
#include <mpi.h>
#include <math.h>

#include <fastpm/libfastpm.h>
#include <fastpm/logging.h>
#include <fastpm/string.h>
#include <fastpm/rfof.h>
#include <fastpm/io.h>

int main(int argc, char * argv[]) {

    MPI_Init(&argc, &argv);

    libfastpm_init();

    MPI_Comm comm = MPI_COMM_WORLD;

    fastpm_set_msg_handler(fastpm_default_msg_handler, comm, NULL);

    FastPMConfig * config = & (FastPMConfig) {
        .nc = 64,
        .boxsize = 0.3 * 64,
        .alloc_factor = 10.0,
        .cosmology = NULL,
        .vpminit = (VPMInit[]) {
            {.a_start = 0, .pm_nc_factor = 2},
            {.a_start = -1, .pm_nc_factor = 0},
        },
        .FORCE_TYPE = FASTPM_FORCE_FASTPM,
        .nLPT = 2.5,
    };

    FastPMSolver solver[1];
    fastpm_solver_init(solver, config, comm);

    FastPMFloat * rho_init_ktruth = pm_alloc(solver->basepm);
    FastPMFloat * rho_final_ktruth = pm_alloc(solver->basepm);
    FastPMFloat * rho_final_xtruth = pm_alloc(solver->basepm);

    /* First establish the truth by 2lpt -- this will be replaced with PM. */
    struct fastpm_powerspec_eh_params eh = {
        .Norm = 2e7, /* FIXME: this is not any particular sigma8. */
        .hubble_param = 0.7,
        .omegam = 0.260,
        .omegab = 0.044,
    };
    fastpm_ic_fill_gaussiank(solver->basepm, rho_init_ktruth, 2004, FASTPM_DELTAK_GADGET);
    fastpm_ic_induce_correlation(solver->basepm, rho_init_ktruth, (fastpm_fkfunc)fastpm_utils_powerspec_eh, &eh);

    double time_step[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    fastpm_solver_setup_lpt(solver, FASTPM_SPECIES_CDM, rho_init_ktruth, 0.1);

    fastpm_solver_evolve(solver, time_step, sizeof(time_step) / sizeof(time_step[0]));


    // Parameters fitted for Illustris under 0.3 Mpc/h particle sep, 40 steps.
    // l1 = 0.25, A1 = 0.012, l6=0.24, A2=0.06, B1=7.02, B2=6.025.
    FastPMRFOFFinder rfof = {
        .nmin = 8,
        .kdtree_thresh = 1,
        .linkinglength = 0.2 * 0.3,
        .l1 = 0.2 * 0.3, //0.25 * 0.3,
        .l6 = 0.2 * 0.3, 0.24 * 0.3,
        .A1 = 0, // 0.012 * 0.3,
        .A2 = 0, //0.06 * 0.3,
        .B1 = 1000, // 7.02,
        .B2 = 0, // 6.025,
        .periodic = 1,
    };

    FastPMStore * p = fastpm_solver_get_species(solver, FASTPM_SPECIES_CDM);
    fastpm_rfof_init(&rfof, solver->cosmology, p, solver->basepm);

    FastPMStore halos[1];
    fastpm_store_set_name(halos, "RFOFHalos");
    fastpm_rfof_execute(&rfof, halos, NULL, 0.0);

    char * snapshot = fastpm_strdup_printf("rfof-%d", solver->NTask);
    fastpm_sort_snapshot(halos, solver->comm, FastPMSnapshotSortByLength, 0);
    fastpm_store_write(halos, snapshot, "w", 1, solver->comm);

    int task;
    int ntask;
    MPI_Comm_size(MPI_COMM_WORLD, &ntask);
    MPI_Comm_rank(MPI_COMM_WORLD, &task);
    int j;
    for(j = 0; j < ntask; j ++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(j != task) continue;
        int i;
        for(i = 0; i < 10 && i < halos->np; i ++) {
            fastpm_ilog(INFO, "Length of halo %d: %d\n", i, halos->length[i]);
        }
    }

    fastpm_store_destroy(halos);
    fastpm_rfof_destroy(&rfof);

    pm_free(solver->basepm, rho_final_xtruth);
    pm_free(solver->basepm, rho_final_ktruth);
    pm_free(solver->basepm, rho_init_ktruth);

    fastpm_solver_destroy(solver);
    libfastpm_cleanup();
    MPI_Finalize();
    return 0;
}

