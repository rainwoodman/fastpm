#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>
#include <alloca.h>
#include <mpi.h>
#include <math.h>
#include <signal.h>
#include <limits.h>

#include <fastpm/libfastpm.h>
#include <fastpm/prof.h>
#include <fastpm/logging.h>
#include <fastpm/string.h>
#include <fastpm/lightcone.h>
#include <fastpm/constrainedgaussian.h>
#include <fastpm/io.h>
#include <fastpm/fof.h>
#include <bigfile-mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "lua-config.h"
#include "param.h"

/* c99 has no pi. */
#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif

typedef struct {
    CLIParameters * cli;
    LUAParameters * lua;
    int iout; /* index of next unwritten snapshot. */
} RunData;


extern void
init_stacktrace();

static void
_memory_peak_handler(FastPMMemory * mem, void * userdata);

static int 
take_a_snapshot(FastPMSolver * fastpm, RunData * prr);

struct usmesh_ready_handler_data {
    FastPMSolver * fastpm;
    RunData * prr;
    FastPMStore tail[1];
    int64_t * hist;
    int64_t * hist_fof;
    double * aedges;
    int Nedges;
};

static void
usmesh_ready_handler(FastPMUSMesh * mesh, FastPMLCEvent * lcevent, struct usmesh_ready_handler_data * userdata);

int 
read_runpb_ic(FastPMSolver * fastpm, FastPMStore * p, const char * filename);

void 
read_grafic_gaussian(PM * pm, FastPMFloat * g_x, const char * filename);

int
write_runpb_snapshot(FastPMSolver * fastpm, FastPMStore * p, const char * filebase);

int
read_complex(PM * pm, FastPMFloat * data, const char * filename, const char * blockname, int Nwriters);

int
read_powerspectrum(FastPMPowerSpectrum *ps, const char filename[], const double sigma8, MPI_Comm comm);

static void
run_fof(FastPMSolver * fastpm, FastPMStore * snapshot, FastPMStore * halos, RunData * prr);

static void
run_usmesh_fof(FastPMSolver * fastpm,
        FastPMLCEvent * lcevent,
        FastPMStore * halos,
        RunData * prr,
        FastPMStore * tail,
        FastPMLightCone * lc);

static void
write_parameters(const char * filebase, const char * dataset, RunData * prr, MPI_Comm comm)
{
    BigFile bf;
    BigBlock bb;
    if(0 != big_file_mpi_open(&bf, filebase, comm)) {
        fastpm_raise(-1, "Failed to open the file: %s\n", big_file_get_error_message());
    }

    if(0 != big_file_mpi_open_block(&bf, &bb, dataset, comm)) {
        fastpm_raise(-1, "Failed to open the dataset : %s\n", big_file_get_error_message());
    }

    big_block_set_attr(&bb, "ParamFile", prr->lua->string, "S1", strlen(prr->lua->string) + 1);
    double ParticleFraction = CONF(prr->lua, particle_fraction);
    big_block_set_attr(&bb, "ParticleFraction", &ParticleFraction, "f8", 1);

    big_block_mpi_close(&bb, comm);
    big_file_mpi_close(&bf, comm);

}

static void
prepare_cosmology(FastPMCosmology * c, RunData * prr);

int run_fastpm(FastPMConfig * config, RunData * prr, MPI_Comm comm);

int main(int argc, char ** argv) {

    init_stacktrace();

    MPI_Init(&argc, &argv);

    libfastpm_init();

    MPI_Comm comm = MPI_COMM_WORLD; 

    fastpm_set_msg_handler(fastpm_default_msg_handler, comm, NULL);
    fastpm_info("This is FastPM, with libfastpm version %s.\n", LIBFASTPM_VERSION);

    char * error;
    CLIParameters * cli = parse_cli_args_mpi(argc, argv, comm);

#ifdef _OPENMP
    /* because lua lib may read this */
    if(cli->MaxThreads > 0)
        omp_set_num_threads(cli->MaxThreads);
#endif

    LUAParameters * lua = parse_config_mpi(cli->argv[0], cli->argc, cli->argv, &error, comm);

    RunData prr[1] = {{cli, lua, 0}};

    if(prr->lua) {
        fastpm_info("Configuration %s\n", prr->lua->string);
    } else {
        fastpm_info("Parsing configuration failed with error: %s\n", error);
        MPI_Finalize();
        exit(1);
    }

    libfastpm_set_memory_bound(prr->cli->MemoryPerRank * 1024 * 1024);
    fastpm_memory_set_handlers(_libfastpm_get_gmem(), NULL, _memory_peak_handler, &comm);

    /* convert parameter files pm_nc_factor into VPMInit */
    VPMInit * vpminit = NULL;
    if(CONF(prr->lua, ndim_pm_nc_factor) == 0) {
        vpminit = alloca(sizeof(VPMInit) * 2);
        vpminit[0].a_start = 0;
        vpminit[0].pm_nc_factor = CONF(prr->lua, pm_nc_factor)[0];
        vpminit[1].a_start = 1;
        vpminit[1].pm_nc_factor = 0;
    } else
    if(CONF(prr->lua, ndim_pm_nc_factor) == 2) {
        vpminit = alloca(sizeof(VPMInit) * (CONF(prr->lua, shape_pm_nc_factor)[0] + 1));
        int i;
        for(i = 0; i < CONF(prr->lua, n_pm_nc_factor); i ++) {
            vpminit[i].a_start = CONF(prr->lua, pm_nc_factor)[2 * i];
            vpminit[i].pm_nc_factor = CONF(prr->lua, pm_nc_factor)[2 * i + 1];
        }
        /* mark the end */
        vpminit[i].pm_nc_factor = 0;
    } else {
        fastpm_raise(-1, "Unknown format of pm_nc_factor, either a scalar or a 2d array. ");
    }

    fastpm_info("np_alloc_factor = %g\n", CONF(prr->lua, np_alloc_factor));

    FastPMCosmology cosmology[1] = {{0}};

    prepare_cosmology(cosmology, prr);

    FastPMConfig * config = & (FastPMConfig) {
        .nc = CONF(prr->lua, nc),
        .alloc_factor = CONF(prr->lua, np_alloc_factor),
        .lpt_nc_factor = CONF(prr->lua, lpt_nc_factor),
        .vpminit = vpminit,
        .boxsize = CONF(prr->lua, boxsize),
        .cosmology = cosmology,
        .USE_DX1_ONLY = CONF(prr->lua, za),
        .nLPT = -2.5f,
        .USE_SHIFT = CONF(prr->lua, shift),
        .FORCE_TYPE = CONF(prr->lua, force_mode),
        .KERNEL_TYPE = CONF(prr->lua, kernel_type),
        .SOFTENING_TYPE = CONF(prr->lua, force_softening_type),
        .PAINTER_TYPE = CONF(prr->lua, painter_type),
        .painter_support = CONF(prr->lua, painter_support),
        .NprocY = prr->cli->NprocY,
        .UseFFTW = prr->cli->UseFFTW,
        .ExtraAttributes = 0,
        .pgdc = CONF(prr->lua, pgdc),
        .pgdc_alpha0 = CONF(prr->lua, pgdc_alpha0),
        .pgdc_A = CONF(prr->lua, pgdc_A),
        .pgdc_B = CONF(prr->lua, pgdc_B),
        .pgdc_kl = CONF(prr->lua, pgdc_kl),
        .pgdc_ks = CONF(prr->lua, pgdc_ks),
    };

    if(CONF(prr->lua, compute_potential)) {
        config->ExtraAttributes |= COLUMN_POTENTIAL;
    }
    if(CONF(prr->lua, pgdc)) {
        config->ExtraAttributes |= COLUMN_PGDC;
    }

    run_fastpm(config, prr, comm);

    free_lua_parameters(prr->lua);
    free_cli_parameters(prr->cli);

    libfastpm_cleanup();

    MPI_Finalize();

    return 0;
}

static int 
check_snapshots(FastPMSolver * fastpm, FastPMInterpolationEvent * event, RunData * prr);

static int 
check_lightcone(FastPMSolver * fastpm, FastPMInterpolationEvent * event, FastPMUSMesh * lc);

static int 
write_powerspectrum(FastPMSolver * fastpm, FastPMForceEvent * event, RunData * prr);

static int 
report_domain(FastPMSolver * fastpm, FastPMForceEvent * event, RunData * prr);

static int 
report_lpt(FastPMSolver * fastpm, FastPMLPTEvent * event, RunData * prr);

static double * 
prepare_time_step(RunData * prr, double a0, size_t * n_time_step);

static void 
prepare_cdm(FastPMSolver * fastpm, RunData * prr, double a0, MPI_Comm comm);

static void 
prepare_ncdm(FastPMSolver * fastpm, RunData * prr, double a0, MPI_Comm comm);

static void
report_memory(MPI_Comm);

static void
prepare_lc(FastPMSolver * fastpm, RunData * prr,
        FastPMLightCone * lc, FastPMUSMesh ** usmesh);

static int 
print_transition(FastPMSolver * fastpm, FastPMTransitionEvent * event, RunData * prr);

int run_fastpm(FastPMConfig * config, RunData * prr, MPI_Comm comm) {
    FastPMSolver fastpm[1];

    CLOCK(init);
    CLOCK(cdmic);
    CLOCK(ncdmic);
    CLOCK(evolve);
    CLOCK(io);
    CLOCK(sort);
    CLOCK(indexing);

    MPI_Barrier(comm);
    ENTER(init);

    fastpm_solver_init(fastpm, config, comm);

    const double M0 = fastpm->cosmology->Omega_cdm * FASTPM_CRITICAL_DENSITY
                    * pow(CONF(prr->lua, boxsize) / CONF(prr->lua, nc), 3.0);
    fastpm_info("mass of a CDM particle is %g 1e10 Msun/h\n", M0);

    fastpm_info("BaseProcMesh : %d x %d\n",
            pm_nproc(fastpm->basepm)[0], pm_nproc(fastpm->basepm)[1]);

#ifdef _OPENMP
    fastpm_info("%d Threads\n", omp_get_max_threads());
#endif

    LEAVE(init);

    fastpm_add_event_handler(&fastpm->event_handlers,
        FASTPM_EVENT_LPT,
        FASTPM_EVENT_STAGE_AFTER,
        (FastPMEventHandlerFunction) report_lpt,
        prr);

    fastpm_add_event_handler(&fastpm->event_handlers,
        FASTPM_EVENT_FORCE,
        FASTPM_EVENT_STAGE_BEFORE,
        (FastPMEventHandlerFunction) report_domain,
        prr);

    fastpm_add_event_handler(&fastpm->event_handlers,
        FASTPM_EVENT_FORCE,
        FASTPM_EVENT_STAGE_AFTER,
        (FastPMEventHandlerFunction) write_powerspectrum,
        prr);

    fastpm_add_event_handler(&fastpm->event_handlers,
        FASTPM_EVENT_INTERPOLATION,
        FASTPM_EVENT_STAGE_BEFORE,
        (FastPMEventHandlerFunction) check_snapshots,
        prr);

    fastpm_add_event_handler(&fastpm->event_handlers,
        FASTPM_EVENT_TRANSITION,
        FASTPM_EVENT_STAGE_BEFORE,
        (FastPMEventHandlerFunction) print_transition,
        prr);

    /* initialize the lightcone */
    FastPMLightCone lc[1] = {{
        .speedfactor = CONF(prr->lua, dh_factor),
        .cosmology = fastpm->cosmology,
        .fov = CONF(prr->lua, lc_fov),
        .octants = {0, 0, 0, 0, 0, 0, 0, 0},
        .tol = 2. / CONF(prr->lua, nc) / pow(CONF(prr->lua, particle_fraction), 0.33333),
    }};

    MPI_Barrier(comm);

    double a_restart = 0.0;
    if(prr->cli->RestartSnapshotPath) {
        read_snapshot_header(fastpm, prr->cli->RestartSnapshotPath, &a_restart, comm);
        fastpm_info("Restarting from %s at a = %06.4f", prr->cli->RestartSnapshotPath, a_restart);
    } else {
        a_restart = CONF(prr->lua, time_step)[0];
    }

    size_t n_time_step;
    double * time_step = prepare_time_step(prr, a_restart, &n_time_step);

    ENTER(cdmic);
    prepare_cdm(fastpm, prr, time_step[0], comm);
    LEAVE(cdmic);

    ENTER(ncdmic);
    prepare_ncdm(fastpm, prr, time_step[0], comm);
    LEAVE(ncdmic);

    /* FIXME: subsample all species -- probably need different fraction for each species */
    FastPMStore * p = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);
    fastpm_store_fill_subsample_mask(p, CONF(prr->lua, particle_fraction), p->mask, comm);

    FastPMUSMesh * usmesh = NULL;

    if(CONF(prr->lua, lc_write_usmesh)) {
        prepare_lc(fastpm, prr, lc, &usmesh);
    }

    MPI_Barrier(comm);
    ENTER(evolve);
    fastpm_solver_evolve(fastpm, time_step, n_time_step);
    LEAVE(evolve);

    free(time_step);
    if(usmesh)
        fastpm_usmesh_destroy(usmesh);

    free(usmesh);

    if(CONF(prr->lua, lc_write_usmesh)) {
        fastpm_lc_destroy(lc);
    }

    {
        /* destroy ncdm if allocated */
        FastPMStore * ncdm = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_NCDM);
        if(ncdm) {
            fastpm_store_destroy(ncdm);
            free(ncdm);
        }
    }
    fastpm_solver_destroy(fastpm);

    report_memory(comm);

    fastpm_clock_stat(comm);

    return 0;
}

static void
prepare_cosmology(FastPMCosmology * c, RunData * prr) {
    c->h = CONF(prr->lua, h);
    c->Omega_m = CONF(prr->lua, Omega_m);
    c->T_cmb = CONF(prr->lua, T_cmb);
    c->Omega_k = CONF(prr->lua, Omega_k);
    c->w0 = CONF(prr->lua, w0);
    c->wa = CONF(prr->lua, wa);
    c->N_eff = CONF(prr->lua, N_eff);
    c->N_nu = CONF(prr->lua, N_nu);
    c->N_ncdm = CONF(prr->lua, n_m_ncdm);
    c->ncdm_matterlike = CONF(prr->lua, ncdm_matterlike);
    c->ncdm_freestreaming = CONF(prr->lua, ncdm_freestreaming);
    c->growth_mode = CONF(prr->lua, growth_mode);

    int i;
    for(i = 0; i < CONF(prr->lua, n_m_ncdm); i ++) {
        c->m_ncdm[i] = CONF(prr->lua, m_ncdm)[i];
    }
}

static void
rescale_deltak(FastPMSolver * fastpm, PM * pm, FastPMFloat * delta_k, RunData * prr, double aout, double linear_density_redshift)
{
    /* Rescale deltak from the input redshift (linear_density_redshift) to aout */
    FastPMGrowthInfo gi_out;
    FastPMGrowthInfo gi_in;
    fastpm_growth_info_init(&gi_out, aout, fastpm->cosmology);
    fastpm_growth_info_init(&gi_in, 1. / (linear_density_redshift + 1), fastpm->cosmology);
    double linear_evolve = gi_out.D1 / gi_in.D1;

    fastpm_info("Reference linear density is calibrated at redshift %g; multiply by %g to extract to redshift %g.\n", linear_density_redshift, linear_evolve, 1./aout-1);

    fastpm_apply_multiply_transfer(pm, delta_k, delta_k, linear_evolve);
}

static void
prepare_deltak(FastPMSolver * fastpm, PM * pm, FastPMFloat * delta_k, RunData * prr, double aout, double linear_density_redshift,
               const char lineark_filename[],
               const char powerspectrum_filename[])
{
    /*
    computes delta_k given either the filename of the powerspectrum or linear k.
    the input deltak/powerspectrum is at redshift = linear_density_redshift
    the output is at a=aout.
    allows input of the appropraite file for a given species.
    */
    
    /* at this point generating the ic involves delta_k [first check if delta_k has been input]*/
    if(lineark_filename) {
        fastpm_info("Reading Fourier space linear overdensity from %s\n", lineark_filename);
        read_complex(pm, delta_k, lineark_filename, "LinearDensityK", prr->cli->Nwriters);

        if(CONF(prr->lua, inverted_ic)) {
            fastpm_apply_multiply_transfer(pm, delta_k, delta_k, -1);
        }
         /* The linear density field is not redshift zero, then evolve it with the model cosmology to
          * redshift zero.
          * This matches the linear power at the given redshift, not necessarily redshift 0. */
        rescale_deltak(fastpm, pm, delta_k, prr, aout, linear_density_redshift);
        return;
    }

    /* at this point [i.e. delta_k not input] we need a powerspectrum file to convolve the guassian */
    if(!powerspectrum_filename) {
        fastpm_raise(-1, "Need a power spectrum to start the simulation.\n");
    }

    FastPMPowerSpectrum linear_powerspectrum;

    read_powerspectrum(&linear_powerspectrum, powerspectrum_filename, CONF(prr->lua, sigma8), pm_comm(pm));

    
    if(CONF(prr->lua, read_grafic)) {
        fastpm_info("Reading grafic white noise file from '%s'.\n", CONF(prr->lua, read_grafic));
        fastpm_info("GrafIC noise is Fortran ordering. FastPMSolver is in C ordering.\n");
        fastpm_info("The simulation will be transformed x->z y->y z->x.\n");

        FastPMFloat * g_x = pm_alloc(pm);

        read_grafic_gaussian(pm, g_x, CONF(prr->lua, read_grafic));

        /* r2c will reduce the variance. Compensate here.*/
        fastpm_apply_multiply_transfer(pm, g_x, g_x, sqrt(pm_norm(pm)));
        pm_r2c(pm, g_x, delta_k);

        pm_free(pm, g_x);

        goto induce;
    }

    if(CONF(prr->lua, read_whitenoisek)) {
        fastpm_info("Reading Fourier white noise file from '%s'.\n", CONF(prr->lua, read_whitenoisek));

        read_complex(pm, delta_k, CONF(prr->lua, read_whitenoisek), "WhiteNoiseK", prr->cli->Nwriters);
        goto induce;
    }

    /* Nothing to read from, just generate a gadget IC with the seed. */
    fastpm_ic_fill_gaussiank(pm, delta_k, CONF(prr->lua, random_seed), FASTPM_DELTAK_GADGET);

induce:
    if(CONF(prr->lua, remove_cosmic_variance)) {
        fastpm_info("Remove Cosmic variance from initial condition.\n");
        fastpm_ic_remove_variance(pm, delta_k);
    }

    if(CONF(prr->lua, set_mode)) {
        int method = 0;
        /* FIXME: use enums */
        if(0 == strcmp(CONF(prr->lua, set_mode_method), "add")) {
            method = 1;
            fastpm_info("SetMode is add\n");
        } else {
            fastpm_info("SetMode is override\n");
        }
        int i;
        double * c = CONF(prr->lua, set_mode);
        for(i = 0; i < CONF(prr->lua, n_set_mode); i ++) {
            ptrdiff_t mode[4] = {
                c[i * 5 + 0],
                c[i * 5 + 1],
                c[i * 5 + 2],
                c[i * 5 + 3],
            };
            double value = c[i * 5 + 4];
            fastpm_apply_set_mode_transfer(pm, delta_k, delta_k, mode, value, method);
            double result = fastpm_apply_get_mode_transfer(pm, delta_k, mode);
            fastpm_info("SetMode %d : %td %td %td %td value = %g, to = %g\n", i, mode[0], mode[1], mode[2], mode[3], value, result);
        }
    }

    if(CONF(prr->lua, inverted_ic)) {
        fastpm_apply_multiply_transfer(pm, delta_k, delta_k, -1);
    }

    double variance = pm_compute_variance(pm, delta_k);
    fastpm_info("Variance of input white noise is %0.8f, expectation is %0.8f\n", variance, 1.0 - 1.0 / pm_norm(pm));

    if(CONF(prr->lua, write_whitenoisek)) {
        fastpm_info("Writing Fourier white noise to file '%s'.\n", CONF(prr->lua, write_whitenoisek));
        write_complex(pm, delta_k, CONF(prr->lua, write_whitenoisek), "WhiteNoiseK", prr->cli->Nwriters);
    }

    /* introduce correlation */
    if(CONF(prr->lua, f_nl_type) == FASTPM_FNL_NONE) {
        fastpm_info("Inducing correlation to the white noise.\n");

        fastpm_ic_induce_correlation(pm, delta_k,
            (fastpm_fkfunc) fastpm_powerspectrum_eval2, &linear_powerspectrum);
    } else {
        double kmax_primordial;
        kmax_primordial = CONF(prr->lua, nc) / 2.0 * 2.0*M_PI/CONF(prr->lua, boxsize) * CONF(prr->lua, kmax_primordial_over_knyquist);
        fastpm_info("Will set Phi_Gaussian(k)=0 for k>=%f.\n", kmax_primordial);
        FastPMPNGaussian png = {
            .fNL = CONF(prr->lua, f_nl),
            .type = CONF(prr->lua, f_nl_type),
            .kmax_primordial = kmax_primordial,
            .pkfunc = (fastpm_fkfunc) fastpm_powerspectrum_eval2,
            .pkdata = &linear_powerspectrum,
            .h = CONF(prr->lua, h),
            .scalar_amp = CONF(prr->lua, scalar_amp),
            .scalar_spectral_index = CONF(prr->lua, scalar_spectral_index),
            .scalar_pivot = CONF(prr->lua, scalar_pivot)
        };
        fastpm_info("Inducing non gaussian correlation to the white noise.\n");
        fastpm_png_induce_correlation(&png, pm, delta_k);
    }

    /* The linear density field is not redshift zero, then evolve it with the model cosmology to 
     * redshift zero.
     * This matches the linear power at the given redshift, not necessarily redshift 0. */
    rescale_deltak(fastpm, pm, delta_k, prr, aout, linear_density_redshift);

    /* set the mean to 1.0 */
    ptrdiff_t mode[4] = { 0, 0, 0, 0, };

    fastpm_apply_modify_mode_transfer(pm, delta_k, delta_k, mode, 1.0);

    /* add constraints */
    if(CONF(prr->lua, constraints)) {
        FastPM2PCF xi;

        fastpm_2pcf_from_powerspectrum(&xi, (fastpm_fkfunc) fastpm_powerspectrum_eval2, &linear_powerspectrum, CONF(prr->lua, boxsize), CONF(prr->lua, nc));

        FastPMConstrainedGaussian cg = {
            .constraints = malloc(sizeof(FastPMConstraint) * (CONF(prr->lua, n_constraints) + 1)),
        };
        fastpm_info("Applying %d constraints.\n", CONF(prr->lua, n_constraints));
        int i;
        for(i = 0; i < CONF(prr->lua, n_constraints); i ++) {
            double * c = CONF(prr->lua, constraints);
            cg.constraints[i].x[0] = c[4 * i + 0];
            cg.constraints[i].x[1] = c[4 * i + 1];
            cg.constraints[i].x[2] = c[4 * i + 2];
            cg.constraints[i].c = c[4 * i + 3];
            fastpm_info("Constraint %d : %g %g %g peak-sigma = %g\n", i, c[4 * i + 0], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]);
        }
        cg.constraints[i].x[0] = -1;
        cg.constraints[i].x[1] = -1;
        cg.constraints[i].x[2] = -1;
        cg.constraints[i].c = -1;

        if(CONF(prr->lua, write_lineark)) {
            fastpm_info("Writing fourier space linear field before constraints to %s\n", CONF(prr->lua, write_lineark));
            write_complex(pm, delta_k, CONF(prr->lua, write_lineark), "UnconstrainedLinearDensityK", prr->cli->Nwriters);
        }
        fastpm_cg_apply_constraints(&cg, pm, &xi, delta_k);

        free(cg.constraints);
    }

    fastpm_powerspectrum_destroy(&linear_powerspectrum);
}

static double * 
prepare_time_step(RunData * prr, double a0, size_t * n_time_step) 
{
    double * time_step = (double*) malloc((CONF(prr->lua, n_time_step) + 1) * sizeof(double));
    double * all_time_step = CONF(prr->lua, time_step);
    int i;
    for(i = -1; i < CONF(prr->lua, n_time_step) - 1; i++) {
        /* give some slack to cover inaccurate equal. */
        if(all_time_step[i + 1] > a0 + 1e-7) {
            break;
        }
    }
    /* copy from i + 1: -> 1: Is this buggy?*/
    time_step[0] = a0;
    int j;
    for(j = 1; j + i < CONF(prr->lua, n_time_step); j ++ ) {
        time_step[j] = all_time_step[j + i];
    }
    *n_time_step = j;
    return time_step;
}

static void
prepare_cdm(FastPMSolver * fastpm, RunData * prr, double a0, MPI_Comm comm)
{
    if(prr->cli->RestartSnapshotPath) {
        if(CONF(prr->lua, particle_fraction) != 1) {
            fastpm_raise(-1, "Cannot restart because subsampling of particles is enabled.\n");
        }
        fastpm_info("Restarting from snapshot at `%s`.\n", prr->cli->RestartSnapshotPath);
        FastPMStore * p = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);
        FastPMStore po[1];
        fastpm_set_species_snapshot(fastpm, p, NULL, NULL, po, 1.0);
        fastpm_store_read(po, prr->cli->RestartSnapshotPath, prr->cli->Nwriters, comm);
        if(po->meta.a_x != po->meta.a_v) {
            fastpm_raise(-1, "Snapshot velocity and position are out of sync. a_x =% g, a_v = %g.\n", p->meta.a_x, p->meta.a_v);
        }
        if(po->meta.a_x != a0) {
            fastpm_raise(-1, "Snapshot velocity and position are out of sync. a_x =% g, first step = %g.\n", p->meta.a_x, a0);
        }
        fastpm_unset_species_snapshot(fastpm, p, NULL, NULL, po, po->meta.a_x);

        return;
    }
    /* we may need a read gadget ic here too */
    if(CONF(prr->lua, read_runpbic)) {                 //runpbic is old code. dont think about when it comes to ncdm.
        FastPMStore * p = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);
        int temp_dx1 = 0;
        int temp_dx2 = 0;
        if(p->dx1 == NULL) {
            p->dx1 = fastpm_memory_alloc(p->mem, "DX1", sizeof(p->dx1[0]) * p->np_upper, FASTPM_MEMORY_STACK);
            temp_dx1 = 1;
        }
        if(p->dx2 == NULL) {
            p->dx2 = fastpm_memory_alloc(p->mem, "DX2", sizeof(p->dx2[0]) * p->np_upper, FASTPM_MEMORY_STACK);
            temp_dx2 = 1;
        }

        read_runpb_ic(fastpm, p, CONF(prr->lua, read_runpbic));
        fastpm_solver_setup_lpt(fastpm, FASTPM_SPECIES_CDM, NULL, NULL, a0);
        if(temp_dx2) {
            fastpm_memory_free(p->mem, p->dx2);
            p->dx2 = NULL;
        }
        if(temp_dx1) {
            fastpm_memory_free(p->mem, p->dx1);
            p->dx1 = NULL;
        }
        return;
    }

    FastPMFloat * delta_k = pm_alloc(fastpm->pm);

    prepare_deltak(fastpm, fastpm->pm, delta_k, prr, 1.0,
                   CONF(prr->lua, linear_density_redshift), 
                   CONF(prr->lua, read_lineark), 
                   CONF(prr->lua, read_powerspectrum));

    /* Check if linear growth rate has been input for cdm */
    FastPMFuncK * growth_rate_func_k = NULL;
    if (CONF(prr->lua, read_linear_growth_rate)) {
        growth_rate_func_k = malloc(sizeof(FastPMFuncK));
        read_funck(growth_rate_func_k, CONF(prr->lua, read_linear_growth_rate), comm);
        fastpm_info("Reading cdm linear growth rate from file: %s\n", CONF(prr->lua, read_linear_growth_rate));
    } else {
        fastpm_info("No cdm linear growth rate file input.\n");
    }

    if(CONF(prr->lua, write_lineark)) {
        fastpm_info("Writing fourier space linear field to %s\n", CONF(prr->lua, write_lineark));
        write_complex(fastpm->pm, delta_k, CONF(prr->lua, write_lineark), "LinearDensityK", prr->cli->Nwriters);
    }

    if(CONF(prr->lua, write_powerspectrum)) {
        FastPMPowerSpectrum ps;
        /* calculate the power spectrum */
        fastpm_powerspectrum_init_from_delta(&ps, fastpm->pm, delta_k, delta_k);

        char buf[1024];
        sprintf(buf, "%s_linear.txt", CONF(prr->lua, write_powerspectrum));
        fastpm_info("writing linear power spectrum to %s\n", buf);
        if(fastpm->ThisTask == 0) {
            fastpm_path_ensure_dirname(CONF(prr->lua, write_powerspectrum));
            fastpm_powerspectrum_write(&ps, buf, pow(fastpm->config->nc, 3.0));
        }
        fastpm_powerspectrum_destroy(&ps);
    }

    /* set the mass */
    double BoxSize = fastpm->config->boxsize;
    uint64_t NC = fastpm->config->nc;
    double Omega_cdm = fastpm->cosmology->Omega_cdm;
    double M0 = Omega_cdm * FASTPM_CRITICAL_DENSITY * (BoxSize / NC) * (BoxSize / NC) * (BoxSize / NC);
    fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM)->meta.M0 = M0;

    fastpm_solver_setup_lpt(fastpm, FASTPM_SPECIES_CDM, delta_k, growth_rate_func_k, CONF(prr->lua, time_step)[0]);

    if (growth_rate_func_k)
        fastpm_funck_destroy(growth_rate_func_k);
    pm_free(fastpm->basepm, delta_k);
}

static void 
prepare_ncdm(FastPMSolver * fastpm, RunData * prr, double a0, MPI_Comm comm) 
{
    /* don't prepare ncdm when there are no ncdm particles */
    if(CONF(prr->lua, n_m_ncdm) == 0
        || CONF(prr->lua, n_shell) == 0) return;

    if(prr->cli->RestartSnapshotPath) {
        fastpm_raise(-1, "FIXME: add ncdm restart support.\n");
    }

    double boxsize = CONF(prr->lua, boxsize);
    double BoxSize[3] = {boxsize, boxsize, boxsize};
    int n_shell = CONF(prr->lua, n_shell);
    int n_side = CONF(prr->lua, n_side);
    int lvk = CONF(prr->lua, lvk);
    int every = CONF(prr->lua, every_ncdm);

    size_t nc_cdm = CONF(prr->lua, nc);
    size_t nc_ncdm = nc_cdm / every;

    if (CONF(prr->lua, nc) % every != 0) {
        fastpm_raise(-1, "TODO: check this in parameter file. ");
    }
    
    // init the nid
    FastPMncdmInitData* nid = fastpm_ncdm_init_create(
            boxsize,
            fastpm->cosmology, 1 / CONF(prr->lua, time_step)[0] - 1, n_shell, n_side, lvk,
            CONF(prr->lua, ncdm_sphere_scheme));
    
    size_t total_np_ncdm_sites = nc_ncdm * nc_ncdm * nc_ncdm;
    size_t total_np_ncdm = total_np_ncdm_sites * nid->n_split;
    
    FastPMStore * cdm = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);
    
    // create ncdm store for after the split. need to make first for memory order
    FastPMStore * ncdm = malloc(sizeof(FastPMStore));
    fastpm_store_init_evenly(ncdm,
          fastpm_species_get_name(FASTPM_SPECIES_NCDM),
          total_np_ncdm,
          cdm->attributes | COLUMN_MASS,
          fastpm->config->alloc_factor,
          comm);
    
    // create store for ncdm sites (i.e. before splitting)
    // (analogously to how cdm is created in solver_init)
    FastPMStore * ncdm_sites = malloc(sizeof(FastPMStore));
    fastpm_store_init_evenly(ncdm_sites,
          fastpm_species_get_name(FASTPM_SPECIES_NCDM),
          total_np_ncdm_sites,
          cdm->attributes,                           //dont need mass col for sites
          fastpm->config->alloc_factor,
          comm);

    // call fastpm_store_fill on ncdm to give correct qs
    double shift0;
    if(fastpm->config->USE_SHIFT) {
        shift0 = fastpm->config->boxsize / nc_ncdm * 0.5;
    } else {
        shift0 = 0;
    }
    double shift[3] = {shift0, shift0, shift0};

    // fill the ncdm store to make a grid with nc_ncdm grid points in each dim
    ptrdiff_t Nc_ncdm[3] = {nc_ncdm, nc_ncdm, nc_ncdm}; 
    fastpm_store_fill(ncdm_sites, fastpm->pm, shift, Nc_ncdm);

    // stagger the ncdm grid wrt the cdm grid. FIXME: Does this conflict with the shift stuff above?
    int i, d;
    for(i = 0; i < ncdm_sites->np; i ++) {
        for(d = 0; d < 3; d ++) {
            ncdm_sites->x[i][d] += fastpm->config->boxsize / nc_cdm * 0.5;
            if (ncdm_sites->q)
                ncdm_sites->q[i][d] += fastpm->config->boxsize / nc_cdm * 0.5;
        }
    }

    // SPLIT
    fastpm_split_ncdm(nid, ncdm_sites, ncdm, comm);

    // replace ncdm species with the split ncdm
    fastpm_solver_add_species(fastpm,
                              FASTPM_SPECIES_NCDM,
                              ncdm);

    /* after splitting, the ncdm particles have moved so we
       must wrap and move particles to the correct rank */
    int NTask;
    MPI_Comm_size(fastpm->comm, &NTask);
    fastpm_store_wrap(ncdm, BoxSize);
    fastpm_store_decompose(ncdm,
                           (fastpm_store_target_func) FastPMTargetPM,
                           fastpm->pm,
                           fastpm->comm);

    // compute delta_k for ncdm
    FastPMFloat * delta_k = pm_alloc(fastpm->pm);
    
    if(!CONF(prr->lua, read_lineark_ncdm) && !CONF(prr->lua, read_powerspectrum_ncdm)){
        fastpm_info("WARNING: No ncdm powerspectrum input; using cdm's instead."); 
        /*FIXME: would make more sense (better approximation) to use a flat power spectrum instead*/
        prepare_deltak(fastpm, fastpm->pm, delta_k, prr, 1.0, 
                        CONF(prr->lua, linear_density_redshift), 
                        CONF(prr->lua, read_lineark), 
                        CONF(prr->lua, read_powerspectrum));
    } else {
        prepare_deltak(fastpm, fastpm->pm, delta_k, prr, 1.0, 
                        CONF(prr->lua, linear_density_redshift_ncdm), 
                        CONF(prr->lua, read_lineark_ncdm), 
                        CONF(prr->lua, read_powerspectrum_ncdm));
    }
    
    /* Check if linear growth rate has been input for ncdm */
    FastPMFuncK * growth_rate_func_k = NULL;
    if (CONF(prr->lua, read_linear_growth_rate_ncdm)) {
        growth_rate_func_k = malloc(sizeof(FastPMFuncK));
        read_funck(growth_rate_func_k, CONF(prr->lua, read_linear_growth_rate_ncdm), comm);
        fastpm_info("Reading ncdm linear growth rate from file: %s\n", CONF(prr->lua, read_linear_growth_rate_ncdm));
    } else {
        fastpm_info("No ncdm linear growth rate file input. Using internal scale-independent linear growth rate for ICs instead\n");
    }
    // perform lpt
    fastpm_solver_setup_lpt(fastpm, FASTPM_SPECIES_NCDM, delta_k, growth_rate_func_k, a0);

    // FIXME: could add writing of Pncdm functionality (as in prepare_cdm for m).
    if (growth_rate_func_k)
        fastpm_funck_destroy(growth_rate_func_k);
    pm_free(fastpm->pm, delta_k);
    
    fastpm_store_destroy(ncdm_sites);
    fastpm_ncdm_init_free(nid);
}

static void
_usmesh_ready_handler_free(void * userdata) {
    struct usmesh_ready_handler_data * data = userdata;
    fastpm_store_destroy(data->tail);
    free(data->aedges);
    free(data->hist);
    free(data->hist_fof);
    free(data);
}


static void
prepare_lc(FastPMSolver * fastpm, RunData * prr,
        FastPMLightCone * lc, FastPMUSMesh ** usmesh)
{
    if(prr->cli->RestartSnapshotPath) {
        /* We do not svae the tail store, thus the lightcone will have gaps. */
        fastpm_raise(-1, "FIXME: Restarting and lightcone are currently incompatible.");
    }

    {
        if(CONF(prr->lua, ndim_lc_glmatrix) != 2 ||
           CONF(prr->lua, shape_lc_glmatrix)[0] != 4 ||
           CONF(prr->lua, shape_lc_glmatrix)[1] != 4
        ) {
            fastpm_raise(-1, "GL Matrix must be a 4x4 matrix\n");
        }

        int i, j;
        double * c = CONF(prr->lua, lc_glmatrix);

        for (i = 0; i < 4; i ++) {
            for (j = 0; j < 4; j ++) {
                lc->glmatrix[i][j] = *c;
                c ++;
            }
            fastpm_info("GLTransformation [%d] : %g %g %g %g\n", i,
                lc->glmatrix[i][0], lc->glmatrix[i][1], lc->glmatrix[i][2], lc->glmatrix[i][3]);
        }
    }

    {
        int i;
        for(i = 0; i < CONF(prr->lua, n_lc_octants); i ++) {
            int oct = CONF(prr->lua, lc_octants)[i];
            lc->octants[oct % 8] = 1;
            fastpm_info("Using Octant %d\n", oct);
        }
    }

    fastpm_lc_init(lc);


    double lc_amin = HAS(prr->lua, lc_amin)?CONF(prr->lua, lc_amin):CONF(prr->lua, time_step)[0];

    double lc_amax = HAS(prr->lua, lc_amax)?CONF(prr->lua, lc_amax):CONF(prr->lua, time_step)[CONF(prr->lua, n_time_step) - 1];

    fastpm_info("Unstructured Lightcone amin= %g amax=%g\n", lc_amin, lc_amax);

    *usmesh = NULL;
    if(CONF(prr->lua, lc_write_usmesh)) {
        /* FIXME: 1 USMesh per species -- refactor this to a function ;
          */
        FastPMStore * p = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);

        *usmesh = malloc(sizeof(FastPMUSMesh));

        double (*tiles)[3];
        int ntiles;

        if(CONF(prr->lua, ndim_lc_usmesh_tiles) != 2 ||
           CONF(prr->lua, shape_lc_usmesh_tiles)[1] != 3
        ) {
            fastpm_raise(-1, "tiles must be a nx3 matrix, one row per tile.\n");
        }

        ntiles = CONF(prr->lua, shape_lc_usmesh_tiles)[0];
        tiles = malloc(sizeof(tiles[0]) * ntiles);
        int i, j;
        double * c = CONF(prr->lua, lc_usmesh_tiles);

        for (i = 0; i < ntiles; i ++) {
            for (j = 0; j < 3; j ++) {
                tiles[i][j] = (*c) * pm_boxsize(fastpm->pm)[j];
                c ++;
            }
            fastpm_info("Lightcone tiles[%d] : %g %g %g\n", i,
                tiles[i][0], tiles[i][1], tiles[i][2]);
        }
        fastpm_usmesh_init(*usmesh, lc,
                CONF(prr->lua, lc_usmesh_alloc_factor) * pm_volume(fastpm->pm),
                p,
                CONF(prr->lua, lc_usmesh_alloc_factor) *
                p->np_upper,
                tiles, ntiles, lc_amin, lc_amax);

        fastpm_add_event_handler(&fastpm->event_handlers,
            FASTPM_EVENT_INTERPOLATION,
            FASTPM_EVENT_STAGE_BEFORE,
            (FastPMEventHandlerFunction) check_lightcone,
            *usmesh);

        free(tiles);

        struct usmesh_ready_handler_data * data = malloc(sizeof(data[0]));
        data->fastpm = fastpm;
        data->prr = prr;

        double amin = CONF(prr->lua, lc_amin);
        double amax = CONF(prr->lua, lc_amax);
        fastpm_info("No structured mesh requested, generating an AemitIndex with 1024 layers for usmesh. \n");

        int nedges = 1024;
        double * edges = malloc(sizeof(double) * (nedges));

        for(i = 0; i < nedges - 1; i ++) {
            edges[i] = (amax - amin) * i / (nedges - 1) + amin;
        }

        edges[nedges - 1] = amax;
        data->aedges = edges;
        data->Nedges = nedges;

        data->hist = calloc(data->Nedges + 1, sizeof(int64_t));
        data->hist_fof = calloc(data->Nedges + 1, sizeof(int64_t));

        fastpm_store_init(data->tail, p->name, 0, 0, FASTPM_MEMORY_FLOATING);
        data->tail->meta = p->meta;

        fastpm_add_event_handler_free(&(*usmesh)->event_handlers,
                FASTPM_EVENT_LC_READY, FASTPM_EVENT_STAGE_AFTER,
                (FastPMEventHandlerFunction) usmesh_ready_handler,
                data, _usmesh_ready_handler_free);
    }

}

static void
usmesh_ready_handler(FastPMUSMesh * mesh, FastPMLCEvent * lcevent, struct usmesh_ready_handler_data * data)
{
    CLOCK(io);
    CLOCK(sort);
    CLOCK(indexing);

    FastPMSolver * fastpm = data->fastpm;
    RunData * prr = data->prr;
    FastPMStore * tail = data->tail;

    FastPMStore halos[1];

    int64_t np = lcevent->p->np;
    MPI_Allreduce(MPI_IN_PLACE, &np, 1, MPI_LONG, MPI_SUM, fastpm->comm);

    double max_np;
    int max_rank;
    MPIU_stats(fastpm->comm, np, ">.", &max_np, &max_rank);

    fastpm_info("Unstructured LightCone ready : ai = %g af = %g, n = %td max = %g on Task %d\n",
             lcevent->ai, lcevent->af, np, max_np, max_rank);

    char * filebase = fastpm_strdup_printf(CONF(prr->lua, lc_write_usmesh));

    if(CONF(prr->lua, write_fof)) {
        run_usmesh_fof(fastpm, lcevent, halos, prr, tail, mesh->lc);
    }

    /* subsample, this will remove the tail particles that were appended. */
    fastpm_store_subsample(lcevent->p, lcevent->p->mask, lcevent->p);

    ENTER(sort);
    fastpm_sort_snapshot(lcevent->p, fastpm->comm, FastPMSnapshotSortByAEmit, 0);
    if(CONF(prr->lua, write_fof)) {
        fastpm_sort_snapshot(halos, fastpm->comm, FastPMSnapshotSortByAEmit, 0);
    }
    LEAVE(sort);

    ENTER(indexing);
    fastpm_store_histogram_aemit_sorted(lcevent->p, data->hist, data->aedges, data->Nedges, fastpm->comm);
    if(CONF(prr->lua, write_fof)) {
        fastpm_store_histogram_aemit_sorted(halos, data->hist_fof, data->aedges, data->Nedges, fastpm->comm);
    }
    LEAVE(indexing);

    ENTER(io);
    if(lcevent->whence == TIMESTEP_START) {
        fastpm_info("Creating usmesh catalog in %s\n", filebase);
        write_snapshot_header(fastpm, filebase, fastpm->comm);
        write_parameters(filebase, "Header", prr, fastpm->comm);
        fastpm_store_write(lcevent->p, filebase, "w", prr->cli->Nwriters, fastpm->comm);
    } else {
        fastpm_info("Appending usmesh catalog to %s\n", filebase);
        fastpm_store_write(lcevent->p, filebase, "a", prr->cli->Nwriters, fastpm->comm);
    }
    write_aemit_hist(filebase, "1/.", data->hist, data->aedges, data->Nedges, fastpm->comm);
    LEAVE(io);

    /* halos */
    ENTER(io);

    if(CONF(prr->lua, write_fof)) {
        if(lcevent->whence == TIMESTEP_START) {
            /* usmesh fof is always written after the subsample snapshot; no need to create a header */
            fastpm_store_write(halos, filebase, "w", prr->cli->Nwriters, fastpm->comm);
        } else {
            fastpm_store_write(halos, filebase, "a", prr->cli->Nwriters, fastpm->comm);
        }

        char * dataset_attrs = fastpm_strdup_printf("%s/.", halos->name);
        write_aemit_hist(filebase, dataset_attrs, data->hist_fof, data->aedges, data->Nedges, fastpm->comm);
        free(dataset_attrs);
        fastpm_store_destroy(halos);
    }

    LEAVE(io);
    free(filebase);
}

static int cmp_double(const void * p1, const void * p2)
{
    const double * d1 = (const double *) p1;
    const double * d2 = (const double *) p2;
    return (*d1 > *d2) - (*d2 > *d1);
}
static int
check_snapshots(FastPMSolver * fastpm, FastPMInterpolationEvent * event, RunData * prr)
{
    fastpm_info("Checking Snapshots (%0.4f %0.4f) with K(%0.4f->%0.4f|%0.4f) D(%0.4f->%0.4f|%0.4f)\n",
        event->a1, event->a2,
        event->kick->ai, event->kick->af, event->kick->ac,
        event->drift->ai, event->drift->af, event->drift->ac
    );

    /* interpolate and write snapshots, assuming p 
     * is at time a_x and a_v. */
    int nout = CONF(prr->lua, n_aout);
    double * aout = malloc(sizeof(double) * nout);
    memcpy(aout, CONF(prr->lua, aout), sizeof(double) * nout);

    /* the following algorithm requires a sorted aout.
     * since CONF(prr->lua, aout) is immutable, we will always get the same
     * aout array. but it would be better to store the sorted aout somewhere,
     * e.g. refactor this function to a snapshot checker object. */
    qsort(aout, nout, sizeof(double), cmp_double);

    int iout;
    for(iout = prr->iout; iout < nout; iout ++) {
        if(event->a1 == event->a2) {
            /* initial condition */
            /* not requested */
            if(event->a1 != aout[iout]) continue;
            /* Restarting from this snapshot, no need to write again. */
            if(prr->cli->RestartSnapshotPath) continue;
        } else {
            if(event->a1 >= aout[iout]) continue;
            if(event->a2 < aout[iout]) continue;
        }

        FastPMSolver snapshot[1];
        FastPMStore cdm[1];
        FastPMStore ncdm[1];

        /* mostly the original solver, but with two species replaced */
        memcpy(snapshot, fastpm, sizeof(FastPMSolver));
        fastpm_solver_add_species(snapshot, FASTPM_SPECIES_CDM, cdm);

        if(fastpm_solver_get_species(fastpm, FASTPM_SPECIES_NCDM)) {
            fastpm_solver_add_species(snapshot, FASTPM_SPECIES_NCDM, ncdm);
        }

        fastpm_set_snapshot(fastpm, snapshot, event->drift, event->kick, aout[iout]);

        FastPMGrowthInfo gi;
        fastpm_growth_info_init(&gi, aout[iout], fastpm->cosmology);

        fastpm_info("Snapshot a_x = %6.4f, a_v = %6.4f \n", cdm->meta.a_x, cdm->meta.a_v);
        fastpm_info("Growth factor of snapshot %6.4f (a=%0.4f)\n", gi.D1, aout[iout]);
        fastpm_info("Growth rate of snapshot %6.4f (a=%0.4f)\n", gi.f1, aout[iout]);

        take_a_snapshot(snapshot, prr);

        fastpm_unset_snapshot(fastpm, snapshot, event->drift, event->kick, aout[iout]);

        /* do not rewrite this snapshot. */
        prr->iout = iout + 1;
    }

    free(aout);
    return 0;
}

static void
run_fof(FastPMSolver * fastpm, FastPMStore * snapshot, FastPMStore * halos, RunData * prr)
{
    CLOCK(fof);

    FastPMFOFFinder fof = {
        .periodic = 1,
        .nmin = CONF(prr->lua, fof_nmin),
        .kdtree_thresh = CONF(prr->lua, fof_kdtree_thresh),
    };

    char * dataset = fastpm_strdup_printf("LL-%05.3f", CONF(prr->lua, fof_linkinglength));
    fastpm_store_set_name(halos, dataset);
    free(dataset);

    /* convert from fraction of mean separation to simulation distance units. */
    double linkinglength = CONF(prr->lua, fof_linkinglength) * CONF(prr->lua, boxsize) / CONF(prr->lua, nc);
    fastpm_fof_init(&fof, linkinglength, snapshot, fastpm->pm);

    ENTER(fof);
    fastpm_fof_execute(&fof, linkinglength, halos, NULL, NULL);
    LEAVE(fof);

    fastpm_fof_destroy(&fof);
}

static void
_halos_ready (FastPMFOFFinder * finder,
    FastPMStore * halos,
    FastPMStore * p,
    ptrdiff_t * ihalo,
    void ** userdata)
{
    double rmin = *((double*) userdata[0]);
    double halosize = *((double*) userdata[1]);
    FastPMLightCone * lc = (FastPMLightCone*) userdata[2];
    FastPMParticleMaskType * keep_for_tail = (FastPMParticleMaskType *) userdata[3];

    ptrdiff_t i;

    uint32_t * halos_established = malloc(halos->np * sizeof(halos_established[0]));
    fastpm_info("halos_ready sees %td halos\n", halos->np);
    for(i = 0; i < halos->np; i ++) {
        double r = fastpm_lc_distance(lc, halos->x[i]);

        /* this shall not happen */
        if (halos->length[i] < finder->nmin) abort();

        /* only keep reliable halos */
        if(r > rmin + halosize * 0.5) {
            halos_established[i] = 1;
        } else {
            halos_established[i] = 0;
        }
        halos->mask[i] &= halos_established[i];
    }

    for(i = 0; i < p->np; i ++) {
        ptrdiff_t hid = ihalo[i];
        double r_p = fastpm_lc_distance(lc, p->x[i]);
        if (r_p > rmin + halosize) {
            /* not near the tail of the lightcone, do not keep for tail */
            keep_for_tail[i] = 0;
        } else {
            /* near the tail of the lightcone, do not keep those formed reliable halos */
            if(hid >= 0) {
                /* particles in unreliable halos, keep them */
                keep_for_tail[i] = !halos_established[hid];
            } else {
                /* not in halos, keep them too */
                keep_for_tail[i] = 1;
            }
        }
    }
    free(halos_established);
}


static void
run_usmesh_fof(FastPMSolver * fastpm,
        FastPMLCEvent * lcevent,
        FastPMStore * halos,
        RunData * prr,
        FastPMStore * tail,
        FastPMLightCone * lc)
{
    CLOCK(fof);
    CLOCK(sort);

    char * dataset = fastpm_strdup_printf("LL-%05.3f", CONF(prr->lua, fof_linkinglength));
    fastpm_store_set_name(halos, dataset);
    free(dataset);

    double maxhalosize = CONF(prr->lua, lc_usmesh_fof_padding); /* MPC/h, used to cut along z direction. */
    FastPMStore * p = lcevent->p;
    ptrdiff_t i;

    /* not the first segment, add the left over particles to the FOF */

    /* avoid including the tail particles in the subsample;
     * by clearing their mask */
    for(i = 0; i < tail->np; i ++) {
        tail->mask[i] = 0;
    }
    fastpm_store_extend(lcevent->p, tail);
    fastpm_store_destroy(tail);

    /* FIXME: register event to mask out particles*/
    FastPMParticleMaskType * keep_for_tail = fastpm_memory_alloc(p->mem, "keep",
                                    sizeof(keep_for_tail[0]) * lcevent->p->np_upper, FASTPM_MEMORY_FLOATING);

    ENTER(fof);

    FastPMFOFFinder fof = {
        .periodic = 0,
        .nmin = CONF(prr->lua, fof_nmin),
        .kdtree_thresh = CONF(prr->lua, fof_kdtree_thresh),
    };

    /* convert from fraction of mean separation to simulation distance units. */
    double linkinglength = CONF(prr->lua, fof_linkinglength) * CONF(prr->lua, boxsize) / CONF(prr->lua, nc);
    fastpm_fof_init(&fof, linkinglength, p, fastpm->pm);

    double rmin = lc->speedfactor * HorizonDistance(lcevent->af, lc->horizon);
    /* FIXME: clean this up! halos_ready is converted from an event handler. */
    void * userdata[5];
    userdata[0] = & rmin;
    userdata[1] = & maxhalosize;
    userdata[2] = lc;
    userdata[3] = keep_for_tail;

    ptrdiff_t * ihalo;
    ENTER(fof);
    fastpm_fof_execute(&fof, linkinglength, halos, &ihalo, NULL);
    LEAVE(fof);

    _halos_ready(&fof, halos, p, ihalo, userdata);

    fastpm_store_subsample(halos, halos->mask, halos);
    fastpm_fof_destroy(&fof);

    uint64_t ntail = 0;
    for(i = 0; i < p->np; i ++) {
        if(keep_for_tail[i]) ntail ++;
    }

    fastpm_store_init(tail, p->name, ntail, p->attributes, FASTPM_MEMORY_FLOATING);
    fastpm_store_subsample(p, keep_for_tail, tail);

    MPI_Allreduce(MPI_IN_PLACE, &ntail, 1, MPI_LONG, MPI_SUM, fastpm->comm);

    fastpm_info("%td particles will be reused in next batch for usmesh FOF\n", ntail);

    fastpm_memory_free(p->mem, keep_for_tail);
}

static int 
take_a_snapshot(FastPMSolver * fastpm, RunData * prr) 
{
    FastPMStore * cdm = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);

    double aout = cdm->meta.a_x;
    double z_out= 1.0/aout - 1.0;

    CLOCK(io);
    CLOCK(sort);

    FastPMStore halos[1];

    if(CONF(prr->lua, write_fof)) {
        run_fof(fastpm, cdm, halos, prr);
    }

    /* do this before write_snapshot, because white_snapshot messes up with the domain decomposition. */
    if(CONF(prr->lua, write_nonlineark)) {
        char * filename = fastpm_strdup_printf("%s_%0.04f", CONF(prr->lua, write_nonlineark), aout);
        FastPMPainter painter[1];

        FastPMFloat * rho_x = pm_alloc(fastpm->basepm);
        FastPMFloat * rho_k = pm_alloc(fastpm->basepm);

        fastpm_painter_init(painter, fastpm->basepm, fastpm->config->PAINTER_TYPE, fastpm->config->painter_support);

        fastpm_paint(painter, rho_x, cdm, FASTPM_FIELD_DESCR_NONE);
        pm_r2c(fastpm->basepm, rho_x, rho_k);

        write_complex(fastpm->basepm, rho_k, filename, "DensityK", prr->cli->Nwriters);

        pm_free(fastpm->basepm, rho_k);
        pm_free(fastpm->basepm, rho_x);
        free(filename);
    }

    FastPMStore subsample[FASTPM_SOLVER_NSPECIES];

    int si;
    for(si = 0; si < FASTPM_SOLVER_NSPECIES; si ++) {
        FastPMStore * p = fastpm_solver_get_species(fastpm, si);
        if (!p) continue;

        if(CONF(prr->lua, particle_fraction) < 1) {

            fastpm_store_init(&subsample[si],
                        p->name,
                        fastpm_store_subsample(p, p->mask, NULL),
                        p->attributes & (~COLUMN_ACC) & (~COLUMN_MASK),
                        FASTPM_MEMORY_FLOATING);

            fastpm_store_subsample(p, p->mask, &subsample[si]);
        } else {
            memcpy(&subsample[si], p, sizeof(FastPMStore));
        }

        if(CONF(prr->lua, sort_snapshot)) {
            ENTER(sort);
            fastpm_sort_snapshot(&subsample[si], fastpm->comm, FastPMSnapshotSortByID, 0);
            LEAVE(sort);
        }
    }

    if(CONF(prr->lua, write_snapshot)) {
        char filebase[1024];
        sprintf(filebase, "%s_%0.04f", CONF(prr->lua, write_snapshot), aout);

        ENTER(io);
        write_snapshot_header(fastpm, filebase, fastpm->comm);
        write_parameters(filebase, "Header", prr, fastpm->comm);

        for(si = 0; si < FASTPM_SOLVER_NSPECIES; si ++) {
            if(!fastpm_solver_get_species(fastpm, si)) continue;
            fastpm_store_write(&subsample[si], filebase, "w", prr->cli->Nwriters, fastpm->comm);
        }

        LEAVE(io);
        fastpm_info("snapshot %s [%s] written at z = %6.4f a = %6.4f \n", filebase, "1", z_out, aout);
    }

    if(CONF(prr->lua, write_fof)) {
        char filebase[1024];
        sprintf(filebase, "%s_%0.04f", CONF(prr->lua, write_fof), aout);

        ENTER(sort);
        fastpm_sort_snapshot(halos, fastpm->comm, FastPMSnapshotSortByLength, 0);
        LEAVE(sort);

        ENTER(io);
        /* over write the header of write_fof and write_snapshot are the same */
        write_snapshot_header(fastpm, filebase, fastpm->comm);
        write_parameters(filebase, "Header", prr, fastpm->comm);

        fastpm_store_write(halos, filebase, "w", prr->cli->Nwriters, fastpm->comm);

        LEAVE(io);

        fastpm_info("fof %s [%s] written at z = %6.4f a = %6.4f \n", filebase, halos->name, z_out, aout);

        fastpm_store_destroy(halos);
    }

    if(CONF(prr->lua, write_runpb_snapshot)) {
        /* RunPB only has CDM*/
        char filebase[1024];

        sprintf(filebase, "%s_%0.04f.bin", CONF(prr->lua, write_runpb_snapshot), aout);

        ENTER(io);
        write_runpb_snapshot(fastpm, &subsample[FASTPM_SPECIES_CDM], filebase);
        LEAVE(io);

        fastpm_info("runpb snapshot %s written z = %6.4f a = %6.4f\n", 
                filebase, z_out, aout);

    }

    if(CONF(prr->lua, particle_fraction) < 1) {
        /* subsamples[] are new store objects rather than references of the snapshot store;
         *  destroy them here */
        int si;
        for(si = FASTPM_SOLVER_NSPECIES - 1; si >= 0; si --) {
            if(!fastpm_solver_get_species(fastpm, si)) continue;

            fastpm_store_destroy(&subsample[si]);
        }
    }
    return 0;
}

static int
check_lightcone(FastPMSolver * fastpm, FastPMInterpolationEvent * event, FastPMUSMesh * usmesh)
{
    double a1 = event->drift->ai > event->drift->af ? event->drift->af: event->drift->ai;
    double a2 = event->drift->ai > event->drift->af ? event->drift->ai: event->drift->af;

    fastpm_usmesh_intersect(usmesh, event->drift, event->kick, a1, a2, event->whence, fastpm->comm);

    int64_t np_lc = usmesh->np_before + usmesh->p->np;
    MPI_Allreduce(MPI_IN_PLACE, &np_lc, 1, MPI_LONG, MPI_SUM, fastpm->comm);
    fastpm_info("Total number of particles wrote into lightcone: %ld\n", np_lc);
    return 0;
}

static int 
print_transition(FastPMSolver * fastpm, FastPMTransitionEvent * event, RunData * prr)
{
    FastPMTransition * trans = event->transition;
    char * action;
    switch (trans->action) {
        case FASTPM_ACTION_FORCE:
            action = "FORCE";
        break;
        case FASTPM_ACTION_KICK:
            action = "KICK";
        break;
        case FASTPM_ACTION_DRIFT:
            action = "DRIFT";
        break;
        default:
            action = "Unknown";
        break;
    }
    fastpm_info("==== -> %03d [%03d %03d %03d] a_i = %6.4f a_f = %6.4f a_r = %6.4f Action = %s(%d) ====\n",
            trans->iend,
            trans->end->x,
            trans->end->v,
            trans->end->force,
            trans->a.i, trans->a.f, trans->a.r, action, trans->action);
    report_memory(fastpm->comm);
    return 0;
}

static char PEAK_ALLOC_STATUS[8192];
static size_t PEAK_ALLOC_BYTES;
static void
_memory_peak_handler(FastPMMemory * mem, void * userdata)
{
    PEAK_ALLOC_BYTES = mem->peak_bytes;

    fastpm_memory_dump_status_str(mem, PEAK_ALLOC_STATUS, 8192);
}

static void
report_memory(MPI_Comm comm)
{
    static double oldpeak = 0;
    static int oldrank = -1;

    double max_used_bytes;
    double min_used_bytes;
    int max_rank;
    int min_rank;

    int ThisTask;
    MPI_Comm_rank(comm, &ThisTask);


    MPIU_stats(comm, PEAK_ALLOC_BYTES, "<,>.",
            &min_used_bytes,
            &min_rank,
            &max_used_bytes,
            &max_rank);

    if(max_used_bytes != oldpeak || max_rank != oldrank) {
        if(max_rank == ThisTask) {
            fastpm_ilog(INFO, "Task %d Peak memory usage max: %g MB min : %g MB\n",
                ThisTask,
                max_used_bytes / 1024. / 1024,
                min_used_bytes / 1024. / 1024
            );
            fastpm_ilog(INFO, PEAK_ALLOC_STATUS);
        }
    }
    oldpeak = max_used_bytes;
    oldrank = max_rank;
}

static int
report_lpt(FastPMSolver * fastpm, FastPMLPTEvent * event, RunData * prr)
{
    double dx1_std[3], dx2_std[3];

    MPI_Comm comm = fastpm->comm;

    fastpm_store_summary(event->p, COLUMN_DX1, comm, "s", dx1_std);
    fastpm_store_summary(event->p, COLUMN_DX2, comm, "s", dx2_std);

    fastpm_info("dx1  : %g %g %g %g\n", 
            dx1_std[0], dx1_std[1], dx1_std[2],
            (dx1_std[0] + dx1_std[1] + dx1_std[2]) / 3.0);

    fastpm_info("dx2  : %g %g %g %g\n", 
            dx2_std[0], dx2_std[1], dx2_std[2],
            (dx2_std[0] + dx2_std[1] + dx2_std[2]) / 3.0);

    return 0;
}

static int
report_domain(FastPMSolver * fastpm, FastPMForceEvent * event, RunData * prr)
{
    MPI_Comm comm = fastpm->comm;

    fastpm_info("Force Calculation Nmesh = %d ====\n", pm_nmesh(event->pm)[0]);
    /* FIXME: find a way to iterat over all species */
    FastPMStore * p = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);
    {
        double np_max;
        double np_min;
        double np_mean;
        double np_std;

        MPIU_stats(comm, p->np, "<->s",
                &np_min, &np_mean, &np_max, &np_std);

        double min = np_min / np_mean;
        double max = np_max / np_mean;
        double std = np_std / np_mean;

        fastpm_info("Load imbalance: min = %g max = %g std = %g\n", min, max, std);
    }
    {
        double min[3], max[3], std[3];

        fastpm_store_summary(p, COLUMN_POS, comm, "<>", min, max);
        fastpm_store_summary(p, COLUMN_VEL, comm, "s", std);

        fastpm_info("Position range (a = %06.4f): min = %g %g %g max = %g %g %g \n",
                p->meta.a_x,
                min[0], min[1], min[2],
                max[0], max[1], max[2]);
        fastpm_info("Velocity dispersion (a = %06.4f): std = %g %g %g\n",
                p->meta.a_v, std[0], std[1], std[2]);
    }

    return 0;
}

static int
write_powerspectrum(FastPMSolver * fastpm, FastPMForceEvent * event, RunData * prr) 
{

    int K_LINEAR = CONF(prr->lua, enforce_broadband_kmax);
    MPI_Comm comm = fastpm->comm;

    {
        FastPMStore * p = fastpm_solver_get_species(fastpm, FASTPM_SPECIES_CDM);
        double fstd[3];

        fastpm_store_summary(p, COLUMN_ACC, comm, "s", fstd);

        fastpm_info("Force dispersion: std = %g %g %g\n",
                fstd[0], fstd[1], fstd[2]);
    }

    CLOCK(compute);
    CLOCK(io);

    ENTER(compute);

    FastPMPowerSpectrum ps;
    /* calculate the power spectrum */
    fastpm_powerspectrum_init_from_delta(&ps, event->pm, event->delta_k, event->delta_k);

    double Plin = fastpm_powerspectrum_large_scale(&ps, K_LINEAR);

    double Sigma8 = fastpm_powerspectrum_sigma(&ps, 8);

    FastPMGrowthInfo gi;
    fastpm_growth_info_init(&gi, event->a_f, fastpm->cosmology);

    Plin /= pow(gi.D1, 2.0);
    Sigma8 /= pow(gi.D1, 2.0);

    fastpm_info("D^2(%g, 1.0) P(k<%g) = %g Sigma8 = %g\n", event->a_f, K_LINEAR * 6.28 / pm_boxsize(event->pm)[0], Plin, Sigma8);

    LEAVE(compute);

    MPI_Barrier(fastpm->comm);

    ENTER(io);
    if(CONF(prr->lua, write_powerspectrum)) {
        char buf[1024];
        sprintf(buf, "%s_%0.04f.txt", CONF(prr->lua, write_powerspectrum), event->a_f);
        fastpm_info("writing power spectrum to %s\n", buf);
        if(fastpm->ThisTask == 0) {
            fastpm_path_ensure_dirname(CONF(prr->lua, write_powerspectrum));
            fastpm_powerspectrum_write(&ps, buf, event->N);
        }
    }
    LEAVE(io);

    fastpm_powerspectrum_destroy(&ps);

    return 0;
}

int
read_powerspectrum(FastPMPowerSpectrum * ps, const char filename[], const double sigma8, MPI_Comm comm)
{
    fastpm_info("Powerspecectrum file: %s\n", filename);

    int myrank;
    MPI_Comm_rank(comm, &myrank);
    char * content;
    if(myrank == 0) {
        content = fastpm_file_get_content(filename);
        if(content == NULL) {
            fastpm_raise(-1, "Failed to read powerspectrum from file %s\n", filename);
        }
        int size = strlen(content);
        MPI_Bcast(&size, 1, MPI_INT, 0, comm);
        MPI_Bcast(content, size + 1, MPI_BYTE, 0, comm);
    } else {
        int size = 0;
        MPI_Bcast(&size, 1, MPI_INT, 0, comm);
        content = malloc(size + 1);
        MPI_Bcast(content, size + 1, MPI_BYTE, 0, comm);
    }
    if (0 != fastpm_powerspectrum_init_from_string(ps, content)) {
        fastpm_raise(-1, "Failed to parse the powerspectrum\n");
    }
    free(content);

    fastpm_info("Found %d pairs of values in input spectrum table\n", ps->base.size);

    double sigma8_input= fastpm_powerspectrum_sigma(ps, 8);
    fastpm_info("Input power spectrum sigma8 %f\n", sigma8_input);

    if(sigma8 > 0) {
        fastpm_info("Expected power spectrum sigma8 %g; correction applied. \n", sigma8);
        fastpm_powerspectrum_scale(ps, pow(sigma8 / sigma8_input, 2));
    }
    return 0;
}


