#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>
#include <alloca.h>
#include <mpi.h>
#include <math.h>
#include <signal.h>
#include <getopt.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fastpm/libfastpm.h>
#include <fastpm/prof.h>
#include <fastpm/logging.h>
#include <fastpm/string.h>

#include "parameters.h"


/* command-line arguments */
static char * ParamFileName;

static void 
parse_args(int * argc, char *** argv, Parameters * prr);

static int 
take_a_snapshot(FastPM * fastpm, PMStore * snapshot, double aout, Parameters * prr);

static void 
_mkdir(const char *dir);
static void 
ensure_dir(char * path);

int 
read_runpb_ic(FastPM * fastpm, PMStore * p, char * filename);

void 
read_grafic_gaussian(PM * pm, FastPMFloat * g_x, char * filename);

int 
write_runpb_snapshot(FastPM * fastpm, PMStore * p, char * filebase);

int 
write_snapshot(FastPM * fastpm, PMStore * p, char * filebase, char * parameters, int Nwriters);

int 
read_snapshot(FastPM * fastpm, PMStore * p, char * filebase);

int 
read_parameters(char * filename, Parameters * param, int argc, char ** argv, MPI_Comm comm);

int
read_powerspectrum(FastPMPowerSpectrum *ps, const char filename[], const double sigma8, MPI_Comm comm);

int run_fastpm(FastPM * fastpm, Parameters * prr, MPI_Comm comm);

struct enum_entry {
    char * str;
    int value;
};

static int parse_enum(const char * str, struct enum_entry * enum_table)
{
    struct enum_entry * p;
    for(p = enum_table; p->str; p ++) {
        if(!strcmp(str, p->str)) {
            return p->value;
        }
    }
    int n = 10;
    for(p = enum_table; p->str; p ++) {
        n += strlen(p->str) + 10;
    }
    char * options = malloc(n);
    options[0] = 0;
    for(p = enum_table; p->str; p ++) {
        if(p != enum_table)
            strcat(options, ", ");
        strcat(options, "`");
        strcat(options, p->str);
        strcat(options, "`");
    }

    fastpm_raise(9999, "value `%s` is not recognized. Options are %s \n",
        str, options);
    return 0;
}


int main(int argc, char ** argv) {

    MPI_Init(&argc, &argv);

    Parameters prr = {0};

    parse_args(&argc, &argv, &prr);

    MPI_Comm comm = MPI_COMM_WORLD; 

    libfastpm_init();

    fastpm_set_msg_handler(fastpm_default_msg_handler, comm, NULL);

    read_parameters(ParamFileName, &prr, argc, argv, comm);

    /* convert parameter files pm_nc_factor into VPMInit */
    VPMInit * vpminit = alloca(sizeof(VPMInit) * (prr.n_pm_nc_factor + 1));
    int i;
    for(i = 0; i < prr.n_pm_nc_factor; i ++) {
        vpminit[i].a_start = prr.change_pm[i];
        vpminit[i].pm_nc_factor = prr.pm_nc_factor[i];
    }
    /* mark the end */
    vpminit[i].pm_nc_factor = 0;

    fastpm_info("np_alloc_factor = %g\n", prr.np_alloc_factor);

    FastPM * fastpm = & (FastPM) {
        .nc = prr.nc,
        .alloc_factor = prr.np_alloc_factor, 
        .vpminit = vpminit,
        .boxsize = prr.boxsize,
        .omega_m = prr.omega_m,
        .USE_NONSTDDA = !prr.cola_stdda,
        .USE_DX1_ONLY = prr.use_dx1_only,
        .nLPT = -2.5f,
        .K_LINEAR = prr.enforce_broadband_kmax,
        .USE_SHIFT = 1, /* compatible with old behavior, shift particles by 0.5 mesh*/
    };

    {
        if(!strcmp(prr.force_mode, "cola")) {
            fastpm->USE_COLA = 1;
            fastpm->USE_ZOLA = 0;
        } else
        if(!strcmp(prr.force_mode, "zola")) {
            fastpm->USE_COLA = 1;
            fastpm->USE_ZOLA = 0;
        } else
        if(!strcmp(prr.force_mode, "pm")) {
            fastpm->USE_COLA = 0;
            fastpm->USE_ZOLA = 0;
        } else {
            fastpm_raise(-1, "Wrong force_mode `%s`. This shall not happen. Check lua schema.\n", prr.force_mode);
        }
    }
    {
        struct enum_entry table[] = {
            {"linear", FASTPM_MODEL_LINEAR},
            {"2lpt", FASTPM_MODEL_2LPT},
            {"za", FASTPM_MODEL_ZA},
            {"pm", FASTPM_MODEL_PM},
            {"none", FASTPM_MODEL_NONE},
            {NULL, -1},
        };
        fastpm->USE_MODEL = parse_enum(prr.enforce_broadband_mode, table);
    }
    {
        struct enum_entry table[] = {
            {"eastwood", FASTPM_KERNEL_EASTWOOD},
            {"3_4", FASTPM_KERNEL_3_4},
            {"5_4", FASTPM_KERNEL_5_4},
            {"3_2", FASTPM_KERNEL_3_2},
            {NULL, -1},
        };
        fastpm->KERNEL_TYPE = parse_enum(prr.kernel_type, table);
    }
    {
        struct enum_entry table[] = {
            {"none", FASTPM_DEALIASING_NONE},
            {"gaussian", FASTPM_DEALIASING_GAUSSIAN},
            {"twothird", FASTPM_DEALIASING_TWO_THIRD},
            {NULL, -1},
        };
        fastpm->DEALIASING_TYPE = parse_enum(prr.dealiasing_type, table);
    }

    run_fastpm(fastpm, &prr, comm);

    libfastpm_cleanup();

    MPI_Finalize();
    return 0;
}

static int 
check_snapshots(FastPM * fastpm, void * unused, Parameters * prr);

static int 
write_powerspectrum(FastPM * fastpm, FastPMFloat * delta_k, double a_x, Parameters * prr);

static void 
prepare_ic(FastPM * fastpm, Parameters * prr, MPI_Comm comm);

int run_fastpm(FastPM * fastpm, Parameters * prr, MPI_Comm comm) {
    CLOCK(init);
    CLOCK(ic);
    CLOCK(evolve);

    const double rho_crit = 27.7455;
    const double M0 = prr->omega_m*rho_crit*pow(prr->boxsize / prr->nc, 3.0);
    fastpm_info("mass of a particle is %g 1e10 Msun/h\n", M0); 

    MPI_Barrier(comm);
    ENTER(init);

    fastpm_init(fastpm, 
        prr->NprocY, prr->UseFFTW, 
        comm);

    LEAVE(init);

    fastpm_add_extension(fastpm,
        FASTPM_EXT_AFTER_FORCE,
        write_powerspectrum,
        prr);

    fastpm_add_extension(fastpm,
        FASTPM_EXT_BEFORE_KICK,
        check_snapshots,
        prr);

    fastpm_add_extension(fastpm,
        FASTPM_EXT_BEFORE_DRIFT,
        check_snapshots,
        prr);

    MPI_Barrier(comm);
    ENTER(ic);
    prepare_ic(fastpm, prr, comm);
    LEAVE(ic);

    MPI_Barrier(comm);
    ENTER(evolve);

    fastpm_evolve(fastpm, prr->time_step, prr->n_time_step);

    LEAVE(evolve);

    fastpm_destroy(fastpm);

    fastpm_clock_stat(comm);
    return 0;
}

static void 
prepare_ic(FastPM * fastpm, Parameters * prr, MPI_Comm comm) 
{
    /* we may need a read gadget ic here too */
    if(prr->read_runpbic) {
        read_runpb_ic(fastpm, fastpm->p, prr->read_runpbic);
        fastpm_setup_ic(fastpm, NULL);
        return;
    } 

    /* at this point generating the ic involves delta_k */
    FastPMFloat * delta_k = pm_alloc(fastpm->pm_2lpt);

    if(prr->read_lineark) {
        fastpm_info("Reading Fourier space linear overdensity from %s\n", prr->read_lineark);
        fastpm_utils_load(fastpm->pm_2lpt, prr->read_lineark, delta_k);
        goto finish;
    } 

    /* at this power we need a powerspectrum file to convolve the guassian */
    if(!prr->read_powerspectrum) {
        fastpm_raise(-1, "Need a power spectrum to start the simulation.\n");
    }

    FastPMPowerSpectrum linear_powerspectrum;

    read_powerspectrum(&linear_powerspectrum, prr->read_powerspectrum, prr->sigma8, comm);

    if(prr->read_grafic) {
        fastpm_info("Reading grafic white noise file from '%s'.\n", prr->read_grafic);
        fastpm_info("GrafIC noise is Fortran ordering. FastPM is in C ordering.\n");
        fastpm_info("The simulation will be transformed x->z y->y z->x.\n");

        FastPMFloat * g_x = pm_alloc(fastpm->pm_2lpt);

        read_grafic_gaussian(fastpm->pm_2lpt, g_x, prr->read_grafic);
        pm_r2c(fastpm->pm_2lpt, g_x, delta_k);
        fastpm_ic_induce_correlation(fastpm->pm_2lpt, delta_k,
            (fastpm_pkfunc) fastpm_powerspectrum_eval2, &linear_powerspectrum);
        pm_free(fastpm->pm_2lpt, g_x);
        goto finish;
    }

    if(prr->read_whitenoisek) {
        fastpm_info("Reading Fourier white noise file from '%s'.\n", prr->read_whitenoisek);

        fastpm_utils_load(fastpm->pm_2lpt, prr->read_whitenoisek, delta_k);
        fastpm_ic_induce_correlation(fastpm->pm_2lpt, delta_k,
            (fastpm_pkfunc) fastpm_powerspectrum_eval2, &linear_powerspectrum);
        goto finish;

    }

    /* Nothing to read from, just generate a gadget IC with the seed. */

    fastpm_ic_fill_gaussiank(fastpm->pm_2lpt, delta_k, prr->random_seed, FASTPM_DELTAK_GADGET);
    //fastpm_utils_fill_deltak(fastpm->pm_2lpt, delta_k, prr->random_seed, FASTPM_DELTAK_FAST);

    if(prr->remove_cosmic_variance) {
        fastpm_ic_remove_variance(fastpm->pm_2lpt, delta_k);
    }
    if(prr->write_whitenoisek) {
        fastpm_utils_dump(fastpm->pm_2lpt, prr->write_whitenoisek, delta_k);
    }
    fastpm_ic_induce_correlation(fastpm->pm_2lpt, delta_k,
        (fastpm_pkfunc) fastpm_powerspectrum_eval2, &linear_powerspectrum);

    fastpm_powerspectrum_destroy(&linear_powerspectrum);

    /* our write out and clean up stuff.*/
finish:
    if(prr->inverted_ic) {
        ptrdiff_t i;
        for(i = 0; i < pm_size(fastpm->pm_2lpt); i ++) {
            delta_k[i] *= -1;
        }
    }

    if(prr->write_lineark) {
        fastpm_info("Writing fourier space noise to %s\n", prr->write_lineark);
        ensure_dir(prr->write_lineark);
        fastpm_utils_dump(fastpm->pm_2lpt, prr->write_lineark, delta_k);
    }

    fastpm_setup_ic(fastpm, delta_k);

    pm_free(fastpm->pm_2lpt, delta_k);
}

static int check_snapshots(FastPM * fastpm, void * unused, Parameters * prr) {
    fastpm_interp(fastpm, prr->aout, prr->n_aout, (fastpm_interp_action)take_a_snapshot, prr);
    return 0;
}

static int 
take_a_snapshot(FastPM * fastpm, PMStore * snapshot, double aout, Parameters * prr) 
{
    CLOCK(io);
    CLOCK(meta);

    if(prr->write_snapshot) {
        char filebase[1024];
        double z_out= 1.0/aout - 1.0;
        int Nwriters = prr->Nwriters;
        if(Nwriters == 0) {
            MPI_Comm_size(fastpm->comm, &Nwriters);
        }
        sprintf(filebase, "%s_%0.04f", prr->write_snapshot, aout);

        fastpm_info("Writing snapshot %s at z = %6.4f a = %6.4f with %d writers\n", 
                filebase, z_out, aout, Nwriters);

        ENTER(meta);
        ensure_dir(filebase);
        LEAVE(meta);

        MPI_Barrier(fastpm->comm);
        ENTER(io);
        write_snapshot(fastpm, snapshot, filebase, prr->string, Nwriters);
        LEAVE(io);

        fastpm_info("snapshot %s written\n", filebase);
    }
    if(prr->write_runpb_snapshot) {
        char filebase[1024];
        double z_out= 1.0/aout - 1.0;

        sprintf(filebase, "%s_%0.04f.bin", prr->write_runpb_snapshot, aout);
        ENTER(meta);
        ensure_dir(filebase);
        LEAVE(meta);

        MPI_Barrier(fastpm->comm);
        ENTER(io);
        write_runpb_snapshot(fastpm, snapshot, filebase);

        LEAVE(io);

        fastpm_info("snapshot %s written z = %6.4f a = %6.4f\n", 
                filebase, z_out, aout);

    }
    if(prr->write_nonlineark) {
        char * filename = fastpm_strdup_printf("%s_%0.04f", prr->write_nonlineark, aout);
        FastPMFloat * rho_k = pm_alloc(fastpm->pm_2lpt);
        fastpm_utils_paint(fastpm->pm_2lpt, snapshot, NULL, rho_k, NULL, 0);
        fastpm_utils_dump(fastpm->pm_2lpt, filename, rho_k);
        pm_free(fastpm->pm_2lpt, rho_k);
        free(filename);
    }
    return 0;
}

static int
write_powerspectrum(FastPM * fastpm, FastPMFloat * delta_k, double a_x, Parameters * prr) 
{
    CLOCK(compute);
    CLOCK(io);

    fastpm_report_memory(fastpm->comm);

    MPI_Barrier(fastpm->comm);
    ENTER(compute);

    FastPMPowerSpectrum ps;
    /* calculate the power spectrum */
    fastpm_powerspectrum_init_from_delta(&ps, fastpm->pm, delta_k, delta_k);

    double Plin = fastpm_powerspectrum_large_scale(&ps, fastpm->K_LINEAR);

    double Sigma8 = fastpm_powerspectrum_sigma(&ps, 8);

    Plin /= pow(fastpm_growth_factor(fastpm, a_x), 2.0);
    Sigma8 /= pow(fastpm_growth_factor(fastpm, a_x), 2.0);

    fastpm_info("D^2(%g, 1.0) P(k<%g) = %g Sigma8 = %g\n", a_x, fastpm->K_LINEAR * 6.28 / fastpm->boxsize, Plin, Sigma8);

    LEAVE(compute);

    MPI_Barrier(fastpm->comm);

    ENTER(io);
    if(prr->write_powerspectrum) {
        if(fastpm->ThisTask == 0) {
            ensure_dir(prr->write_powerspectrum);
            char buf[1024];
            sprintf(buf, "%s_%0.04f.txt", prr->write_powerspectrum, a_x);
            fastpm_powerspectrum_write(&ps, buf, pow(fastpm->nc, 3.0));
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

    fastpm_info("Found %d pairs of values in input spectrum table\n", ps->size);

    double sigma8_input= fastpm_powerspectrum_sigma(ps, 8);
    fastpm_info("Input power spectrum sigma8 %f\n", sigma8_input);

    if(sigma8 > 0) {
        fastpm_info("Expected power spectrum sigma8 %g; correction applied. \n", sigma8);
        fastpm_powerspectrum_scale(ps, pow(sigma8 / sigma8_input, 2));
    }
    return 0;
}


static void 
parse_args(int * argc, char *** argv, Parameters * prr) 
{
    char opt;
    extern int optind;
    extern char * optarg;
    prr->UseFFTW = 0;
    ParamFileName = NULL;
    prr->NprocY = 0;    
    prr->Nwriters = 0;
    while ((opt = getopt(*argc, *argv, "h?y:fW:")) != -1) {
        switch(opt) {
            case 'y':
                prr->NprocY = atoi(optarg);
            break;
            case 'f':
                prr->UseFFTW = 1;
            break;
            case 'W':
                prr->Nwriters = atoi(optarg);
            break;
            case 'h':
            case '?':
            default:
                goto usage;
            break;
        }
    }
    if(optind >= *argc) {
        goto usage;
    }

    ParamFileName = (*argv)[optind];
    *argv += optind;
    *argc -= optind; 
    return;

usage:
    printf("Usage: fastpm [-W Nwriters] [-f] [-y NprocY] paramfile\n"
    "-f Use FFTW \n"
    "-y Set the number of processes in the 2D mesh\n"
    "-n Throttle IO (bigfile only) \n"
);
    MPI_Finalize();
    exit(1);
}

static void 
ensure_dir(char * path) 
{
    int i = strlen(path);
    char * dup = alloca(strlen(path) + 1);
    strcpy(dup, path);
    char * p;
    for(p = i + dup; p >= dup && *p != '/'; p --) {
        continue;
    }
    /* plain file name in current directory */
    if(p < dup) return;
    
    /* p == '/', so set it to NULL, dup is the dirname */
    *p = 0;
    _mkdir(dup);
}

static void 
_mkdir(const char *dir) 
{
    char * tmp = alloca(strlen(dir) + 1);
    strcpy(tmp, dir);
    char *p = NULL;
    size_t len;

    len = strlen(tmp);
    if(tmp[len - 1] == '/')
            tmp[len - 1] = 0;
    for(p = tmp + 1; *p; p++)
            if(*p == '/') {
                    *p = 0;
                    mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO);
                    *p = '/';
            }
    mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO);
}
