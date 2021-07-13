#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include <fastpm/libfastpm.h>
#include <fastpm/prof.h>
#include <fastpm/logging.h>
#include <fastpm/store.h>

#include <fastpm/fof.h>
#include <fastpm/rfof.h>
#include "pmghosts.h"

//#define FASTPM_FOF_DEBUG

struct FastPMRFOFFinderPrivate {
    int ThisTask;
    int NTask;
    double * boxsize;
    MPI_Comm comm;
    int Np[7];
    FastPMCosmology cosmology[1];
    ptrdiff_t * ihalo;
};

void
fastpm_rfof_init(FastPMRFOFFinder * finder,
    FastPMCosmology * cosmology,
    FastPMStore * p,
    PM * pm)
{
    finder->priv = malloc(sizeof(FastPMRFOFFinderPrivate));
    finder->p = p;
    finder->pm = pm;

    finder->event_handlers = NULL;

    if (finder->periodic)
        finder->priv->boxsize = pm_boxsize(pm);
    else
        finder->priv->boxsize = NULL;

    /* In the paper, i starts from 1.*/
    finder->priv->Np[0] = 0;
    finder->priv->Np[1] = 20;
    finder->priv->Np[2] = 40;
    finder->priv->Np[3] = 80;
    finder->priv->Np[4] = 160;
    finder->priv->Np[5] = 320;
    finder->priv->Np[6] = 1<<30; /* inf, eq 2.2 */

    finder->priv->comm = pm_comm(pm);
    MPI_Comm comm = finder->priv->comm;
    MPI_Comm_rank(comm, &finder->priv->ThisTask);
    MPI_Comm_size(comm, &finder->priv->NTask);

    finder->priv->cosmology[0] = cosmology[0];

}

/* linking length for bin i and redshift z. eq 2.3-2.6*/
static double
_fastpm_rfof_get_linkinglength(FastPMRFOFFinder * finder, int i, double z)
{
    if(i == 1) {
        return finder->l1 - finder->A1 / (1 + z);
    }

    if(i == 6) {
        return fmax(finder->l6 - finder->A2 / (1 + z), finder->linkinglength);
    }

    return ((6 - i) * _fastpm_rfof_get_linkinglength(finder, 1, z)
         + (i - 1) * _fastpm_rfof_get_linkinglength(finder, 6, z)) / 5.0;
}

static double
_fastpm_rfof_get_rejection(FastPMRFOFFinder * finder, double z)
{
    return finder->B1 - finder->B2 * log(1 + z);
}

static double
_std_vdisp(double M, double Ez) {
    const double V0 = 1100;
    const double M0 = 1e15;
    return pow(Ez * M / M0, 1 / 3.) * V0;
}

void
fastpm_rfof_execute(FastPMRFOFFinder * finder,
    FastPMStore * halos,
    ptrdiff_t ** ihalo, double z)
{
    int i;
    double Ez = HubbleEa(1 / (z + 1), finder->priv->cosmology);

    ptrdiff_t * icandidate;
    FastPMStore candidates[1];

    fastpm_fof_allocate_halos(halos, finder->p->np / 10, finder->p, finder->priv->boxsize != NULL, finder->priv->comm);
    halos->np = 0;

    finder->priv->ihalo = fastpm_memory_alloc(finder->p->mem,
                "ihalo", sizeof(ptrdiff_t) * finder->p->np,
                FASTPM_MEMORY_STACK);

    FastPMFOFFinder fof = {
        .periodic = finder->periodic,
        .nmin = finder->nmin,
        .kdtree_thresh = finder->kdtree_thresh,
    };
    fastpm_fof_init(&fof, fmax(finder->l1, finder->l6), finder->p, finder->pm);

    FastPMParticleMaskType * active = fastpm_memory_alloc(finder->p->mem,
                    "active",
                    sizeof(active[0]) * finder->p->np,
                    FASTPM_MEMORY_STACK);

    ptrdiff_t j;
    for(j = 0; j < finder->p->np; j++) {
        active[j] = 1;
    }

    for(i = 1; i <= 6; i ++) {
        double ll = _fastpm_rfof_get_linkinglength(finder, i, z);
        fastpm_info("RFOF: FOF with linking length %g (Mpc/h), bin = %d, z= %0.3f, Np=%d", ll, i, z, finder->priv->Np[i]);

        fastpm_store_set_name(candidates, "candidates");
        fastpm_fof_execute(&fof, ll, candidates, &icandidate, active);

        FastPMParticleMaskType * save_mask = fastpm_memory_alloc(finder->p->mem,
                        "SaveMask",
                        sizeof(save_mask[0]) * halos->np,
                        FASTPM_MEMORY_STACK);

        ptrdiff_t j;
        for(j = 0; j < candidates->np; j ++) {
            double vdisp = 0;
            int d;
            for(d = 0; d < 3; d ++) {
                vdisp += candidates->vdisp[j][d];
            }
            /* do not divide by sqrt(3). Total velocity is sum(vi**2) */
            vdisp = sqrt(vdisp);

            double M = candidates->meta.M0 * 1e10 * candidates->length[j];
            double r0 = _fastpm_rfof_get_rejection(finder, z);

            save_mask[j] = candidates->length[j] < finder->priv->Np[i]
                        && vdisp < r0 * _std_vdisp(M, Ez);
        }
        for(j = 0; j < finder->p->np; j ++) {
            if (!active[j]) continue;
            if (icandidate[j] < 0) {
                /* not associated with a halo */
                active[j] = 0;
                continue;
            }
        }
        /* remove halos not to be saved. */
        fastpm_fof_subsample_and_relabel(&fof, candidates, save_mask, icandidate);

        size_t nactive = 0;
        for(j = 0; j < finder->p->np; j ++) {
            if (!active[j]) continue;
            if (icandidate[j] >= 0) {
                /* already saved as a halo, remove particle from next iteration. */
                active[j] = 0;
                finder->priv->ihalo[j] = icandidate[j] + halos->np;
                continue;
            }
            nactive ++;
        }
        fastpm_store_extend(halos, candidates);
        fastpm_info("RFOF: saved %td halos; total halos = %td.", candidates->np, halos->np);
        fastpm_info("RFOF: remaining active particles = %td.", nactive);

        fastpm_memory_free(finder->p->mem, save_mask);
        fastpm_store_destroy(candidates);
    }
    fastpm_memory_free(finder->p->mem, active);
    fastpm_fof_destroy(&fof);

    if(ihalo) {
        *ihalo = finder->priv->ihalo;
    } else {
        fastpm_store_subsample(halos, halos->mask, halos);
    }
}

void
fastpm_rfof_destroy(FastPMRFOFFinder * finder)
{
    fastpm_memory_free(finder->p->mem, finder->priv->ihalo);
    free(finder->priv);
}

