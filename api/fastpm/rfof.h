#ifndef _FASTPM_RFOF_H
#define _FASTPM_RFOF_H

/* Relaxed FOF, Biwei Dai et al 2019.*/

typedef struct FastPMRFOFFinderPrivate FastPMRFOFFinderPrivate;

typedef struct {
    int nmin;
    double linkinglength; /* absolute */
    double l1; /* absolute */
    double l6; /* absolute */
    double A1;
    double A2;
    double B1;
    double B2;

    int periodic;
    int kdtree_thresh;

    FastPMStore * p;
    PM * pm;

    /* private */
    uint64_t * label;

    FastPMRFOFFinderPrivate * priv;

    FastPMEventHandler * event_handlers;
} FastPMRFOFFinder;

void
fastpm_rfof_init(FastPMRFOFFinder * finder,
    FastPMCosmology * cosmology,
    FastPMStore * store,
    PM * pm);

/* create a halo catalog from the heap. halos->name shall be set before this */
void
fastpm_rfof_execute(FastPMRFOFFinder * finder,
    FastPMStore * halos,
    ptrdiff_t ** ihalo, double z);

void
fastpm_rfof_destroy(FastPMRFOFFinder * finder);

#endif
