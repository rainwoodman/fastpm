#ifndef __FASTPM_IO_H__
#define __FASTPM_IO_H__
#include <bigfile.h>
FASTPM_BEGIN_DECLS

typedef void (*FastPMSnapshotSorter)(const void * ptr, void * radix, void * arg);

void
FastPMSnapshotSortByID(const void * ptr, void * radix, void * arg);

void
FastPMSnapshotSortByLength(const void * ptr, void * radix, void * arg);

void
FastPMSnapshotSortByAEmit(const void * ptr, void * radix, void * arg);

int
write_snapshot(FastPMSolver * fastpm,
        FastPMStore * p,
        const char * filebase,
        const char * parameters,
        int Nwriters,
        FastPMSnapshotSorter sorter
    );

int
append_snapshot(FastPMSolver * fastpm,
        FastPMStore * p,
        const char * filebase,
        const char * parameters,
        int Nwriters,
        FastPMSnapshotSorter sorter
    );

int
write_snapshot_data(FastPMStore * p,
        int Nfile,
        int Nwriters,
        FastPMSnapshotSorter sorter,
        int append,
        BigFile * bf,
        MPI_Comm comm
);

void
write_snapshot_header(FastPMSolver * fastpm, FastPMStore * p,
    const char * parameters, BigFile * bf, MPI_Comm comm);

int
read_snapshot(FastPMSolver * fastpm, FastPMStore * p, const char * filebase);

int
write_complex(PM * pm, FastPMFloat * data, const char * filename, const char * blockname, int Nwriters);

int
read_complex(PM * pm, FastPMFloat * data, const char * filename, const char * blockname, int Nwriters);

size_t
read_angular_grid(FastPMStore * store,
        const char * filename,
        const double * r,
        const double * aemit,
        const size_t Nr,
        int sampling_factor,
        MPI_Comm comm);

FASTPM_END_DECLS
#endif
