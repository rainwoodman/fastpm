/*******
 * Memory management.
 * From Jun Koda's PM utilities
 *
 * With a few fixes from Yu Feng <rainwoodman@gmail.com>
 **/
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <fftw3-mpi.h>
#include "mem.h"
#include "msg.h"


void allocate_shared_memory(const int nc, const int nc_factor, const double np_alloc_factor, Memory* const mem)
{
  // Allocate shared memory

  // mem1
  //  2LPT grids / PM density grid / FoF kdtree

  // Memory for 2LPT (6*np_local words)
  ptrdiff_t local_nx, local_x_start;
  ptrdiff_t size_lpt_one=
    fftwf_mpi_local_size_3d(nc, nc, nc / 2 + 1, MPI_COMM_WORLD,
			    &local_nx, &local_x_start);
  ptrdiff_t ncomplex_lpt= 12*size_lpt_one;

  //const int np_alloc= (int)(np_alloc_factor*nc*nc*(nx+1));
  const int np_alloc= (int)(np_alloc_factor*nc*nc*(local_nx+1));

  msg_printf(verbose, "%d Mbytes requested for LPT\n",
	     (int)(ncomplex_lpt*sizeof(fftwf_complex)/(1024*1024)));

  
  // Memory for PM (nc_factor^3 * np_local each)
  const int Ngrid= nc_factor*nc;
  ptrdiff_t local_ny, local_y_start;
  ptrdiff_t size_pm_one= 
    fftwf_mpi_local_size_3d_transposed(Ngrid, Ngrid, Ngrid / 2 + 1, MPI_COMM_WORLD,
	                 &local_nx, &local_x_start, &local_ny, &local_y_start);
  ptrdiff_t ncomplex_pm= size_pm_one;

  msg_printf(verbose, "%d Mbytes requested for one PM grid\n",
	     (int)(ncomplex_pm*sizeof(fftwf_complex)/(1024*1024)));

  msg_printf(verbose, "PM size %d %d %d\n", size_pm_one, local_nx*Ngrid*Ngrid, local_nx);

  ptrdiff_t ncomplex1= ncomplex_lpt > ncomplex_pm ? ncomplex_lpt : ncomplex_pm;
  size_t size1= sizeof(fftwf_complex)*ncomplex1;

  mem->mem1= fftwf_alloc_complex(ncomplex1);
  mem->size1= sizeof(fftwf_complex)*ncomplex1;

  if(mem->mem1 == 0)
    msg_abort(0050, "Error: Unable to allocate %d Mbytes for mem1\n",
	      (int)(mem->size1/(1024*1024)));
  
  // mem2
  // PM density_k mesh and snapshot
  size_t ncomplex2= (Ngrid/2+1)*Ngrid*local_ny; //ncomplex_pm;
  //size_t size2= sizeof(fftwf_complex)*size_pm_one; 
  size_t size2= sizeof(fftwf_complex)*(Ngrid/2+1)*Ngrid*local_ny;

  msg_printf(verbose, "%d Mbytes requested for delta_k mesh (mem2). ny=%d\n",
	     (int)(size2/(1024*1024)), local_ny);

  size_t size_snapshot= sizeof(ParticleMinimum)*np_alloc;
  if(size_snapshot > size2) {
    msg_printf(verbose, "%d Mbytes requested for snapshot in mem2\n",
	       (int)(size_snapshot/(1024*1024)));

    size2= size_snapshot;
    ncomplex2= size_snapshot/sizeof(fftwf_complex) + 1;
  }

  mem->mem2= fftwf_alloc_complex(ncomplex2);
  mem->size2= sizeof(fftwf_complex)*ncomplex2;

  if(mem->mem2 == 0)
    msg_abort(0060, "Error: Unable to allocate %d + %d Mbytes for mem1&2.\n",
	      (int)(mem->size1/(1024*1024)), (int)(mem->size2/(1024*1024)));

  msg_printf(info, "%d Mbytes allocated for mem1.\n", 
	     (int)(mem->size1/(1024*1024)));
  msg_printf(info, "%d Mbytes allocated for mem2.\n", 
	     (int)(mem->size2/(1024*1024)));
}
