#include <math.h>
#include <string.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>

#include <fastpm/libfastpm.h>
#include <fastpm/prof.h>
#include <fastpm/lightcone_potential.h>
#include <fastpm/logging.h>
#include <fastpm/io.h>

#include "pmpfft.h"
#include "pmghosts.h"

void
fastpm_lcp_init(FastPMLightConeP * lcp, FastPMSolver * fastpm,
                double (*tileshifts)[3], int ntiles
                )
{
    gsl_set_error_handler_off(); // Turn off GSL error handler

    int lcp_np_ratio=ntiles;

//XXX Do we need these?
    lcp->p = malloc(sizeof(FastPMStore));//To store (matter) particles as they enter horizon
    /* for saving the density with particles */

    lcp->p0 = malloc(sizeof(FastPMStore));
    lcp->q = malloc(sizeof(FastPMStore));

    fastpm_store_init(lcp->p, fastpm->p->np_upper*lcp_np_ratio,
                  PACK_ID | PACK_POS | PACK_VEL
                | PACK_AEMIT
                | (fastpm->p->potential?PACK_POTENTIAL:0)
    );

    int read_ra_dec=lcp->read_ra_dec;

    /* Allocation */

    int size = 8192;

    lcp->EventHorizonTable.size = size;
    lcp->EventHorizonTable.Dc = malloc(sizeof(double) * size);
    lcp->EventHorizonTable.Growth = malloc(sizeof(double) * size);

    lcp->tileshifts = malloc(sizeof(tileshifts[0]) * ntiles);
    lcp->ntiles = ntiles;

    memcpy(lcp->tileshifts, tileshifts, sizeof(tileshifts[0]) * ntiles);
    /* GSL init solver */
    const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
    lcp->gsl = gsl_root_fsolver_alloc(T);

    int i;

    for(i = 0; i < lcp->EventHorizonTable.size; i ++) {
        double a = 1.0 * i / (lcp->EventHorizonTable.size - 1);
        lcp->EventHorizonTable.Dc[i] = lcp->speedfactor * HubbleDistance * ComovingDistance(a, lcp->cosmology);
        lcp->EventHorizonTable.Growth[i]=GrowthFactor(a, lcp->cosmology);//aemit[i];
    }

    if (read_ra_dec){

      int n_a=20;
      int n_a_subsample=0;
      double *a_subsample;
      double *r_subsample;
      double *a = malloc(sizeof(double)*n_a);
      double *r = malloc(sizeof(double)*n_a);

      double a_max=1;
      double a_min=.1;
      for (int i_a=0;i_a<n_a;i_a++)
        {
            a[i_a]=a_min+i_a*(a_max-a_min)/(n_a-1);
            r[i_a]=fastpm_lcp_horizon(lcp,a[i_a]);
            if (a[i_a]>lcp->subsample_a)
              n_a_subsample+=1;
        }

      size_t size= read_angular_grid(NULL, lcp->ra_dec_filename, r, a, n_a-n_a_subsample, 1,
        MPI_COMM_WORLD);
        size_t size2=0;
      if (n_a_subsample>0){
          size2=read_angular_grid(NULL, lcp->ra_dec_subsample_filename, r, a, n_a_subsample,
                          lcp->grid_subsample_factor, MPI_COMM_WORLD);
          a_subsample = malloc(sizeof(double)*n_a_subsample);
          r_subsample = malloc(sizeof(double)*n_a_subsample);
          for (int i_a=0;i_a<n_a_subsample;i_a++){
              a_subsample[i_a]=a[i_a+n_a-n_a_subsample];
              r_subsample[i_a]=r[i_a+n_a-n_a_subsample];
              fastpm_info("subsample i=%td,a=%g,r=%g\n",i_a,a_subsample[i_a],r_subsample[i_a]);
            }
      }
      fastpm_info("lightcone size=%td size2=%td\n",size,size2);
/*FIXME fudge factor for ghosts*/
      fastpm_store_init(lcp->p0,size+size2, PACK_POS
                    | PACK_POTENTIAL
                    | PACK_TIDAL
                    | PACK_AEMIT);
      /*XXX use the domain decompose */
      //double r[] = {0, 1, 2, 3, 4, 5, 6, 7};

      read_angular_grid(lcp->p0, lcp->ra_dec_filename, r, a, n_a-n_a_subsample, 1, MPI_COMM_WORLD);

      if (n_a_subsample>0){
            read_angular_grid(lcp->p0, lcp->ra_dec_subsample_filename, r_subsample, a_subsample, n_a_subsample, lcp->grid_subsample_factor, MPI_COMM_WORLD);
      }

      fastpm_store_init(lcp->q, lcp->p0->np_upper,   PACK_POS
                    | PACK_POTENTIAL
                    | PACK_TIDAL
                    | PACK_AEMIT);
      fastpm_info("%td max particles in the lightcone\n",lcp->q->np_upper);
    }

    else{
      lcp->p0 = malloc(sizeof(FastPMStore)*lcp_np_ratio);/*fixed grid*/
      lcp->q = malloc(sizeof(FastPMStore)*lcp_np_ratio);/*for storing fixed grid particles as they intersect lightcone*/
      if(lcp->compute_potential) {
          /* p0 is the lagrangian _position */
          fastpm_store_init(lcp->p0, fastpm->p->np_upper*lcp_np_ratio,
                PACK_POS
              | PACK_POTENTIAL
              | PACK_TIDAL
          );

          fastpm_store_set_lagrangian_position(lcp->p0, fastpm->basepm, NULL, NULL);

          /* for saving the lagrangian sampling of potential */
          fastpm_store_init(lcp->q, fastpm->p->np_upper*lcp_np_ratio,
                PACK_POS
              | PACK_POTENTIAL
              | PACK_TIDAL
              | PACK_AEMIT);
            }
    }

    lcp->interp_q_indx=malloc(sizeof(ptrdiff_t) * lcp->q->np_upper);
    lcp->a_now=-1;
    lcp->a_prev=-1;
    lcp->G_prev=0;
    lcp->G_now=0;
    lcp->interp_start_indx=0;
    lcp->interp_stop_indx=0;

    static int handler_i= 0;
    if (handler_i==0){/*to avoid adding multiple handlers if lightcone is initialized many times*/
      fastpm_solver_add_event_handler(fastpm, FASTPM_EVENT_FORCE,
              FASTPM_EVENT_STAGE_AFTER,
              (FastPMEventHandlerFunction) fastpm_lcp_compute_potential,
              lcp);

      fastpm_solver_add_event_handler(fastpm, FASTPM_EVENT_TRANSITION,
                      FASTPM_EVENT_STAGE_AFTER,
                      (FastPMEventHandlerFunction) fastpm_lcp_intersect,
                      lcp);
        handler_i++;
      }
}

double(* fastpm_lcp_tile(FastPMSolver *fastpm, int tile_x, int tile_y, int tile_z, int * ntiles, double (*tiles)[3]))[3]{
  *ntiles=(tile_x*2+1)*(tile_y*2+1)*(tile_z*2+1);
  tiles=malloc(sizeof(tiles[0]) * (*ntiles));

    int nt=0;
    for(int i_x=-tile_x;i_x<=tile_x;i_x++){
      for(int i_y=-tile_y;i_y<=tile_y;i_y++){
        for(int i_z=-tile_z;i_z<=tile_z;i_z++){
          tiles[nt][0]=i_x*pm_boxsize(fastpm->basepm)[0];
          tiles[nt][1]=i_y*pm_boxsize(fastpm->basepm)[1];
          tiles[nt][2]=i_z*pm_boxsize(fastpm->basepm)[2];
          nt++;
        }
      }
    }
    return tiles;
}

void
fastpm_lcp_destroy(FastPMLightConeP * lcp)
{
    /* Free */
    fastpm_store_destroy(lcp->p);

    if(lcp->compute_potential) {
        fastpm_store_destroy(lcp->q);
        fastpm_store_destroy(lcp->p0);
    }
    free(lcp->EventHorizonTable.Dc);
    free(lcp->EventHorizonTable.Growth);
    free(lcp->tileshifts);
    free(lcp->p0);
    free(lcp->q);
    free(lcp->p);
    free(lcp->interp_q_indx);
    /* GSL destroy solver */
    gsl_root_fsolver_free(lcp->gsl);
}

/*FIXME: Unnecessary Code duplication here. Maybe make this function agnostic to which lightcone */
double
fastpm_lcp_horizon(FastPMLightConeP * lcp, double a)
{
    /* It may be worth to switch to log_a interpolation, but it only matters
     * at very high z (~ z = 9). */

    double x = a * (lcp->EventHorizonTable.size - 1);
    int l = floor(x);
    int r = l + 1;
    if(r >= lcp->EventHorizonTable.size) {
        return lcp->EventHorizonTable.Dc[lcp->EventHorizonTable.size - 1];
    }
    if(l <= 0) {
        return lcp->EventHorizonTable.Dc[0];
    }
    return lcp->EventHorizonTable.Dc[l] * (r - x)
         + lcp->EventHorizonTable.Dc[r] * (x - l);
}

double
fastpm_lcp_growth(FastPMLightConeP * lcp, double a)
{
    /* It may be worth to switch to log_a interpolation, but it only matters
     * at very high z (~ z = 9). */

    double x = a * (lcp->EventHorizonTable.size - 1);
    int l = floor(x);
    int r = l + 1;
    if(r >= lcp->EventHorizonTable.size||l<=0) {
        return GrowthFactor(a, lcp->cosmology);
    }

    return lcp->EventHorizonTable.Growth[l] * (r - x)
         + lcp->EventHorizonTable.Growth[r] * (x - l);
}


struct funct_params {/*FIXME *Is it ok to use same name in 2 files?*/
    FastPMLightConeP *lcp;
    FastPMStore * p;
    //FastPMDriftFactor * drift; /*TODO: not needed??*/
    ptrdiff_t i;
    double tileshift[4];
    double a1;
    double a2;
};


/*FIXME: Code duplication*/
static void
gldot(double glmatrix[4][4], double xi[4], double xo[4])
{
    int i, j;
    for(i = 0; i < 4; i ++) {
        xo[i] = 0;
        for(j = 0; j < 4; j ++) {
            xo[i] += glmatrix[i][j] * xi[j];
        }
    }
}

static void
gldotv(double glmatrix[4][4], float vi[3], float vo[3])
{
    int i, j;
    for(i = 0; i < 3; i ++) {
        vo[i] = 0;
        for(j = 0; j < 3; j ++) {
            vo[i] += glmatrix[i][j] * vi[j];
        }
    }
}

static double
funct(double a, void *params)
{
    struct funct_params *Fp = (struct funct_params *) params;

    FastPMLightConeP *lcp = Fp->lcp;
    FastPMStore * p = Fp->p;
    //FastPMDriftFactor * drift = Fp->drift;
    ptrdiff_t i = Fp->i;
    int d;
    double xi[4];
    double xo[4];

    xi[3] = 1;
    // if(p->v) {
    //     fastpm_drift_one(drift, p, i, xi, a);
    // } else {
    for(d = 0; d < 3; d ++) {
            xi[d] = p->x[i][d];
    }
    //}
    for(d = 0; d < 4; d ++) {
        xi[d] += Fp->tileshift[d];
    }
    /* transform the coordinate */
    gldot(lcp->glmatrix, xi, xo);

    /* XXX: may need to worry about periodic boundary */
    double distance;
    if (lcp->fov <= 0) {
        distance = xo[2];
    } else {
        distance = 0;
        for (d = 0; d < 3; d ++) {
            distance += xo[d] * xo[d];
        }
        distance = sqrt(distance);
    }

    return distance - fastpm_lcp_horizon(lcp, a);
}

/*Potential interpolation. Required even when using ra-dec grid*/
int fastpm_lcp_compute_final_potential(FastPMLightConeP * lcp, FastPMForceEvent * event){
  double potfactor = 1.5 * lcp->cosmology->OmegaM / (HubbleDistance * HubbleDistance);

  FastPMStore *p0=lcp->p0;
  FastPMStore *pout=lcp->q;

  ptrdiff_t *indxs=lcp->interp_q_indx;

  lcp->a_now=event->a_f;

  double ai=lcp->a_prev;
  double af=lcp->a_now;

  lcp->G_now=GrowthFactor(af, lcp->cosmology);
  //lcp->G_now=fastpm_lcp_growth( lcp, af);
  double Gi=lcp->G_prev;
  double Gf=lcp->G_now;
  double Gemit=0;

  fastpm_info("%td particles are in the uniform light cone\n", lcp->q->np);
  fastpm_info("Potential interpolation, ai,af,Gi,Gf, start_indx, stop_indx: %g %g %g %g %td %td \n",ai,af,Gi,Gf,lcp->interp_start_indx,lcp->interp_stop_indx);

/*XXX Parallelize if too slow*/
  if (ai>0){
    for(ptrdiff_t i=lcp->interp_start_indx;i<lcp->interp_stop_indx;i++){
         //Gemit=GrowthFactor(pout->aemit[i], lcp->cosmology);//aemit[i];
         Gemit=fastpm_lcp_growth( lcp, pout->aemit[i]);

         //pout->potential[i] =p0->potential[indxs[i]]/ pout->aemit[i] * potfactor;

           pout->potential[i] =pout->potential[i]
                        +(p0->potential[indxs[i]]/ pout->aemit[i] * potfactor
                            -pout->potential[i])/(Gf-Gi)*(Gemit-Gi);
    }
  }
  lcp->interp_start_indx=lcp->interp_stop_indx;
  lcp->a_prev=lcp->a_now;
  lcp->G_prev=lcp->G_now;
}


int
fastpm_lcp_compute_potential(FastPMSolver * fastpm,
        FastPMForceEvent * event,
        FastPMLightConeP * lcp)
{
    PM * pm = fastpm->pm;
    FastPMFloat * delta_k = event->delta_k;
    FastPMFloat * canvas = pm_alloc(pm);/*Allocates memory and returns success*/
    FastPMGravity * gravity = fastpm->gravity;
    FastPMPainter reader[1];
    fastpm_painter_init(reader, pm, gravity->PainterType, gravity->PainterSupport);

    FastPMStore * p = lcp->p0;

/*XXX Following is almost a repeat of potential calc in fastpm_gravity_calculate, though positions are different*/
    int d;
    int ACC[] = {
                 PACK_POTENTIAL,
                 PACK_TIDAL_XX, PACK_TIDAL_YY, PACK_TIDAL_ZZ,
                 PACK_TIDAL_XY, PACK_TIDAL_YZ, PACK_TIDAL_ZX
                };

    PMGhostData * pgd = pm_ghosts_create(pm, p, PACK_POS, NULL);

    for(d = 0; d < 7; d ++) {
        CLOCK(transfer);
        gravity_apply_kernel_transfer(gravity, pm, delta_k, canvas, d+3);
        LEAVE(transfer);

        CLOCK(c2r);
        pm_c2r(pm, canvas);
        LEAVE(c2r);

        CLOCK(readout);
        fastpm_readout_local(reader, canvas, p, p->np + pgd->nghosts, NULL, ACC[d]);
        LEAVE(readout);

        CLOCK(reduce);
        pm_ghosts_reduce(pgd, ACC[d]);
        LEAVE(reduce);

    }
    ptrdiff_t i=lcp->interp_start_indx;
    fastpm_info("potential_intp before i,p %td,%g\n",i, lcp->q->potential[i]);

    fastpm_lcp_compute_final_potential(lcp,event);
      fastpm_info("potential_intp after i,p %td,%g\n",i, lcp->q->potential[i]);

    pm_free(pm, canvas);

    pm_ghosts_free(pgd);

    return 0;
}


static int
_fastpm_lcp_intersect_one(FastPMLightConeP * lcp,
        struct funct_params * params,
        ptrdiff_t i,
        double * solution)
{
    params->i = i;

    int status;
    int iter = 0, max_iter;
    double r, x_lo, x_hi, eps;

    /* Reorganize to struct later */
    x_lo = params->a1;
    x_hi = params->a2;
    max_iter = 100;
    eps = 1e-7;

    gsl_function F;

    F.function = &funct;
    F.params = params;

    status = gsl_root_fsolver_set(lcp->gsl, &F, x_lo, x_hi);

    if(status == GSL_EINVAL || status == GSL_EDOM) {
        /** Error in value or out of range **/
        return 0;
    }

    do
    {
        iter++;
        //
        // Debug printout #1
        //if(iter == 1) {
        //fastpm_info("ID | [x_lo, x_hi] | r | funct(r) | x_hi - x_lo\n");
        //}
        //

        status = gsl_root_fsolver_iterate(lcp->gsl);
        r = gsl_root_fsolver_root(lcp->gsl);

        x_lo = gsl_root_fsolver_x_lower(lcp->gsl);
        x_hi = gsl_root_fsolver_x_upper(lcp->gsl);

        status = gsl_root_test_interval(x_lo, x_hi, eps, 0.0);
        //
        //Debug printout #2
        //fastpm_info("%5d [%.7f, %.7f] %.7f %.7f %.7f\n", iter, x_lo, x_hi, r, funct(r, &params), x_hi - x_lo);
        //

        if(status == GSL_SUCCESS) {
            *solution = r;
            //
            // Debug printout #3.1
            //fastpm_info("fastpm_lcp_intersect() called with parameters %.7f and %.7f, returned status %d.\n\n", a, b, 1);
            //
            return 1;
        }
    }
    while (status == GSL_CONTINUE && iter < max_iter);
    //
    // Debug printout #3.2
    //fastpm_info("fastpm_lcp_intersect() called with parameters %.7f and %.7f, returned status %d.\n\n", a, b, 0);
    //

    return 0;
}

static double
zangle(double * x) {
    double dxy = 0;
    double dz = x[2];
    dxy = x[0] * x[0] + x[1] * x[1];

    return atan2(sqrt(dxy), dz) / M_PI * 180.;
}

/* FIXME:
 * the function shall take ai, af as input,
 *
 * the function shall be able to interpolate potential as
 * well as position and velocity.
 * We need a more general representation of '*drift' and '*kick'.
 *
 * */
static int
fastpm_lcp_intersect_tile_grid(FastPMLightConeP * lcp, int tile,
        FastPMTransitionEvent *event,
        FastPMStore * p,
        FastPMStore * pout
)
{
    struct funct_params params = {
        .lcp = lcp,
        .p = p,
        .i = 0,
        .a1 = event->transition->a.i,//drift->ai > drift->af ? drift->af: drift->ai,
        .a2 = event->transition->a.r//drift->ai > drift->af ? drift->ai: drift->af,
        /*XXX What is a.r?*/
    };
    int d;

    for(d = 0; d < 3; d ++) {
        params.tileshift[d] = lcp->tileshifts[tile][d];
    }
    fastpm_info("tileshift = %g %g %g\n",
        params.tileshift[0],
        params.tileshift[1],
        params.tileshift[2]);

    params.tileshift[3] = 0;

    ptrdiff_t i;

    for(i = 0; i < p->np; i ++) {
        double a_emit = 0;
        if (lcp->read_ra_dec==0){/*XXX This doesnot give exact answer when a_emit is known for ra_dec grid*/
          if(0 == _fastpm_lcp_intersect_one(lcp, &params, i, &a_emit)) continue;
        }
        else{
          a_emit=p->aemit[i];
          if(a_emit>params.a2||a_emit<params.a1)continue;
        }
        /* A solution is found */
        /* move the particle and store it. */
        ptrdiff_t next = pout->np;
        if(next == pout->np_upper) {
            fastpm_raise(-1, "Too many particles in the light cone next=%td, max=%td, aemit=%g \n",
                    next,pout->np_upper,a_emit);
        }
        double xi[4];
        double xo[4];
        int d;

        xi[3] = 1;

            for(d = 0; d < 3; d ++) {
                xi[d] = p->x[i][d];
            }

        for(d = 0; d < 4; d ++) {
            xi[d] += params.tileshift[d];
        }
        /* transform the coordinate */
        gldot(lcp->glmatrix, xi, xo);

        /* does it fall into the field of view? */
        if(lcp->fov > 0 && zangle(xo) > lcp->fov * 0.5) continue;

        /* copy the position if desired */
        if(pout->x) {
            for(d = 0; d < 3; d ++) {
                pout->x[next][d] = xo[d];
            }
        }

        if(pout->aemit)
            pout->aemit[next] = a_emit;

        double potfactor = 1.5 * lcp->cosmology->OmegaM / (HubbleDistance * HubbleDistance);
        /* convert to dimensionless potential */
        if(pout->potential)
            pout->potential[next] = p->potential[i] / a_emit * potfactor;

        if(pout->tidal) {
            for(d = 0; d < 6; d++) {
                pout->tidal[next][d] = p->tidal[i][d] / a_emit * potfactor;
            }
        }
        lcp->interp_q_indx[next]=i;
        pout->np ++;
        lcp->interp_stop_indx=next;
    }
    return 0;
}

int
fastpm_lcp_intersect(FastPMSolver * fastpm, FastPMTransitionEvent *event, FastPMLightConeP * lcp)
{
  fastpm_info("fastpm_lcp_intersect, action ai,ar,af: %d %g %g %g \n", event->transition->action, event->transition->a.i,event->transition->a.r,event->transition->a.f);

    if (event->transition->action!=FASTPM_ACTION_FORCE)
    {
      return 0;
      /*FIXME we donot nned to intersect LC after every transition, may even lead to particle
              duplication. Decide which ones we need.*/
    }
    /* for each tile */
    int t;
    for(t = 0; t < lcp->ntiles; t ++) {
        // fastpm_lcp_intersect_tile(lcp, t, event, fastpm->p, lcp->p);/*Store particle to get
        //                                                           density*/

        if(lcp->compute_potential)
            fastpm_lcp_intersect_tile_grid(lcp, t, event, lcp->p0, lcp->q);/*store potential on fixed
                                                                        grid*/
    }
    return 0;
}
