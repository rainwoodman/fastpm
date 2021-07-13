#include <string.h>
#include <mpi.h>

#include <kdcount/kdtree.h>

#include <fastpm/libfastpm.h>

#include <fastpm/prof.h>
#include <fastpm/logging.h>
#include <fastpm/store.h>

#include <fastpm/fof.h>
#include "pmghosts.h"

//#define FASTPM_FOF_DEBUG

struct FastPMFOFFinderPrivate {
    int ThisTask;
    int NTask;
    double * boxsize;
    MPI_Comm comm;
    PMGhostData * pgd;
    ptrdiff_t * head;
    KDTree tree;
    KDNode * root;
};

/* creating a kdtree struct
 * for store with np particles starting from start.
 * */
struct KDTreeNodeBuffer {
    void * mem;
    void * base;
    char * ptr;
    char * end;
    struct KDTreeNodeBuffer * prev;
};


static void *
_kdtree_buffered_malloc(void * userdata, size_t size)
{
    struct KDTreeNodeBuffer ** pbuffer = (struct KDTreeNodeBuffer**) userdata;

    struct KDTreeNodeBuffer * buffer = *pbuffer;

    if(buffer->base == NULL || buffer->ptr + size >= buffer->end) {
        struct KDTreeNodeBuffer * newbuffer = malloc(sizeof(newbuffer[0]));
        newbuffer->mem = buffer->mem;
        size_t newsize = 1024 * 1024 * 4; /* 4 MB for each block */
        if(newsize < size) {
            newsize = size;
        }
        newbuffer->base = fastpm_memory_alloc(buffer->mem, "KDTreeBase", newsize, FASTPM_MEMORY_STACK);
        newbuffer->ptr = newbuffer->base;
        newbuffer->end = newbuffer->base + newsize;
        newbuffer->prev = buffer;

        *pbuffer = newbuffer;
        buffer = newbuffer;
    }

    void * r = buffer->ptr;
    buffer->ptr += size;
    return r;
}

static void
_kdtree_buffered_free(void * userdata, size_t size, void * ptr)
{
    /* do nothing; */
}

static 
KDNode *
_create_kdtree (KDTree * tree, int thresh,
    FastPMStore ** stores, int nstore,
    double boxsize[], int use_mask)
{
    /* if boxsize is NULL the tree will be non-periodic. */
    /* the allocator; started empty. */
    struct KDTreeNodeBuffer ** pbuffer = malloc(sizeof(void*));
    struct KDTreeNodeBuffer * headbuffer = malloc(sizeof(headbuffer[0]));
    *pbuffer = headbuffer;

    headbuffer->mem = stores[0]->mem;
    headbuffer->base = NULL;
    headbuffer->prev = NULL;

    int s;
    ptrdiff_t i;

    tree->userdata = pbuffer;

    tree->input.dims[0] = 0;

    for(s = 0; s < nstore; s ++) {
        tree->input.dims[0] += stores[s]->np;
    }

    tree->input.dims[1] = 3;

    FastPMParticleMaskType * active = NULL;
    if(tree->input.dims[0] < stores[0]->np_upper) {
        /* if the first store is big enough, use it for the tree */
        tree->input.buffer = (void*) &stores[0]->x[0][0];
        if(use_mask) {
            active = &stores[0]->mask[0];
        }
    } else {
        /* otherwise, allocate a big buffer and make a copy */
        tree->input.buffer = _kdtree_buffered_malloc(pbuffer,
                    tree->input.dims[0] * sizeof(stores[0]->x[0]));
        memcpy(tree->input.buffer, &stores[0]->x[0][0], stores[0]->np * sizeof(stores[0]->x[0]));
        if(use_mask) {
            active = _kdtree_buffered_malloc(pbuffer,
                        tree->input.dims[0] * sizeof(active[0]));
            memcpy(active, &stores[0]->mask[0], stores[0]->np * sizeof(stores[0]->mask[0]));
        }
    }

    i = stores[0]->np;

    /* copy the other positions to the base pointer. */
    for(s = 1; s < nstore; s ++) {
        memcpy(((char*) tree->input.buffer) + i * sizeof(stores[0]->x[0]),
                &stores[s]->x[0][0],
                stores[s]->np * sizeof(stores[0]->x[0]));
        if(use_mask) {
            memcpy(((char*) active) + i * sizeof(stores[0]->mask[0]),
                &stores[s]->mask[0],
                stores[s]->np * sizeof(stores[0]->mask[0])
            );
        }
        i = i + stores[s]->np;
    }

    tree->input.strides[0] = sizeof(stores[0]->x[0]);
    tree->input.strides[1] = sizeof(stores[0]->x[0][0]);
    tree->input.elsize = sizeof(stores[0]->x[0][0]);
    tree->input.cast = NULL;

    tree->ind = _kdtree_buffered_malloc(pbuffer, tree->input.dims[0] * sizeof(tree->ind[0]));
    ptrdiff_t j;
    for(i = 0, j = 0; i < tree->input.dims[0]; i ++) {
        if(!use_mask || active[i]) {
            tree->ind[j] = i;
            j++;
        }
    }
    tree->ind_size = j;

    tree->malloc = _kdtree_buffered_malloc;
    tree->free = _kdtree_buffered_free;

    tree->thresh = thresh;

    tree->boxsize = boxsize;

    KDNode * root = kd_build(tree);
    fastpm_info("Creating KDTree with %td nodes for %td particles\n", tree->size, tree->ind_size);
    return root;
}

void 
_free_kdtree (KDTree * tree, KDNode * root)
{
    kd_free(root);
    struct KDTreeNodeBuffer * buffer, *q, **pbuffer = tree->userdata;

    for(buffer = *pbuffer; buffer; buffer = q) {
        if(buffer->base)
            fastpm_memory_free(buffer->mem, buffer->base);
        q = buffer->prev;
        free(buffer);
    }
    free(pbuffer);
}

static int
FastPMTargetFOF(FastPMStore * store, ptrdiff_t i, void * userdata);

void
fastpm_fof_init(FastPMFOFFinder * finder, double max_linkinglength,
                FastPMStore * p, PM * pm)
{
    finder->priv = malloc(sizeof(FastPMFOFFinderPrivate));
    finder->p = p;
    finder->pm = pm;

    if (finder->periodic)
        finder->priv->boxsize = pm_boxsize(pm);
    else
        finder->priv->boxsize = NULL;

    finder->priv->comm = pm_comm(pm);
    MPI_Comm comm = finder->priv->comm;
    MPI_Comm_rank(comm, &finder->priv->ThisTask);
    MPI_Comm_size(comm, &finder->priv->NTask);

    /* only do wrapping for periodic data */
    if(finder->priv->boxsize)
        fastpm_store_wrap(p, finder->priv->boxsize);

    double npmax, npmin, npstd, npmean;

    MPIU_stats(comm, p->np, "<->s", &npmin, &npmean, &npmax, &npstd);

    fastpm_info("load balance before fof decompose : min = %g max = %g mean = %g std = %g\n",
        npmin, npmax, npmean, npstd
        );

#if 1
    /* still route particles to the pm pencils as if they are periodic. */
    /* should still work (albeit use crazy memory) if we skip this. */
    if(0 != fastpm_store_decompose(p,
                (fastpm_store_target_func) FastPMTargetFOF, pm, comm)
    ) {
        fastpm_raise(-1, "out of storage space decomposing for FOF\n");
    }
#endif
    MPIU_stats(comm, p->np, "<->s", &npmin, &npmean, &npmax, &npstd);

    fastpm_info("load balance after fof decompose : min = %g max = %g mean = %g std = %g\n",
        npmin, npmax, npmean, npstd
        );

    /* create ghosts mesh size is usually > ll so we are OK here. */
    double below[3], above[3];

    int d;
    for(d = 0; d < 3; d ++) {
        /* bigger padding reduces number of iterations */
        below[d] = -max_linkinglength * 1;
        above[d] = max_linkinglength * 1;
    }

    PMGhostData * pgd = pm_ghosts_create_full(pm, p,
            COLUMN_MASK | COLUMN_POS | COLUMN_ID | COLUMN_MINID,
            below, above
        );

    pm_ghosts_send(pgd, COLUMN_POS);
    pm_ghosts_send(pgd, COLUMN_ID);

    finder->priv->pgd = pgd;

    size_t np_and_ghosts = p->np + pgd->p->np;
    finder->priv->head = fastpm_memory_alloc(p->mem, "FOFHead",
                    sizeof(finder->priv->head[0]) * np_and_ghosts, FASTPM_MEMORY_STACK);

    #ifdef FASTPM_FOF_DEBUG
    fastpm_ilog(INFO, "Rank %d has %td particles including ghost\n", finder->priv->ThisTask, np_and_ghosts);
    #endif

}


static int
_merge(uint64_t * src, ptrdiff_t isrc, uint64_t * dest, ptrdiff_t idest, ptrdiff_t * head)
{
    int merge = 0;
    ptrdiff_t j = head[idest];
    if(src[isrc] < dest[j]) {
        merge = 1;
    }

    if(merge) {
        dest[j] = src[isrc];
    }
    return merge;
}

struct reduce_fof_data {
    ptrdiff_t * head;
    size_t nmerged;
};

static void
FastPMReduceFOF(FastPMStore * src, ptrdiff_t isrc, FastPMStore * dest, ptrdiff_t idest, int ci, void * userdata)
{

    struct reduce_fof_data * data = userdata;

    data->nmerged += _merge(src->minid, isrc, dest->minid, idest, data->head);
}


static void
_fof_global_merge(
    FastPMFOFFinder * finder,
    FastPMStore * p,
    PMGhostData * pgd,
    uint64_t * minid,
    ptrdiff_t * head
)
{
    ptrdiff_t i;

    MPI_Comm comm = finder->priv->comm;

    size_t npmax = p->np;

    MPI_Allreduce(MPI_IN_PLACE, &npmax, 1, MPI_LONG, MPI_MAX, comm);

    /* initialize minid, used as a global tag of groups as we merge */
    for(i = 0; i < p->np; i ++) {
        /* assign unique ID to each particle; could use a better scheme with true offsets */
        minid[i] = i + finder->priv->ThisTask * npmax;
        #ifdef FASTPM_FOF_DEBUG
        /* for debugging, overwrite the previous unique ID with the true ID of particles */
        minid[i] = p->id[i];
        #endif
    }

    /* send minid */
    p->minid = minid; /* only send up to p->np */
    pm_ghosts_send(pgd, COLUMN_MINID);
    p->minid = NULL;

    /* copy over to minid for storage; FIXME: allow overriding  */
    for(i = 0; i < pgd->p->np; i ++) {
        minid[i + p->np] = pgd->p->minid[i];
    }

    /* reduce the minid of the head items according to the local connection. */

    while(1) {
        size_t nmerge = 0;
        for(i = 0; i < p->np + pgd->p->np; i ++) {
            nmerge += _merge(minid, i, minid, i, head);
        }
        if(nmerge == 0) break;
    }

#ifdef FASTPM_FOF_DEBUG
    {
        FILE * fp = fopen(fastpm_strdup_printf("dump-pos-%d.f8", finder->priv->ThisTask), "w");
        fwrite(p->x, p->np, sizeof(double) * 3, fp);
        fwrite(pgd->p->x, pgd->p->np, sizeof(double) * 3, fp);
        fclose(fp);
    }
    {
        FILE * fp = fopen(fastpm_strdup_printf("dump-id-%d.f8", finder->priv->ThisTask), "w");
        fwrite(p->id, p->np, sizeof(int64_t), fp);
        fwrite(pgd->p->id, pgd->p->np, sizeof(int64_t) * 3, fp);
        fclose(fp);
    }

#endif
    int iter = 0;

    while(1) {

        /* prepare the communication buffer, every ghost has
         * the minid and task of the current head. such that
         * they will connect to the other ranks correctly */

        for(i = 0; i < pgd->p->np; i ++) {
            pgd->p->minid[i] = minid[head[i + p->np]];
        }

        /* at this point all items on ghosts have local minid and task, ready to reduce */

        struct reduce_fof_data data = {
            .head = head,
            .nmerged = 0,
        };

        /* merge ghosts into the p, reducing the MINID on p */

        p->minid = minid; /* only update up to p->np */
        pm_ghosts_reduce(pgd, COLUMN_MINID, FastPMReduceFOF, &data);
        p->minid = NULL;

        size_t nmerged = data.nmerged;

        /* at this point all items on p->fof with head[i] = i have present minid and task */

        MPI_Allreduce(MPI_IN_PLACE, &nmerged, 1, MPI_LONG, MPI_SUM, comm);

        MPI_Barrier(comm);

        fastpm_info("FOF reduction iteration %d : merged %td crosslinks\n", iter, nmerged);

        if(nmerged == 0) break;

        for(i = 0; i < p->np + pgd->p->np; i ++) {
            if(minid[i] < minid[head[i]]) {
                fastpm_raise(-1, "p->fof invariance is broken i = %td np = %td\n", i, p->np);
            }
        }

        iter++;
    }

    /* previous loop only updated head[i]; now make sure every particles has the correct minid. */
    for(i = 0; i < p->np + pgd->p->np; i ++) {
        minid[i] = minid[head[i]];
    }

    #ifdef FASTPM_FOF_DEBUG
    {
    for(i = 0; i < p->np + pgd->p->np ; i ++) {
        uint64_t id = i <p->np?p->id[i]:pgd->p->id[i - p->np];
        if(minid[i] == 88) {
            fastpm_ilog(INFO, "%d MINID == %ld ID = %ld headminid == %ld i = %td / %td head=%td", finder->priv->ThisTask,
                minid[i], id, minid[head[i]], i, p->np, head[i]);
        }
        if(id == 88 || id == 96 || id == 152 || id == 160) {
            fastpm_ilog(INFO, "%d MINID == %ld ID = %ld headminid == %ld i = %td / %td head=%td", finder->priv->ThisTask,
                minid[i], id, minid[head[i]], i, p->np, head[i]);
        }
    }
    }
    #endif
}

/* set head[i] to hid*/
static size_t
_assign_halo_attr(FastPMFOFFinder * finder, PMGhostData * pgd, ptrdiff_t * head, size_t np, size_t np_ghosts, int nmin)
{
    ptrdiff_t * offset = fastpm_memory_alloc(finder->p->mem, "FOFOffset", sizeof(offset[0]) * (np + np_ghosts), FASTPM_MEMORY_STACK);
    uint8_t * has_remote = fastpm_memory_alloc(finder->p->mem, "FOFHasRemote", sizeof(has_remote[0]) * (np + np_ghosts), FASTPM_MEMORY_STACK);
    uint8_t * has_local = fastpm_memory_alloc(finder->p->mem, "FOFHasLocal", sizeof(has_local[0]) * (np + np_ghosts), FASTPM_MEMORY_STACK);

    ptrdiff_t i;
    for(i = 0; i < np + np_ghosts; i ++) {
        offset[i] = 0;
        has_remote[i] = 0;
        has_local[i] = 0;
    }

    /* set offset to number of particles in the halo */
    for(i = 0; i < np + np_ghosts; i ++) {
        offset[head[i]] ++;
    }

    for(i = 0; i < np; i ++) {
        has_local[head[i]] = 1;
    }

    /* if the group is connected to a remote component */
    /* if the particle has a ghost, then */
    pm_ghosts_has_ghosts(pgd, has_remote);

    /* if connected to a particle that has ghost */
    for(i = 0; i < np; i ++) {
        if(has_remote[i]) has_remote[head[i]] = 1;
    }

    /* if connected to a ghost */
    for(i = np; i < np + np_ghosts; i ++) {
        has_remote[head[i]] = 1;
    }

    size_t it = 0;

    /* assign attr index for groups at least contain 1 local particle */
    for(i = 0; i < np + np_ghosts; i ++) {
        /* only assign for the head */
        if(head[i] == i && has_local[i] && (has_remote[i] || offset[i] >= nmin)) {
            offset[i] = it;
            it ++;
        } else {
            offset[i] = -1;
        }
    }

    size_t nhalos = it;

    /* update head [i] to offset[head[i]], which stores the index in the halo store for this particle. */
    for(i = 0; i < np + np_ghosts; i ++) {
        head[i] = offset[head[i]];
        /* this will not happen if nmin == 1 */
        if(head[i] != -1 && head[i] > nhalos) {
            fastpm_raise(-1, "head[i] (%td) > nhalos (%td) This shall not happen.\n", head[i], nhalos);
        }
    }

    fastpm_memory_free(finder->p->mem, has_local);
    fastpm_memory_free(finder->p->mem, has_remote);
    fastpm_memory_free(finder->p->mem, offset);

    return it;
}
static double
periodic_add(double x, double wx, double y, double wy, double L)
{
    if(wx > 0) {
        while(y - x > L / 2) y -= L;
        while(y - x < -L / 2) y += L;

        return wx * x + wy * y;
    } else {
        return wy * y;
    }
}

/*
 * apply mask to a halo storage.
 * relable head, and halos->id
 * */
void
fastpm_fof_subsample_and_relabel(FastPMFOFFinder * finder,
    FastPMStore * halos,
    FastPMParticleMaskType * mask,
    ptrdiff_t * head)
{
    /* mapping goes from the old head[i] value to the new head[i] value */
    ptrdiff_t * mapping = fastpm_memory_alloc(finder->p->mem, "FOFMapping", sizeof(mapping[0]) * halos->np, FASTPM_MEMORY_STACK);

    ptrdiff_t i;
    for(i = 0; i < halos->np; i ++) {
        mapping[i] = -1;
        halos->id[i] = i;
    }

    /* remove non-contributing halo segments */
    fastpm_store_subsample(halos, mask, halos);

    for(i = 0; i < halos->np; i ++) {
        mapping[halos->id[i]] = i;
        halos->id[i] = i;
    }

    /* adjust head[i] */

    for(i = 0; i < finder->p->np; i ++) {
        if(head[i] >= 0) {
            head[i] = mapping[head[i]];
        }
    }

    fastpm_memory_free(finder->p->mem, mapping);
}

static void
fastpm_fof_apply_length_cut(FastPMFOFFinder * finder, FastPMStore * halos, ptrdiff_t * head)
{
    MPI_Comm comm = finder->priv->comm;

    FastPMParticleMaskType * mask = fastpm_memory_alloc(finder->p->mem, "LengthMask", sizeof(mask[0]) * halos->np, FASTPM_MEMORY_STACK);
    ptrdiff_t i;
    for(i = 0; i < halos->np; i ++) {
        /* remove halos that are shorter than nmin */
        if(halos->length[i] < finder->nmin) {
            mask[i] = 0;
        } else {
            mask[i] = 1;
        }
    }
    fastpm_fof_subsample_and_relabel(finder, halos, mask, head);
    fastpm_memory_free(finder->p->mem, mask);

    fastpm_info("After length cut we have %td halos (%td including ghost halos).\n", 
            fastpm_store_get_mask_sum(halos, comm),
            fastpm_store_get_np_total(halos, comm));
}

/* first every undecided halo; then the real halo with particles */
static int
FastPMLocalSortByMinID(const int i1,
                    const int i2,
                    FastPMStore * p)
{
    int v1 = (p->minid[i1] < p->minid[i2]);
    int v2 = (p->minid[i1] > p->minid[i2]);

    return v2 - v1;
}

static int
FastPMTargetMinID(FastPMStore * store, ptrdiff_t i, void * userdata)
{
    FastPMFOFFinder * finder = userdata;

    const uint32_t GOLDEN32 = 2654435761ul;
    /* const uint64_t GOLDEN64 = 11400714819323198549; */
    /* may over flow, but should be okay here as the periodicity is ggt NTask */
    int key = (store->minid[i] * GOLDEN32) % (unsigned) finder->priv->NTask;
    return key;
}

static int
FastPMTargetTask(FastPMStore * store, ptrdiff_t i, void * userdata)
{
    return store->task[i];
}

/* for debugging, move particles to a spatially unrelated rank */
static int
FastPMTargetFOF(FastPMStore * store, ptrdiff_t i, void * userdata)
{
#ifdef FASTPM_FOF_DEBUG
    PM * pm = userdata;
    int NTask;
    MPI_Comm_size(pm_comm(pm), &NTask);
    const uint32_t GOLDEN32 = 2654435761ul;
    /* const uint64_t GOLDEN64 = 11400714819323198549; */
    /* may over flow, but should be okay here as the periodicity is ggt NTask */
    int key = (store->id[i] * GOLDEN32) % (unsigned) NTask;
    return key;
#else
    return FastPMTargetPM(store, i, userdata);
#endif
}
/*
 * This function group halos by halos->minid[i].
 *
 * All halos with the same minid will have the same attribute values afterwards, except
 * a few book keeping items named in this function (see comments inside)
 *
 * */
static void
fastpm_fof_reduce_halo_attrs(FastPMFOFFinder * finder, FastPMStore * halos,
        void (* add_func) (FastPMFOFFinder * finder, FastPMStore * halo1, ptrdiff_t i1, FastPMStore * halo2, ptrdiff_t i2 ),
        void (* reduce_func)(FastPMFOFFinder * finder, FastPMStore * halo1, ptrdiff_t i1)
)
{

    ptrdiff_t i;

    ptrdiff_t first = -1;
    uint64_t lastminid = 0;

    /* ind is the array to use to replicate items */
    int * ind = fastpm_memory_alloc(finder->p->mem, "HaloPermutation", sizeof(ind[0]) * halos->np, FASTPM_MEMORY_STACK);

    /* the following items will have mask[i] == 1, but we will mark some to 0 if
     * they are not the principle (first) halos segment with this minid */
    for(i = 0; i < halos->np + 1; i++) {
        /* we use i == halos->np to terminate the last segment */
        if(first == -1 || i == halos->np || lastminid != halos->minid[i]) {
            if (first >= 0) {
                /* a segment ended */
                ptrdiff_t j;
                for(j = first; j < i; j ++) {
                    ind[j] = first;
                }
            }
            if (i < halos->np) {
                /* a segment started */
                halos->mask[i] = 1;
                lastminid = halos->minid[i];
            }
            /* starting a segment of the same halos */
            first = i;
        } else {
            /* inside segment, simply add the ith halo to the first of the segment */
            halos->mask[i] = 0;
            add_func(finder, halos, first, halos, i);
        }
    }

    /* use permute to replicate the first halo attr to the rest:
     *
     * we do not want to replicate
     * - fof, as fof.task is the original mpi rank of the halo
     * - id, as it is the original location of the halo on the original mpi rank. 
     * - mask, whether it is primary or not
     *
     * we need to replicate because otherwise when we return the head array on the
     * original ranks will be violated.
     *   */

    FastPMStore save[1];

    /* FIXME: add a method for this! */
    memcpy(save->columns, halos->columns, sizeof(save->columns));

    halos->minid = NULL;
    halos->task = NULL;
    halos->id = NULL;
    halos->mask = NULL;

    fastpm_store_permute(halos, ind);

    memcpy(halos->columns, save->columns, sizeof(save->columns));

    for(i = 0; i < halos->np; i++) {
        reduce_func(finder, halos, i);
    }

    fastpm_memory_free(finder->p->mem, ind);
}


static void
fastpm_fof_compute_halo_attrs(FastPMFOFFinder * finder, FastPMStore * halos,
            ptrdiff_t * head,
            void (*convert_func)(FastPMFOFFinder * finder, FastPMStore * p, ptrdiff_t i, FastPMStore * halos),
            void (*add_func)(FastPMFOFFinder * finder, FastPMStore * halos, ptrdiff_t i1, FastPMStore * halos2, ptrdiff_t i2),
            void (*reduce_func)(FastPMFOFFinder * finder, FastPMStore * halos, ptrdiff_t i)
)
{
    MPI_Comm comm = finder->priv->comm;

    FastPMStore h1[1];
    fastpm_store_init(h1, "FOF", 1, halos->attributes, FASTPM_MEMORY_HEAP);
    ptrdiff_t i;

    for(i = 0; i < finder->p->np; i++) {
        ptrdiff_t hid = head[i];
        if(hid < 0) continue;


        if(hid >= halos->np) {
            fastpm_raise(-1, "halo of a particle out of bounds (%td > %td)\n", hid, halos->np);
        }

        /* initialize h1 with existing halo attributes for this particle */
        fastpm_store_take(halos, hid, h1, 0);

        convert_func(finder, finder->p, i, h1);

        add_func(finder, halos, hid, h1, 0);
    }

    fastpm_store_destroy(h1);
    /* decompose halos by minid (gather); if all halo segments of the same minid are on the same rank, we can combine
     * these into a single entry, then replicate for each particle to look up;
     * halo segments that have no local particles are never exchanged to another rank. */
    if(0 != fastpm_store_decompose(halos,
            (fastpm_store_target_func) FastPMTargetMinID, finder, comm)) {

        fastpm_raise(-1, "out of space sending halos by MinID.\n");
    }

    /* now head[i] is no longer the halo attribute of particle i. */

    /* to combine, first local sort by minid; those without local particles are moved to the beginning
     * so we can easily skip them. */
    fastpm_store_sort(halos, FastPMLocalSortByMinID);

    /* reduce and update properties */
    fastpm_fof_reduce_halo_attrs(finder, halos, add_func, reduce_func);

    /* decompose halos by task (return) */
    if (0 != fastpm_store_decompose(halos,
            (fastpm_store_target_func) FastPMTargetTask, finder, comm)) {
        fastpm_raise(-1, "out of space for gathering halos this shall never happen.\n");
    }
    /* local sort by id (restore the order) */
    fastpm_store_sort(halos, FastPMLocalSortByID);

    /* now head[i] is again the halo attribute of particle i. */
}

/*
 * compute the attrs of the local halo segments based on local particles.
 * head : the hid of each particle; 
 * fofsave : the minid of each particle (unique label of each halo)
 *
 * if a halo has no local particles,  halos->mask[i] is set to 0.
 * if a halo has any local particles, and halos->mask[i] is set to 1.
 * */
static void
fastpm_fof_remove_empty_halos(FastPMFOFFinder * finder, FastPMStore * halos, uint64_t * minid, ptrdiff_t * head)
{

    /* */
    ptrdiff_t i;

    /* set minid and task of the halo; all of the halo particles of the same minid needs to be reduced */
    for(i = 0; i < finder->p->np; i++) {
        /* set the minid of the halo; need to take care of the ghosts too.
         * even though we do not add them to the attributes */
        ptrdiff_t hid = head[i];

        if(hid < 0) continue;

        if(halos->mask[hid] == 0) {
            /* halo will be reduced by minid */
            halos->minid[hid] = minid[i];
            halos->mask[hid] = 1;
        } else {
            if(halos->minid[hid] != minid[i]) {
                fastpm_raise(-1, "Consistency check failed after FOF global merge.\n");
            }
        }
    }

    /* halo will be returned to this task;
     * we save ThisTask here.*/
    for(i = 0; i < halos->np; i ++) {
        halos->task[i] = finder->priv->ThisTask;
    }

    fastpm_fof_subsample_and_relabel(finder, halos, halos->mask, head);
}

static void
_add_basic_halo_attrs(FastPMFOFFinder * finder, FastPMStore * halos, ptrdiff_t hid, FastPMStore * h1, ptrdiff_t i)
{
    double * boxsize = finder->priv->boxsize;

    if(halos->aemit)
        halos->aemit[hid] += h1->aemit[i];

    int d;

    for(d = 0; d < 3; d++) {
        if(halos->v)
            halos->v[hid][d] += h1->v[i][d];
        if(halos->dx1)
            halos->dx1[hid][d] += h1->dx1[i][d];
        if(halos->dx2)
            halos->dx2[hid][d] += h1->dx2[i][d];
    }

    if(halos->x) {
        for(d = 0; d < 3; d++) {
            if(boxsize) {
                halos->x[hid][d] = periodic_add(
                    halos->x[hid][d] / halos->length[hid], halos->length[hid],
                    h1->x[i][d] / h1->length[i], h1->length[i], boxsize[d]);
            } else {
                halos->x[hid][d] += h1->x[i][d];
            }
        }
    }

    if(halos->q) {
        for(d = 0; d < 3; d ++) {
            if (boxsize) {
                halos->q[hid][d] = periodic_add(
                    halos->q[hid][d] / halos->length[hid], halos->length[hid],
                    h1->q[i][d] / h1->length[i], h1->length[i], boxsize[d]);
            } else {
                halos->q[hid][d] += h1->q[i][d];
            }
        }
    }
    /* do this after the loop because x depends on the old length. */
    halos->length[hid] += h1->length[i];
}

/* convert a particle to the first halo in the halo store */
static void
_convert_basic_halo_attrs(FastPMFOFFinder * finder, FastPMStore * p, ptrdiff_t i, FastPMStore * halos)
{
    int hid = 0;

    int d;

    double q[3];

    if(halos->q && fastpm_store_has_q(p)) {
        fastpm_store_get_q_from_id(p, p->id[i], q);
    }
    halos->length[hid] = 1;

    for(d = 0; d < 3; d++) {
        if(halos->x)
            halos->x[hid][d] = p->x[i][d];
        if(halos->v)
            halos->v[hid][d] = p->v[i][d];
        if(halos->dx1)
            halos->dx1[hid][d] = p->dx1[i][d];
        if(halos->dx2)
            halos->dx2[hid][d] = p->dx2[i][d];
        if(halos->q && fastpm_store_has_q(p)) {
            halos->q[hid][d] = q[d];
        }
    }

    if(halos->aemit)
        halos->aemit[hid] = p->aemit[i];
}
static void
_reduce_basic_halo_attrs(FastPMFOFFinder * finder, FastPMStore * halos, ptrdiff_t i)
{
    int d;
    double n = halos->length[i];

    for(d = 0; d < 3; d++) {
        if(halos->x)
            halos->x[i][d] /= n;
        if(halos->v)
            halos->v[i][d] /= n;
        if(halos->dx1)
            halos->dx1[i][d] /= n;
        if(halos->dx2)
            halos->dx2[i][d] /= n;
        if(halos->q)
            halos->q[i][d] /= n;
    }
    if(halos->aemit)
        halos->aemit[i] /= n;
}

static void
_add_extended_halo_attrs(FastPMFOFFinder * finder, FastPMStore * h1, ptrdiff_t i1, FastPMStore * h2, ptrdiff_t i2)
{
    int d;
    if(h1->rvdisp) {
        for(d = 0; d < 9; d ++) {
            h1->rvdisp[i1][d] += h2->rvdisp[i2][d];
        }
    }
    if(h1->vdisp) {
        for(d = 0; d < 6; d ++) {
            h1->vdisp[i1][d] += h2->vdisp[i2][d];
        }
    }
    if(h1->rdisp) {
        for(d = 0; d < 6; d ++) {
            h1->rdisp[i1][d] += h2->rdisp[i2][d];
        }
    }
}

static void
_convert_extended_halo_attrs(FastPMFOFFinder * finder, FastPMStore * p, ptrdiff_t i, FastPMStore * halos)
{
    int hid = 0;

    int d;
    double rrel[3];

    for(d = 0; d < 3; d ++) {
        rrel[d] = p->x[i][d] - halos->x[hid][d];

        if(finder->priv->boxsize) {
            double L = finder->priv->boxsize[d];
            while(rrel[d] > L / 2) rrel[d] -= L;
            while(rrel[d] < -L / 2) rrel[d] += L;
        }
    }

    if(halos->vdisp) {
        /* FIXME: add hubble expansion term based on aemit and hubble function? needs to modify Finder object */

        double vrel[3];
        for(d = 0; d < 3; d ++) {
            vrel[d] = p->v[i][d] - halos->v[hid][d];
        }
        for(d = 0; d < 3; d ++) {
            halos->vdisp[hid][d] = vrel[d] * vrel[d];
            halos->vdisp[hid][d + 3] = vrel[d] * vrel[(d + 1) % 3];
        }
    }
    if(halos->rvdisp) {
        /* FIXME: add hubble expansion term based on aemit and hubble function? needs to modify Finder object */

        double vrel[3];
        for(d = 0; d < 3; d ++) {
            vrel[d] = p->v[i][d] - halos->v[hid][d];
        }
        for(d = 0; d < 3; d ++) {
            halos->rvdisp[hid][d] = rrel[d] * vrel[d];
            halos->rvdisp[hid][d + 3] = rrel[d] * vrel[(d + 1) % 3];
            halos->rvdisp[hid][d + 6] = rrel[d] * vrel[(d + 2) % 3];
        }
    }
    if(halos->rdisp) {
        for(d = 0; d < 3; d ++) {
            halos->rdisp[hid][d] = rrel[d] * rrel[d];
            halos->rdisp[hid][d + 3] = rrel[d] * rrel[(d + 1) % 3];
        }
    }
}
static void
_reduce_extended_halo_attrs(FastPMFOFFinder * finder, FastPMStore * halos, ptrdiff_t hid)
{
    double n = halos->length[hid];
    int d;
    if(halos->rvdisp) {
        for(d = 0; d < 9; d ++) {
            halos->rvdisp[hid][d] /= n;
        }
    }
    if(halos->vdisp) {
        for(d = 0; d < 6; d ++) {
            halos->vdisp[hid][d] /= n;
        }
    }
    if(halos->rdisp) {
        for(d = 0; d < 3; d ++) {
            halos->rdisp[hid][d] /= n;
        }
    }
}

/* This function creates the storage object for halo segments that are local on this
 * rank. We store many attributes. We only allow a flucutation of 2 around avg_halos.
 * this should be OK, since we will only redistribute by the MinID, which are supposed
 * to be very uniform.
 * */
void
fastpm_fof_allocate_halos(FastPMStore * halos,
    size_t nhalos,
    FastPMStore * p,
    int include_q,
    MPI_Comm comm)
{

    FastPMColumnTags attributes = p->attributes;
    attributes |= COLUMN_MASK;
    attributes |= COLUMN_LENGTH | COLUMN_MINID | COLUMN_TASK;
    attributes |= COLUMN_RDISP | COLUMN_VDISP | COLUMN_RVDISP;
    attributes |= COLUMN_ACC; /* ACC used as the first particle position offset */
    attributes &= ~COLUMN_POTENTIAL;
    attributes &= ~COLUMN_DENSITY;
    attributes &= ~COLUMN_TIDAL;

    /* store initial position only for periodic case. non-periodic suggests light cone and
     * we cannot infer q from ID sensibly. (crashes there) */
    if(include_q) {
        attributes |= COLUMN_Q;
    } else {
        attributes &= ~COLUMN_Q;
    }

    double avg_halos;
    double max_halos;
    /* + 1 to ensure avg_halos > 0 */
    MPIU_stats(comm, nhalos + 1, "->", &avg_halos, &max_halos);

    fastpm_info("Allocating %d halos per rank for final catalog.\n", (size_t) max_halos * 2);

    /* give it enough space for rebalancing. */
    fastpm_store_init(halos, NULL, (size_t) (max_halos * 2),
            attributes,
            FASTPM_MEMORY_FLOATING);

    halos->np = nhalos;
    halos->meta = p->meta;

    ptrdiff_t i;
    for(i = 0; i < halos->np; i++) {
        halos->id[i] = i;

        halos->mask[i] = 0; /* unselect the halos ; will turn this only if any particle is used */
        /* everthing should have been set to zero already by fastpm_store_init */
    }
}

void
fastpm_fof_execute(FastPMFOFFinder * finder,
    double linkinglength,
    FastPMStore * halos,
    ptrdiff_t ** ihalo,
    FastPMParticleMaskType * active)
{
    /* initial decompose -- reduce number of ghosts */
    FastPMStore * p = finder->p;
    PMGhostData * pgd = finder->priv->pgd;
    ptrdiff_t * head = finder->priv->head;

    FastPMParticleMaskType * old_mask = finder->p->mask;
    int use_mask;

    if(active) {
        finder->p->mask = active;
        pm_ghosts_send(pgd, COLUMN_MASK);
        use_mask = 1;
    } else {
        use_mask = 0;
    }

    FastPMStore * stores[2] = {p, pgd->p};

    finder->priv->root = _create_kdtree(&finder->priv->tree,
                                        finder->kdtree_thresh,
                                        stores, 2, finder->priv->boxsize, use_mask);

    size_t np_and_ghosts = p->np + pgd->p->np;

    FastPMStore savebuff[1];
    fastpm_store_init(savebuff, p->name, np_and_ghosts, COLUMN_MINID, FASTPM_MEMORY_STACK);

    ptrdiff_t i;
    /* kdcount will only modify the head of active particles.
     * thus inactive particles are never linked. */
    for(i = 0; i < np_and_ghosts; i ++) {
        head[i] = i;
    }
    /* local find of p and the ghosts */
    kd_fof(finder->priv->root, linkinglength, head);

    _fof_global_merge (finder, p, pgd, savebuff->minid, head);

    /* assign halo attr entries. This will keep only candidates that can possibly reach to nmin */
    size_t nsegments = _assign_halo_attr(finder, pgd, head, p->np, pgd->p->np, finder->nmin);

    fastpm_info("Found %td halos segments >= %d particles; or cross linked. \n", nsegments, finder->nmin);
    /* create local halos */
    fastpm_fof_allocate_halos(halos, nsegments, finder->p, finder->priv->boxsize != NULL, finder->priv->comm);
    /* remove halos without any local particles */
    fastpm_fof_remove_empty_halos(finder, halos, savebuff->minid, head);

    fastpm_store_destroy(savebuff);

    /* reduce the primary halo attrs */
    fastpm_fof_compute_halo_attrs(finder, halos, head, _convert_basic_halo_attrs, _add_basic_halo_attrs, _reduce_basic_halo_attrs);

    #ifdef FASTPM_FOF_DEBUG
    {
        int i;
        for(i  = 0; i < halos->np; i ++) {
            fastpm_ilog(INFO, "Task = %d, Halo[%d] = %d mask=%d MINID=%ld\n", finder->priv->ThisTask, i, halos->length[i], halos->mask[i], halos->minid[i]);
        }
    }
    #endif

    /* apply length cut */
    fastpm_fof_apply_length_cut(finder, halos, head);

    /* reduce the primary halo attrs */
    fastpm_fof_compute_halo_attrs(finder, halos, head, _convert_extended_halo_attrs, _add_extended_halo_attrs, _reduce_extended_halo_attrs);

    /* halos stores halos that spans to this rank, with duplication.
     * only those where mask==1 are primary
     * the others are ghosts with the correct properties but shall not show up in the
     * catalog.
     *
     * halos[ihalo[i]] is the hosting halo of particle i, if ihalo[i] >= 0.
     * */
    if (ihalo) {
        /* return the full halo catalog. */
        *ihalo = head;
    } else {
        fastpm_store_subsample(halos, halos->mask, halos);
    }

    _free_kdtree(&finder->priv->tree, finder->priv->root);

    /* restore mask */
    finder->p->mask = old_mask;
}

void
fastpm_fof_destroy(FastPMFOFFinder * finder)
{
    fastpm_memory_free(finder->p->mem, finder->priv->head);
    pm_ghosts_free(finder->priv->pgd);
    free(finder->priv);
}


