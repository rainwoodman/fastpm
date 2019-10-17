FASTPM_BEGIN_DECLS

extern double HubbleConstant;
extern double HubbleDistance;

struct FastPMCosmology {
    double h;
    double Omega_cdm;
    double T_cmb;    /*related to Omega_g*/
    //double T_nu;    //todays neutrino temperature (maybe use T_nu0?) HARD CODED FOR NOW.
    double N_eff;   //3.046
    int N_nu;  // total number of neutrino species (massive and massless) (could use 'N_ur' as number of ur nus, instead, but dont want conflict with CLASS defn)
    double m_ncdm[3]; // masses of massive neutrinos (ncdm) for now assume max of 3 ncdm nus.
    int N_ncdm;
    
    /* Fermi Dirac interp for ncdm 
       It's not really cosmology dependant,
       but only cosmology uses it, so I guess
       it makes sense to have a pointer to it here */
    FastPMFDInterp * FDinterp;   //FIXME: I guess this will just be NULL in the case of no ncdm. Is this ugly?
};

double interpolate(const double xa[], const double ya[], size_t size, double xi);

double Omega_g(FastPMCosmology * c);
double Gamma_nu(FastPMCosmology * c);
double Omega_ur(FastPMCosmology * c);
double Omega_r(FastPMCosmology * c);

double getFtable(int F_id, double y, FastPMCosmology * c);
double Fconst(int ncdm_id, FastPMCosmology * c);

double Omega_ncdm_iTimesHubbleEaSq(double a, int ncdm_id, FastPMCosmology * c);

double Omega_ncdmTimesHubbleEaSq(double a, FastPMCosmology * c);
double DOmega_ncdmTimesHubbleEaSqDa(double a, FastPMCosmology * c);
double D2Omega_ncdmTimesHubbleEaSqDa2(double a, FastPMCosmology * c);

double w_ncdm_i(double a, int ncdm_id, FastPMCosmology * c);

double Omega_Lambda(FastPMCosmology * c);

double HubbleEa(double a, FastPMCosmology * c);
double Omega_ncdm_i(double a, int ncdm_id, FastPMCosmology * c);
double Omega_ncdm(double a, FastPMCosmology * c);
double Omega_ncdm_i_m(double a, int ncdm_id, FastPMCosmology * c);
double Omega_ncdm_m(double a, FastPMCosmology * c);
double Omega_cdm_a(double a, FastPMCosmology * c);
double OmegaA(double a, FastPMCosmology * c);
double Omega_m(double a, FastPMCosmology * c);
double DHubbleEaDa(double a, FastPMCosmology * c);
double D2HubbleEaDa2(double a, FastPMCosmology * c);

double OmegaSum(double a, FastPMCosmology* c);

typedef struct {
    double y0;
    double y1;
    double y2;
    double y3;
} ode_soln;

//dont put these in .h
//static int growth_odeNu(double a, const double y[], double dyda[], void *params);
//static ode_soln growth_ode_solve(double a, FastPMCosmology * c);

double growth(double a, FastPMCosmology * c);
double DgrowthDlna(double a, FastPMCosmology * c);
double growth2(double a, FastPMCosmology * c);
double Dgrowth2Dlna(double a, FastPMCosmology * c);

double GrowthFactor(double a, FastPMCosmology * c);
double GrowthFactor2(double a, FastPMCosmology * c);

double DLogGrowthFactor(double a, FastPMCosmology * c);
double DLogGrowthFactor2(double a, FastPMCosmology * c);

double DGrowthFactorDa(double a, FastPMCosmology * c);
double D2GrowthFactorDa2(double a, FastPMCosmology * c);

double ComovingDistance(double a, FastPMCosmology * c);
double OmegaA(double a, FastPMCosmology * c);

FASTPM_END_DECLS
