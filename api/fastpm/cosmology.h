#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    double OmegaM;
    double OmegaLambda;
} Cosmology;

double GrowthFactor(double a, Cosmology c);
double GrowthFactor2(double a, Cosmology c);

double DLogGrowthFactor(double a, Cosmology c);
double DLogGrowthFactor2(double a, Cosmology c);

double HubbleEa(double a, Cosmology c);
double OmegaA(double a, Cosmology c);

#ifdef __cplusplus
}
#endif