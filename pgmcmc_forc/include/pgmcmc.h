#ifndef PGMCMC_INCLUDES
#define PGMCMC_INCLUDES

#define MAX_NAME_LEN 20

typedef struct pgmcmc_ioblk
{
  /* Stores the parameters for the user specific problem. The user needs
      to fill these in order to run pgmcmc. 'Physical variables' are
      the 'user friendly' parameter set that one sets the problem up with.
      'Calculation (calc) variables' are the variables that the mcmc algorithm
      is actually run on. */
  int nparm; // Number of parameters
  double *origests; // original estimates of the 'physical values' for parameters
  int *fixed; // flags 1=parameter is fixed during solving; 0=parameter is free and solved for
  double *physval_mins; // Minimum physical value for parameters
  double *physval_maxs; // Max physcial value for parameters
  double *physvals; // storage for physcial variables
  double *physvalsavs; // more storage for physical variables
  double *bestphysvals; // set of physical variables that minimize chi2 or max likehood
  double *calcvals; // storage for calculation variables
  double *calcvalsavs; // more storage for calculation variables
  double *bestcalcvals; // set of calculation variables that minimize chi2 or max likehood
  double *calcval_mins; // Minimum calculation value for parameters
  double *calcval_maxs; // Maximum calculation value for parameters
  double *scls; // MCMC step size/scales
  double chi2min; // Minimum chi2 during run
  double likenorm; // normalization for likelikhood that is the same for all steps
  double expchi2; // save the expected chi2
  double chi2add; // chisquare value that gets calculated in prior and is added to chi2 value in likehood
} pgmcmc_ioblk;

typedef struct pgmcmc_userdata
{
  /* Holds all the data for the users specific problem */
  size_t ndata;
  gsl_vector *xData;
  gsl_vector *yData;
  gsl_vector *yErrData;
  gsl_vector *yModel;
} pgmcmc_userdata;

typedef struct pgmcmc_algo_params
{
  /*Stores parameters for the mcmc run. User will usually want to change some of
     these things*/
  int debugLevel; // 3 - Debug level setting
  size_t maxstps; // 300000 - Maximum number of MCMC steps in run
  double fracwant; // 0.25 - Target for fraction of steps accepted
  size_t initNSteps; // 200 - burn in before starting to gather any stats
  size_t coarseNSteps; // 100 - Initial Coarse solving for mcmc step scales. number of steps before calculating accepted fraction
  double coarseLowFrac; // 0.1 - acceptable range for coarse scale solving
  double coarseHghFrac; // 0.8
  size_t refineNSteps; // 400 - Refined mcmc step scale solving number of steps
  double refineLowFrac; // 0.15  - acceptabel range for refined scale solving
  double refineHghFrac; // 0.42
  size_t maxPropTries; // 50 - Number of tries to adjusting step sizes before giving up
  size_t saveNSteps; // 100 - Write out results after this many steps
  char *outputFilename; // output prefix for hdf5 chain output
} pgmcmc_algo_params;

typedef struct pgmcmc_mcmc
{
  /* Stores variables and things for the runtime mcmc algorithm
  One shouldnt need to set anything here as it gets set in pgmcmc_setup*/
  int nfreepar; // Number of free parameters
  size_t pos; // position in chain
  double *pars; // array of the free parameters only
  int *paridx; // indices from physcals (all variables) into pars
  double like; // current likelihood
  double prior; // current prior
  double chi2; // current chi square
  long *attempts; // keeps track of step attempts for each free parameter
  long *accepts; // keep track of step accepts for each parameter
  int curpar; // Current parameter index
  double *curp; // mcmc step auxilliary storage
  double *sigs; // "
  double *nexp; // "
  double *fracs; //
  // Random number generation variables
  unsigned long int ranseed; // Fixed random seed
  gsl_rng *ranobj; // Generator object
} pgmcmc_mcmc;

// typdef the standard function call for the user functions
typedef int (*pgmcmc_standard_func)(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud);

typedef struct pgmcmc_userfuncs
{
  /* Holds all the pointers to the user specific problem */
  pgmcmc_standard_func func_physvals; // convert calculation values to physical values
  pgmcmc_standard_func func_calcvals; // convert physical values to calculation values
  pgmcmc_standard_func func_prior; // calculate prior
  pgmcmc_standard_func func_likehood; // calculate likehood
  pgmcmc_standard_func func_calcmodel; // return model y values
} pgmcmc_userfuncs;

// pgmcmc_core
int pgmcmc_setup(pgmcmc_ioblk*, pgmcmc_algo_params*, pgmcmc_mcmc*, pgmcmc_userdata*, pgmcmc_userfuncs*);
int pgmcmc_one_mcmc_step(pgmcmc_ioblk*, pgmcmc_algo_params*, pgmcmc_mcmc*, pgmcmc_userdata*, pgmcmc_userfuncs*);
int pgmcmc_run_mcmc(pgmcmc_ioblk*, pgmcmc_algo_params*, pgmcmc_mcmc*, pgmcmc_userdata*, pgmcmc_userfuncs*);

// pgmcmc_report
int pgmcmc_report(pgmcmc_ioblk*, pgmcmc_algo_params*, pgmcmc_mcmc*, pgmcmc_userdata*, pgmcmc_userfuncs*, char*, char*);

#endif
