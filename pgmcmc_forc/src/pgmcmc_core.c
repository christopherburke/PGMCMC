#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <math.h>
#include <pgmcmc.h>
#include <hdf5.h>

int pgmcmc_setup(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud, pgmcmc_userfuncs *uf) {
  // This routine is called before running mcmc
  // it fills out all the other runtime variables that need setting up
  int nparm, i, nfreepar, j, err;
  size_t ndata, ii;

  err = 0;

  nparm = io->nparm;
  for (i=0; i<nparm; i++) {
    io->physvalsavs[i] = io->physvals[i];
    io->bestphysvals[i] = io->physvals[i];
    io->calcvalsavs[i] = io->calcvals[i];
    io->bestcalcvals[i] = io->calcvals[i];
  }

  // To allow some parameters to be fixed only the free parameters
  // get used paridx stores index of original parameter sequence
  nfreepar = 0;
  j = 0;
  for (i=0; i<nparm; i++) {
    if (io->fixed[i] == 0) {
      nfreepar++;
      mc->paridx[j] = i;
      mc->pars[j] = io->calcvals[i];
      mc->attempts[j] = 0;
      mc->accepts[j] = 0;
      j++;
    }
  }
  mc->nfreepar = nfreepar;
  mc->pos = 0;
  mc->like = 0.0;
  mc->prior = 0.0;
  mc->chi2 = 0.0;
  mc->curpar = 0;
  if (pm->debugLevel > 0) {
    fprintf(stdout, "%d parameters %d are free\n", io->nparm, mc->nfreepar);
  }
  ndata = ud->ndata;
  io->chi2min = ndata * 9.99e9; // initialize chi2 min to a value much larger than expected
  io->expchi2 = (double) ndata - nfreepar; // expected chi2
  // Calculate the Gaussian log likelihood normalization constant
  io->likenorm = 0.0;
  for (ii=0; ii<ndata; ii++) {
    io->likenorm = io->likenorm + log(gsl_vector_get(ud->yErrData, ii));
  }
  io->likenorm = -io->likenorm - ((double) ndata) * log(sqrt(2.0*M_PI));
  if (pm->debugLevel > 0) {
    fprintf(stdout,"Number of data points %zu. Expected Chi2 %lf\n", ud->ndata, io->expchi2);
  }
  // Start testing user functions
  // confirm that calcvals->physvals function returns

  // Execute Prior and verify it does not have error
  err = uf->func_prior(io, pm, mc, ud);
  if (err != 0) {
    fprintf(stderr, "Error encountered on test call to func_prior\n");
    fprintf(stderr, "Physical variable min start max\n");
    for (i=0; i<io->nparm; i++) {
      fprintf(stderr, "%d %lf %lf %lf\n", i, io->physval_mins[i],
              io->physvals[i], io->physval_maxs[i]);
    }
    fprintf(stderr, "Calculation variable min start max\n");
    for (i=0; i<io->nparm; i++) {
      fprintf(stderr, "%d %lf %lf %lf\n", i, io->calcval_mins[i],
              io->calcvals[i], io->calcval_maxs[i]);
    }
    exit(EXIT_FAILURE);
  } else {
    if (pm->debugLevel > 0) {
      fprintf(stdout, "Passed Initial Prior Test Value: %lf\n", mc->prior);
    }
  }

  // Execute model and verify it does not have error
  err = uf->func_calcmodel(io, pm, mc, ud);
  if (err != 0) {
    fprintf(stderr, "Error encountered on test call to func_calcmodel\n");
    fprintf(stderr, "Physical variable value ; calculation value\n");
    for (i=0; i<io->nparm; i++) {
      fprintf(stderr, "%d %lf %lf\n", i, io->physvals[i], io->calcvals[i]);
    }
    exit(EXIT_FAILURE);
  } else {
    if (pm->debugLevel > 0) {
      fprintf(stdout, "Passed Initial Model Test\n");
    }
  }

  // Execute likehood and verify it does not have error
  err = uf->func_likehood(io, pm, mc, ud);
  if (err != 0) {
    fprintf(stderr, "Error encountered on test call to func_likehood\n");
    fprintf(stderr, "Physical variable value ; calculation value\n");
    for (i=0; i<io->nparm; i++) {
      fprintf(stderr, "%d %lf %lf\n", i, io->physvals[i], io->calcvals[i]);
    }
    exit(EXIT_FAILURE);
  } else {
    if (pm->debugLevel > 0) {
      fprintf(stdout, "Passed Initial Likehood Test\n");
    }
  }

  // Run test of taking a single mcmc step through the parameters
  err = pgmcmc_one_mcmc_step(io, pm, mc, ud, uf);
  if (err == 0) {
    if (pm->debugLevel > 0) {
      fprintf(stdout, "Passed One MCMC Step Test\n");
    }
  }

  return err;
}


int pgmcmc_one_mcmc_step(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud, pgmcmc_userfuncs *uf) {
  // Core MCMC step functionality run through all free parameters
  int err=0, ip, j, nfreepar, gdstp;
  double curchi2, curprior, nexchi2, nexprior, rat, lnrat;
  // Make copies of parameter values and set things up for step
  nfreepar = mc->nfreepar;
  for (ip=0; ip<nfreepar; ip++) {
    j = mc->paridx[ip];
    mc->curp[ip] = io->calcvals[j];
    mc->sigs[ip] = io->scls[j];
    mc->nexp[ip] = mc->curp[ip];
  }
  curchi2 = mc->like;
  curprior = mc->prior;

  // Iterate through each free parameter
  for (ip=0; ip<nfreepar; ip++) {
    mc->attempts[ip] += 1;
    mc->curpar = mc->paridx[ip];
    gdstp = 0;
    // Jump in current parameter only
    mc->nexp[ip] = gsl_ran_gaussian(mc->ranobj, mc->sigs[ip]) + mc->curp[ip];
    for (j=0; j<nfreepar; j++) {
      mc->pars[j] = mc->nexp[j];
    }
    // Check the prior for new set of parameters
    err = uf->func_prior(io, pm, mc, ud);
    if (err == 0) {
      // Passes prior now calculate likelihood
      err = uf->func_calcmodel(io, pm, mc, ud);
      err = uf->func_likehood(io, pm, mc, ud);
      nexchi2 = mc->like;
      nexprior = mc->prior;
      lnrat = nexprior - curprior + nexchi2 - curchi2;
      rat = pow(M_E, lnrat);
      if (lnrat >= 0.0) {
        // Pass by large margin
        mc->curp[ip] = mc->nexp[ip];
        mc->accepts[ip] += 1;
        gdstp = 1;
      } else if (gsl_ran_flat(mc->ranobj, 0.0, 1.0) < rat) {
        // randomly passes
        mc->curp[ip] = mc->nexp[ip];
        mc->accepts[ip] += 1;
        gdstp = 1;
      } else {
        // Did not pass. Repeat previous parameters
        nexchi2 = curchi2;
        nexprior = curprior;
        mc->nexp[ip] = mc->curp[ip];
      }
    } else {
      // Did not pass prior or prior failed. Repeat previous parameters
      nexchi2 = curchi2;
      nexprior = curprior;
      mc->nexp[ip] = mc->curp[ip];
    }
    // Reset for next parameter
    curchi2 = nexchi2;
    curprior = nexprior;
    mc->curp[ip] = mc->nexp[ip];
  }

  // The case of the last parameter not being good (gdstp == 0) needs special
  // attention to reset things and sync things up
  if (gdstp == 0) {
    mc->like = curchi2;
    mc->prior = curprior;
    for (ip=0; ip<nfreepar; ip++) {
      j = mc->paridx[ip];
      mc->pars[ip] = mc->curp[ip];
      io->calcvals[j] = mc->curp[ip];
    }
    err = uf->func_physvals(io, pm, mc, ud);
  }

  return err;
}

int pgmcmc_burnit(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud, pgmcmc_userfuncs *uf, size_t nBurn) {
  // Burn through several chain steps and dont save results, but return
  // acceptance fraction data.
  int err=0, i, nparm, nfreepar;
  size_t istp;
  
  nparm = io->nparm;
  nfreepar = mc->nfreepar;
  // Zero out the attempt and accept data
  for (i=0; i<nparm; i++) {
    mc->accepts[i] = 0;
    mc->attempts[i] = 0;
  }
  // Execute the nBurn steps
  for (istp=0; istp<nBurn; istp++) {
    err = pgmcmc_one_mcmc_step(io, pm, mc, ud, uf);
  }
  // Calculate the acceptance fractions
  for (i=0; i<nfreepar; i++) {
    mc->fracs[i] = ((double) mc->accepts[i]) / ((double) mc->attempts[i]);
  }
  return err;
}

int pgmcmc_iterate_proposals(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud, pgmcmc_userfuncs *uf) {
  int err = 0, nparm, nfreepar, i, j, converg = 0, nfreepar2;
  size_t istp;

  nparm = io->nparm;
  nfreepar = mc->nfreepar;
  nfreepar2 = nfreepar * 2;

  if (pm->debugLevel > 0) {
    fprintf(stdout, "Start Initial Burn\n");
  }
  err = pgmcmc_burnit(io, pm, mc, ud, uf, pm->initNSteps);
  if (pm->debugLevel > 0) {
    fprintf(stdout, "Initial Burn Finished\n");
    // Display the acceptance fraction
    for (i=0; i<nfreepar; i++) {
      fprintf(stdout,"Parm %d Acceptance frac: %g\n", i, mc->fracs[i]);
    }
  }
  
  // Start coarse proposal step size corrections
  istp = 0;
  converg = 0; // Every time a free parameter is within the fraction limits
              // converg is increased. If all free parameters are within frac limits
              // then converg == nfreepar *2.
  while ((converg < nfreepar2) && (istp < pm->maxPropTries)) {
    converg = 0;
    // Go over the free parameters and find ones that the sigs
    // need adjusting because the acceptance fractions are beyond limits
    for (i=0; i<nfreepar; i++) {
      j = mc->paridx[i];
      if (mc->fracs[i] < pm->coarseLowFrac) {
        io->scls[j] /= 2.0;
      } else {
        converg += 1;
      }
      if (mc->fracs[i] > pm->coarseHghFrac) {
        io->scls[j] *= 2.0;
      } else {
        converg += 1;
      }
    }
    // Do a coarse burn
    err = pgmcmc_burnit(io, pm, mc, ud, uf, pm->coarseNSteps);
    if (pm->debugLevel > 0) {
      fprintf(stdout, "Coarse Iteration: %zu\n", istp);
      // Display the acceptance fraction
      for (i=0; i<nfreepar; i++) {
        j = mc->paridx[i];
        fprintf(stdout,"Parm %d Acceptance frac: %lf Scale: %g\n", j, mc->fracs[i], io->scls[j]);
      }
    }
    istp += 1;
  }
  if (pm->debugLevel > 0) {
    fprintf(stdout,"Done Coarse Proposal Iteration\n");
  }
  // Check if we reached the maximum number of coarse iterations if so exit
  if (istp == pm->maxPropTries) {
    fprintf(stderr, "Coarse Proposal Iteration did not converge in steps: %zu!\n", pm->maxPropTries);
    exit(EXIT_FAILURE);
  }
  
  // Start the Refined Proposal Step Size corrections
  istp = 0;
  converg = 0; // Every time a free parameter is within the fraction limits
              // converg is increased. If all free parameters are within frac limits
              // then converg == nfreepar *2.
  while ((converg < nfreepar2) && (istp < pm->maxPropTries)) {
    converg = 0;
    // Go over the free parameters and find ones that the sigs
    // need adjusting because the acceptance fractions are beyond limits
    for (i=0; i<nfreepar; i++) {
      j = mc->paridx[i];
      if (mc->fracs[i] < pm->refineLowFrac) {
        io->scls[j] = io->scls[j] / (1.0 + fabs(mc->fracs[i] - pm->fracwant));
      } else {
        converg += 1;
      }
      if (mc->fracs[i] > pm->refineHghFrac) {
        io->scls[j] = io->scls[j] * (1.0 + fabs(mc->fracs[i] - pm->fracwant));
      } else {
        converg += 1;
      }
    }
    // Do a refine burn
    err = pgmcmc_burnit(io, pm, mc, ud, uf, pm->refineNSteps);
    if (pm->debugLevel > 0) {
      fprintf(stdout, "Refine Iteration: %zu\n", istp);
      // Display the acceptance fraction
      for (i=0; i<nfreepar; i++) {
        j = mc->paridx[i];
        fprintf(stdout,"Parm %d Acceptance frac: %lf Scale: %g\n", j, mc->fracs[i], io->scls[j]);
      }
    }
    istp += 1;
  }
  if (pm->debugLevel > 0) {
    fprintf(stdout,"Done Refine Proposal Iteration\n");
  }
  // Check if we reached the maximum number of coarse iterations if so exit
  if (istp == pm->maxPropTries) {
    fprintf(stderr, "Refine Proposal Iteration did not converge in steps: %zu!\n", pm->maxPropTries);
    exit(EXIT_FAILURE);
  }
  return err;
}

int pgmcmc_run_mcmc(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud, pgmcmc_userfuncs *uf) {
  int err = 0, nparm, nfreepar, i, j;
  size_t istp, maxstps;
  gsl_matrix *pvals, *cvals, *bvals;
  hid_t file_id, dataset_id, dataspace_id;
  hsize_t h5dims[2];
  herr_t h5status;

  nparm = io->nparm;
  nfreepar = mc->nfreepar;
  maxstps = pm->maxstps;
  
  // Allocate parameter storage arrays for saving results
  pvals = gsl_matrix_alloc(maxstps, nfreepar);
  cvals = gsl_matrix_alloc(maxstps, nfreepar);
  bvals = gsl_matrix_alloc(maxstps, 3);
  
  // Reset things
  mc->pos = 0;
  mc->curpar = 0;
  for (i=0; i<nparm; i++) {
    mc->accepts[i] = 0;
    mc->attempts[i] = 0;
  }
  
  // Determine proposal step sizes
  err = pgmcmc_iterate_proposals(io, pm, mc, ud, uf);

  // Reset things
  mc->pos = 0;
  mc->curpar = 0;
  for (i=0; i<nparm; i++) {
    mc->accepts[i] = 0;
    mc->attempts[i] = 0;
  }
  if (pm->debugLevel > 0) {
    fprintf(stdout, "Start MCMC Run\n");
  }
  for (istp=0; istp<pm->maxstps; istp++) {
    mc->pos = istp;
    err = pgmcmc_one_mcmc_step(io, pm, mc, ud, uf);
    if (fmod(istp, 1000) == 0 && pm->debugLevel > 0) {
      fprintf(stdout, "Step %zu\n", istp);
    }
    // Save results for this step
    for (i=0; i<nfreepar; i++) {
      j = mc->paridx[i];
      gsl_matrix_set(pvals, istp, i, io->physvals[j]);
      gsl_matrix_set(cvals, istp, i, io->calcvals[j]);
    }
    gsl_matrix_set(bvals, istp, 0, mc->like);
    gsl_matrix_set(bvals, istp, 1, mc->prior);
    gsl_matrix_set(bvals, istp, 2, mc->chi2);
  }
  
  // Write out chains to hdf5 files
  // Create file
  file_id = H5Fcreate(pm->outputFilename,  H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  // Create the h5 object that describes the dimensions of array called a data space
  h5dims[0] = maxstps;
  h5dims[1] = nfreepar;
  dataspace_id = H5Screate_simple(2, h5dims, NULL);
  // Now create the h5 object that puts together data space with data type (int, double, etc) called the dataset
  dataset_id = H5Dcreate(file_id, "/pvals", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the pvals data to the h5 data set
  h5status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gsl_matrix_ptr(pvals, 0, 0));
  // Close dataset
  h5status = H5Dclose(dataset_id);
  // Close dataspace
  h5status = H5Sclose(dataspace_id);
  
  // Repeat for cvals
  h5dims[0] = maxstps;
  h5dims[1] = nfreepar;
  dataspace_id = H5Screate_simple(2, h5dims, NULL);
  dataset_id = H5Dcreate(file_id, "/cvals", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  h5status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gsl_matrix_ptr(cvals, 0, 0));
  h5status = H5Dclose(dataset_id);
  h5status = H5Sclose(dataspace_id);

  // Repeat for bvals
  h5dims[0] = maxstps;
  h5dims[1] = 3;
  dataspace_id = H5Screate_simple(2, h5dims, NULL);
  dataset_id = H5Dcreate(file_id, "/bvals", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  h5status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gsl_matrix_ptr(bvals, 0, 0));
  h5status = H5Dclose(dataset_id);
  h5status = H5Sclose(dataspace_id);

  // Close File
  h5status = H5Fclose(file_id);
  
  gsl_matrix_free(pvals);
  gsl_matrix_free(cvals);
  gsl_matrix_free(bvals);
  return err;
}
