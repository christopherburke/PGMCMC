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
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_fit.h>
#include <plplot/plplot.h>
#include <math.h>
#include <pgmcmc.h>

extern int errno ;

#define MAX_LINE 1000
#define NPARM 7

int func_physvals(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud) {
  // Convert calculation variables to physical variables
  int err=0, i;
  double alpha, beta, log10errscl, x0, xs, y0, ys;
  double a, b, errscl;

  alpha = io->calcvals[0];
  beta = io->calcvals[1];
  log10errscl = io->calcvals[2];
  x0 = io->calcvals[3];
  xs = io->calcvals[4];
  y0 = io->calcvals[5];
  ys = io->calcvals[6];

  a = ys * alpha / xs;
  b = ys * beta + y0 - a * x0;
  errscl = pow(10.0, log10errscl) * ys;

  //printf("%lf %lf %lf %lf %lf %lf\n", alpha, a, beta, b, log10errscl, errscl);
  io->physvals[0] = a;
  io->physvals[1] = b;
  io->physvals[2] = errscl;
  io->physvals[3] = x0;
  io->physvals[4] = xs;
  io->physvals[5] = y0;
  io->physvals[6] = ys;

  // Check for Nans
  for (i=0; i<io->nparm; i++) {
    if (!isfinite(io->physvals[i])) {
      err = 1;
      fprintf(stderr, "Non Finite physval encountered in func_physvals\n");
      for (i=0; i<io->nparm; i++) {
        fprintf(stderr,"Param: %d Calcval: %lf -> Physval: %lf\n",
                i, io->calcvals[i], io->physvals[i]);
      }
      exit(EXIT_FAILURE);
    }
  }
  return err;
}

int func_calcvals(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud) {
  // Convert physical variables to calculation variables
  int err=0, i;
  double alpha, beta, log10errscl, x0, xs, y0, ys;
  double a, b, errscl;

  a = io->physvals[0];
  b = io->physvals[1];
  errscl = io->physvals[2];
  x0 = io->physvals[3];
  xs = io->physvals[4];
  y0 = io->physvals[5];
  ys = io->physvals[6];

  alpha = xs * a / ys;
  beta = (b - y0 + a*x0) / ys;
  log10errscl = log10(errscl / ys);

  //printf("%lf %lf %lf %lf %lf %lf\n", alpha, a, beta, b, log10errscl, errscl);
  io->calcvals[0] = alpha;
  io->calcvals[1] = beta;
  io->calcvals[2] = log10errscl;
  io->calcvals[3] = x0;
  io->calcvals[4] = xs;
  io->calcvals[5] = y0;
  io->calcvals[6] = ys;

  // Check for Nans
  for (i=0; i<io->nparm; i++) {
    if (!isfinite(io->calcvals[i])) {
      err = 1;
      fprintf(stderr, "Non Finite calcval encountered in func_calcvals\n");
      for (i=0; i<io->nparm; i++) {
        fprintf(stderr,"Param: %d Physval: %lf -> Calcval: %lf\n",
                i, io->physvals[i], io->calcvals[i]);
      }
      exit(EXIT_FAILURE);
    }
  }
  return err;
}

int func_prior(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud) {
  // Calculate the prior for the current parameters
  int err=0, i, j, anybadcheck, anybadcheck2, anybadcheck3, tmpi;
  double prior = 0.0; // log prior value

  // Copy the current calculate variables mc->pars into io.calcvals
  for (i=0; i<mc->nfreepar; i++) {
    j = mc->paridx[i];
    io->calcvals[j] = mc->pars[i];
  }
  // First check that calcvals are in within limits
  i = 0;
  anybadcheck = 0;
  anybadcheck2 = 0;
  anybadcheck3 = 0;
  io->chi2add = 0.0; // Can be used to calculate a term to add to chi2 in the likelihood function
  while ((i < io->nparm) && (anybadcheck == 0)) {
    if ((io->calcvals[i] < io->calcval_mins[i]) || (io->calcvals[i] > io->calcval_maxs[i])) {
      anybadcheck = 1;
    }
    i++;
  }
  if (anybadcheck == 1) {
    // A parameter is outside calcval limits; set err=1; and skip rest of prior
    err = 1;
  } else {
    // calcvals are within limits. Now calculate physvals
    tmpi = func_physvals(io, pm, mc, ud);
    // ensure that the new physvals are within limts
    i = 0;
    while ((i < io->nparm) && (anybadcheck2 == 0)) {
      if ((io->physvals[i] < io->physval_mins[i]) || (io->physvals[i] > io->physval_maxs[i])) {
        anybadcheck2 = 1;
      }
      i++;
    }
    if (anybadcheck2 == 1) {
      err = 1;
    } else {
      // If one wants to they can now enforce any other kind constraints on the physvals
      // and the calcvals or combinations there of.
      // If a custom violation is found just set err=1;
      // This particular problem does not have any other constraints, thus essentially
      // the placeholder of anybadcheck3==0;
      anybadcheck3 = 0;
    }
  }

  // passed all checks and ready to actually calculate the prior
  // and Jacobian if that is needed. For this problem there is not a jacobian needed.
  if (err == 0) {
    prior = 0.0;
    prior += log(1.0 / (io->physval_maxs[0] - io->physval_mins[0])); // a
    prior += log(1.0 / (io->physval_maxs[1] - io->physval_mins[1])); // b
    prior += log(1.0 / (io->calcval_maxs[2] - io->calcval_mins[2])); // errscl uniform logarithm.
    if (!isfinite(prior)) {
      fprintf(stderr, "Non finite prior detected!\n");
      for (i=0; i<io->nparm; i++) {
        fprintf(stderr, "Parm: %d Value: %lf\n", i, io->physvals[i]);
      }
      exit(EXIT_FAILURE);
    }
  }
  mc->prior = prior;
  return err;
}

int func_calcmodel(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud) {
  // Calculate the problem specific model y values
  // In this case it is a line fit to x-y data.
  // This assumes both physvals and calcvals are populated
  // bounds checking is done in func_prior and during mcmc func_prior is always called before func_calcmodel
  int err=0;
  size_t i;
  double alpha, beta, yValue, xValue;

  alpha = io->calcvals[0];
  beta = io->calcvals[1];
  for (i=0; i<ud->ndata; i++) {
    xValue = gsl_vector_get(ud->xData, i);
    yValue = alpha * xValue + beta;
    gsl_vector_set(ud->yModel, i, yValue);
    if (!isfinite(yValue)) {
      err = 1;
    }
  }
  return err;
}

int func_likehood(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud) {
  // Calculate the problem specific likelihood
  // In this case it is a line fit to x-y data with Gaussian likelihood
  // Assumes both physvals and calcvals are populated
  // ***Also assumes that func_calcmodel has been run and yModel is populated***
  // bounds checking is done in func_prior and during mcmc func_prior is always called before func_calcmodel
  int err = 0;
  double loglike=0.0, chi2=0.0, curresid, errscl, ndatad, overallchi2;
  size_t i, ndatai;

  ndatai = ud->ndata;
  ndatad = (double) ndatai;
  if (err == 0) {
    // Model is valid calculate chi2 and loglikelihood
    loglike = io->likenorm;
    errscl = pow(10.0, io->calcvals[2]);
    for (i=0; i<ndatai; i++) {
      curresid = (gsl_vector_get(ud->yData, i) - gsl_vector_get(ud->yModel, i)) / gsl_vector_get(ud->yErrData, i);
      chi2 += curresid*curresid;
    }
    // Convert chi2 into loglikelihood while also applying the errscl factor to errorbarrs
    loglike = loglike - (ndatad * log(errscl)) - (chi2/2.0/errscl/errscl);
    // Check if there is a new best chi2min
    overallchi2 = chi2 + io->chi2add;
    if (overallchi2 < io->chi2min) {
      if (pm->debugLevel > 0) {
        fprintf(stdout, "New Best Chi2: %lf Old: %lf Expected: %lf\n",
           overallchi2, io->chi2min, io->expchi2);
      }
      io->chi2min = overallchi2;
      for (i=0; i<io->nparm; i++) {
        io->bestphysvals[i] = io->physvals[i];
        io->bestcalcvals[i] = io->calcvals[i];
        if (pm->debugLevel > 0) {
          fprintf(stdout, "%ld Best phys: %lf calc: %lf\n", i, io->physvals[i], io->calcvals[i]);
        }
      }
    }
    if (!isfinite(loglike) || !isfinite(chi2)) {
      err = 1;
    }
  }
  mc->like = loglike;
  mc->chi2 = chi2;

  return err;
}

void func_showmodel(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud, char *plot_output_filename, int show_model) {
  // Using plplot png output
  // **input plot_output_filename should include .png suffix
  double data_xmin, data_xmax, data_ymin, data_ymax;
  double tmp, ubx, uby;
  size_t i, ndata;
  gsl_vector *tmpx, *tmpy;
  gsl_vector_view srcx, srcy;
  
  char *plot_dev = "pngcairo";
  //Plotting formatting
  const char *symbout = "\xE2\x97\x8B"; // UTF-8 circle
  //const char *symbout = "\xE2\xAD\x98"; // heavy circle
  const char *symbfil = "\xE2\x97\x8F"; // black circle
  // PNG output
  double dpi = 400.0; double xWidthIn = 7.0; double yWidthIn = 5.0; // dpi, x axis width in inches
  char *xLabel = "X";
  char *yLabel = "Y";
  PLINT label_family = PL_FCI_SANS; PLINT label_style=PL_FCI_UPRIGHT; PLINT label_weight=PL_FCI_BOLD;
  char *pltTitle = "Linear Model With Gaussian Noise";
  PLINT title_family = PL_FCI_SANS; PLINT title_style=PL_FCI_UPRIGHT; PLINT title_weight=PL_FCI_BOLD;

  double labelScl = 1.5;
  double titleScl = 1.5;
  double mrkScl = 1.5;
  double boxWidth = 3.0;
  double edgebuff_fracx = 0.1; // fraction of data width to add to edges of axis as buffer
  double edgebuff_fracy = 0.1;
  unsigned char bkgclr[] = {0xff, 0xff, 0xff}; // White background
  unsigned char drwclr[] = {0, 0, 0}; // black color
  unsigned char fillclr[] = {61, 156, 235}; // light blue


  // Now lets try to make a figure of the data using plplot
  plscolbg(bkgclr[0], bkgclr[1], bkgclr[2]); // set background color
  plsdev(plot_dev); // Set output device
  plsfnam(plot_output_filename); // plot output file name
  // ***PNG SET DPI***
  plspage(dpi, dpi, lround(dpi*xWidthIn), lround(dpi*yWidthIn), 0, 0);
  plinit(); // initialize plot
  plscol0(15, drwclr[0], drwclr[1], drwclr[2]); // colormap entry 15 to drwclor
  plcol0(15); // switch to drwclr for drawing axis
  // Get the min and max of the x & y data vectors
  ndata = ud->ndata;
  data_xmin = gsl_vector_get(ud->xData, 0);
  data_xmax = gsl_vector_get(ud->xData, 0);
  data_ymin = gsl_vector_get(ud->yData, 0);
  data_ymax = gsl_vector_get(ud->yData, 0);
  for (i=0; i<ndata; i++) {
    tmp = gsl_vector_get(ud->xData, i);
    if (tmp < data_xmin) {
      data_xmin = tmp;
    }
    if (tmp > data_xmax) {
      data_xmax = tmp;
    }
    tmp = gsl_vector_get(ud->yData, i);
    if (tmp < data_ymin) {
      data_ymin = tmp;
    }
    if (tmp > data_ymax) {
      data_ymax = tmp;
    }
  }

  ubx = fabs((data_xmax - data_xmin) * edgebuff_fracx);
  uby = fabs((data_ymax - data_ymin) * edgebuff_fracy);
  // define data coordinates of plot box
  plsfont(label_family, label_style, label_weight); // font format
  plwidth(boxWidth); // set box line width
  plenv(data_xmin - ubx, data_xmax + ubx, data_ymin - uby, data_ymax + uby, 0, 0);
  // Labels
  plschr(0.0, labelScl); // scale labels
  plsfont(label_family, label_style, label_weight); // font format
  plmtex("b", 2.0, 0.5, 0.5, xLabel); // x-axis
  plmtex("l", 2.5, 0.5, 0.5, yLabel);
  plschr(0.0, 1.0/labelScl); // undo scale labels

  // Title
  plschr(0.0, titleScl); // scale title
  plsfont(title_family, title_style, title_weight); // font format
  plmtex("t", 1.0, 0.5, 0.5, pltTitle); // Title
  plschr(0.0, 1.0/titleScl); // undo scale labels

  //plot data
  plschr(0.0, mrkScl); // resize markers
  plscol0(15, fillclr[0], fillclr[1], fillclr[2]); // fill color
  plcol0(15);
  plstring(ndata, gsl_vector_ptr(ud->xData,0), gsl_vector_ptr(ud->yData,0), symbfil); // plot fill markers
  plscol0(15, drwclr[0], drwclr[1], drwclr[2]); // draw color
  plcol0(15);
  plstring(ndata, gsl_vector_ptr(ud->xData,0), gsl_vector_ptr(ud->yData,0), symbout); // plot outline markers
  if (show_model == 1) {
    // Draw model line
    // This could be done much easier with just the min & max x data values
    // Doing this complicated way to show how to do it for more complicated model
    // That is not just a straight line
    // Need to sort x data
    tmpx = gsl_vector_alloc(ndata);
    tmpy = gsl_vector_alloc(ndata);
    // Before copying data over to temporary gsl_vectors need to convert
    // data vector to a vector view with same length
    srcx = gsl_vector_subvector(ud->xData, 0, ndata);
    srcy = gsl_vector_subvector(ud->yModel, 0, ndata);
    gsl_vector_memcpy(tmpx, &(srcx.vector));
    gsl_vector_memcpy(tmpy, &(srcy.vector));
    gsl_sort_vector2(tmpx, tmpy);
    plline(ndata, gsl_vector_ptr(tmpx,0), gsl_vector_ptr(tmpy,0));
    gsl_vector_free(tmpx);
    gsl_vector_free(tmpy);
  }
  plschr(0.0, 1.0/mrkScl); // undo symbol scaling
  plend();

}

size_t guess_data_length(const char *fileInput, int nSample) {
  struct stat st;
  off_t fileSize;
  const char commChar[] = "#"; // Beginning of line comment character
  size_t i, commLines, dataLines, sumLineLen, lineLenEst, sumCommLen, tmp2;
  FILE *fp;
  char lineBuff[MAX_LINE];
  char *tmp;
  double avgLineLen;

  // Get the fileSize in bytes
  if (stat(fileInput, &st) == 0) {
    fileSize = st.st_size;
  } else {
    fprintf(stderr, "Cannot determine size of %s: %s\n",
        fileInput, strerror(errno));
    exit(EXIT_FAILURE);
  }

  // Read nSample non comment lines to calculate the mean line length
  fp = fopen(fileInput, "r");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open file %s: %s\n",
      fileInput, strerror(errno));
    exit(EXIT_FAILURE);
  }
  // Check for initial comment lines
  commLines = 0;
  sumCommLen = 0;
  do {
      tmp = fgets(lineBuff, MAX_LINE, fp);
      // fgets returns null in case of EOF or error check for these
      if (tmp == NULL) {
        fprintf(stderr, "Unexpected end-of-file or error reading file %s: %s\n",
          fileInput, strerror(errno));
        exit(EXIT_FAILURE);
      }
      // Ensure the line is not too long to exceed buffer
      tmp2 = strlen(lineBuff);
      if (tmp2 >= MAX_LINE-1) {
        fprintf(stderr,"Line %zu in file %s exceeds maximum allowed line length: %d\n",
            commLines+1, fileInput, MAX_LINE);
        exit(EXIT_FAILURE);
      }
      sumCommLen = sumCommLen + tmp2;
      commLines++;
  } while (lineBuff[0] == commChar[0]);
  commLines--;
  sumCommLen = sumCommLen - tmp2;
  fprintf(stdout,"Found %zu Header Lines using %zu bytes\n", commLines, sumCommLen);

  // Read in nSample lines to calculate average line length
  // or if EOF encountered before nSample return actual number of lines
  // In finding comment lines one non-comment line already read
  sumLineLen = 0;
  dataLines = 1;
  lineLenEst = 0;
  do {
    sumLineLen = sumLineLen + strlen(lineBuff);
    // read next line
    tmp = fgets(lineBuff, MAX_LINE, fp);
    // check for EOF or error
    if (tmp == NULL) {
      // Check for EOF
      if (feof(fp)) { // End of file detected
        // We got the exact line length
        lineLenEst = dataLines;
      } else { // Error reading file, exit
        fprintf(stderr, "Error reading line %zu in file %s\n",
            commLines+dataLines, fileInput);
        exit(EXIT_FAILURE);
      }
    }
    // Ensure the line is not too long to exceed buffer
    if (strlen(lineBuff) >= MAX_LINE-1) {
      fprintf(stderr,"Line %zu in file %s exceeds maximum allowed line length: %d\n",
          commLines+1, fileInput, MAX_LINE);
      exit(EXIT_FAILURE);
    }
    dataLines++;
  } while ((dataLines <= nSample) & (lineLenEst == 0));

  // If line length was not encountered before nSample, then
  // calculate the average line length
  if (lineLenEst == 0) {
    avgLineLen = ((double) sumLineLen) / ((double) nSample);
    lineLenEst = (size_t) (((double) (fileSize - sumCommLen)) / avgLineLen);
  }

  fclose(fp);
  return lineLenEst;
}

size_t read_data_file(const char *fileInput, gsl_vector *xData, gsl_vector *yData, size_t maxN) {
  const char commChar[] = "#"; // Beginning of line comment character
  FILE *fp;
  char lineBuff[MAX_LINE];
  char *tmp;
  size_t i, actN, readN, readOK;
  double x, y;

  // Read nSample non comment lines to calculate the mean line length
  fp = fopen(fileInput, "r");
  if (fp == NULL) {
    fprintf(stderr, "Cannot open file %s: %s\n",
      fileInput, strerror(errno));
    exit(EXIT_FAILURE);
  }
  // Check for initial comment lines
  do {
      tmp = fgets(lineBuff, MAX_LINE, fp);
      // fgets returns null in case of EOF or error check for these
      if (tmp == NULL) {
        fprintf(stderr, "Unexpected end-of-file or error reading file %s: %s\n",
          fileInput, strerror(errno));
        exit(EXIT_FAILURE);
      }
  } while (lineBuff[0] == commChar[0]);
  // The last line read was not a comment, therefore copy results to data arrays
  i = 0;
  actN = 1;
  readN = sscanf(lineBuff, "%lf %lf", &x, &y);
  gsl_vector_set(xData, i, x);
  gsl_vector_set(yData, i, y);
  // No read through rest of file storing results in arrays
  readOK = 1;
  do {
    readN = fscanf(fp, "%lf %lf", &x, &y);
    // if read in the expected 2 elements add them to vectors
    if (readN == 2) {
      i++;
      actN++;
      // Make sure this won't exceed vector storage space
      if (actN > maxN) {
        fprintf(stderr, "The number of lines read %zu exceeds the storage space allotted %zu\n",
          actN, maxN);
          exit(EXIT_FAILURE);
      }
      // ensure data is finite
      if (isfinite(x) & isfinite(y)) {
        gsl_vector_set(xData, i, x);
        gsl_vector_set(yData, i, y);
      } else {
        fprintf(stderr, "Non Finite value encountered on data line %zu\n", actN);
        exit(EXIT_FAILURE);
      }
    } else { // Check for EOF, not getting the expected 2 elements, and unexpected errors
      if (feof(fp)) { // EOF detected
        readOK = 0;
      } else {
        if (ferror(fp)) { // Error detected
          fprintf(stderr, "Error reading %s at data line %zu\n",
              fileInput, actN);
          exit(EXIT_FAILURE);
        } else { // Must have read non expected number of elements
          fprintf(stderr, "Did not get the expected number of elements\n");
          exit(EXIT_FAILURE);
        }
      }
    }
  } while (readOK);

  fclose(fp);
  return actN;
}

void data_load_format(char *fileInput, gsl_vector **xData, gsl_vector **yData, size_t *actN) {
  size_t guessN, maxN;
  double overNFrac = 1.1;

  guessN = guess_data_length(fileInput, 10);
  fprintf(stdout,"Input File %s has %zu estimated lines\n",
      fileInput, guessN);
  // Multiply guessn by overNFrac to leave some breathing room
  maxN = (size_t) guessN * overNFrac;
  // Allocate arrays for data
  *xData = gsl_vector_alloc(maxN);
  *yData = gsl_vector_alloc(maxN);

  *actN = read_data_file(fileInput, *xData, *yData, maxN);
  fprintf(stdout, "Found %zu lines %lf %lf\n", *actN, gsl_vector_get(*xData,*actN-1), gsl_vector_get(*yData, *actN-1));

  return;
}

void data_prep_scale(gsl_vector *xData, gsl_vector *yData, size_t actN, double *scls, double *zpts, double *mins, double *maxs) {
  // Use single pass gsl routines to get statistics of data vectors
  size_t i;
  gsl_rstat_workspace *rstat_p = gsl_rstat_alloc();

  // get xData stats
  for (i=0;i<actN;i++) {
    gsl_rstat_add(gsl_vector_get(xData, i), rstat_p);
  }
  zpts[0] = gsl_rstat_mean(rstat_p);
  scls[0] = gsl_rstat_sd(rstat_p);
  mins[0] = gsl_rstat_min(rstat_p);
  maxs[0] = gsl_rstat_max(rstat_p);
  // apply zeropoint and scaling to xData
  gsl_vector_add_constant(xData, -zpts[0]);
  gsl_vector_scale(xData, 1.0/scls[0]);
  mins[0] = (mins[0] - zpts[0]) / scls[0];
  maxs[0] = (maxs[0] - zpts[0]) / scls[0];


  // get yData stats
  gsl_rstat_reset(rstat_p);
  for (i=0;i<actN;i++) {
    gsl_rstat_add(gsl_vector_get(yData, i), rstat_p);
  }
  zpts[1] = gsl_rstat_mean(rstat_p);
  scls[1] = gsl_rstat_sd(rstat_p);
  mins[1] = gsl_rstat_min(rstat_p);
  maxs[1] = gsl_rstat_max(rstat_p);
  // apply zeropoint and scaling to yData
  gsl_vector_add_constant(yData, -zpts[1]);
  gsl_vector_scale(yData, 1.0/scls[1]);
  mins[1] = (mins[1] - zpts[1]) / scls[1];
  maxs[1] = (maxs[1] - zpts[1]) / scls[1];


  //cleanup memory
  gsl_rstat_free(rstat_p);
  return;
}

int main(int argc, char const *argv[]) {
  char *fileInput = "/Users/cjburke/Work/pgmcmc_forc/build_xcode/Debug/gen_linear_data.txt";
  char *outputFilename = "pgmcmc_linfit.h5";
  gsl_vector *xData, *yData, *yErrData, *yModel;
  size_t actN;
  double dataScales[2], dataZpts[2], dataMins[2], dataMaxs[2];
  pgmcmc_ioblk ioblk;
  pgmcmc_userdata userdata;
  pgmcmc_algo_params  algo_params;
  pgmcmc_mcmc mcmc;
  pgmcmc_userfuncs userfuncs;
  // Hard coded problem specific parameter names
  // linear model y=ax+b ; a - slope; b - constant;
  // errscl error scaling; x0,xs data is centered and scaled with x0 and xs
  // these parameters will be fixed during fit and they are here
  // just to conveniently have them available when converting scaled data values
  // to unscaled data values during the runtime.
  // same with y0 & ys
  char physval_names[NPARM][MAX_NAME_LEN] = {"a","b","errscl","x0","xs","y0","ys"};
  char calcval_names[NPARM][MAX_NAME_LEN] = {"alpha","beta","log10errslc","x0_c","xs_c","y0_c","ys_c"};
  double origests[NPARM], physval_mins[NPARM], physval_maxs[NPARM];
  double scls[NPARM], calcval_mins[NPARM], calcval_maxs[NPARM];
  int fixed[NPARM];
  double physvals[NPARM], physvalsavs[NPARM], bestphysvals[NPARM];
  double calcvals[NPARM], calcvalsavs[NPARM], bestcalcvals[NPARM];
  double pars[NPARM], curp[NPARM], sigs[NPARM], nexp[NPARM];
  int paridx[NPARM];
  long attempts[NPARM], accepts[NPARM];
  double fracs[NPARM];

  double sclrat, sclrat2, tmp1, tmp2;
  double c0, c1, cov00, cov01, cov11, sumsq;
  int i, tmpi;


  // xData and yData are allocated in here after estimating vector size from a partial read
  // For the data_load_format function call the gsl_vectors need their addresses
  // this is only because we are allocating them in the data_load_format function and we need to keep
  // the original addresses to free them later outside the function.
  data_load_format(fileInput, &xData, &yData, &actN);

  // subtract the mean of each vector and scale by standard deviation
  // return these data scalings and zeropoints
  data_prep_scale(xData, yData, actN, dataScales, dataZpts, dataMins, dataMaxs);
  //fprintf(stdout, "X mean: %lf sd: %lf Y mean: %lf sd: %lf\n",
  //    dataZpts[0], dataScales[0], dataZpts[1], dataScales[1]);

  // Do linear regression fit using gsl routine
  tmpi = gsl_fit_linear(gsl_vector_ptr(xData, 0), 1, gsl_vector_ptr(yData,0), 1, actN, &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
  printf("co: %g c1: %g c00: %g c01: %g c11: %g sumsq: %g\n", c0, c1, cov00, cov01, cov11, sumsq);
  
  // Need to start loading the problem specific things into ioblk
  // original estimates and min max
  // **NOTE in this example the origests are in calcvals system
  // rather than physvals
  // a - alpha
  origests[0] = 0.0;
  calcval_mins[0] = -1.0e3;
  calcval_maxs[0] = 1.0e3;
  sclrat = dataScales[1]/dataScales[0];
  physval_mins[0] = calcval_mins[0] * sclrat;
  physval_maxs[0] = calcval_maxs[0] * sclrat;
  fixed[0] = 0;
  scls[0] = 1.0 / sqrt((double) actN);
  // b - beta
  origests[1] = 0.0;
  calcval_mins[1] = dataMins[1];
  calcval_maxs[1] = dataMaxs[1];
  sclrat2 = sclrat * dataZpts[0];
  if (sclrat2 >= 0.0) {
    tmp1 = sclrat2 * calcval_maxs[0];
    tmp2 = sclrat2 * calcval_mins[0];
  } else {
    tmp1 = sclrat2 * calcval_mins[0];
    tmp2 = sclrat2 * calcval_maxs[0];
  }
  physval_mins[1] = dataScales[1] * calcval_mins[1] + dataZpts[1] - tmp1;
  physval_maxs[1] = dataScales[1] * calcval_maxs[1] + dataZpts[1] - tmp2;
  fixed[1] = 0;
  scls[1] = 1.0 / sqrt((double) actN);
  // errscl - log10(errscl)
  origests[2] = 0.0;
  calcval_mins[2] = -4.0;
  calcval_maxs[2] = 4.0;
  physval_mins[2] = 1.0e-4 * dataScales[1];
  physval_maxs[2] = 1.0e4 * dataScales[1];
  fixed[2] = 0;
  scls[2] = 0.2;
  // data zeropoints and data scales
  origests[3] = dataZpts[0];
  calcval_mins[3] = origests[3] - 1.0;
  calcval_maxs[3] = origests[3] + 1.0;
  physval_mins[3] = calcval_mins[3];
  physval_maxs[3] = calcval_maxs[3];
  fixed[3] = 1;
  scls[3] = 1.0;
  origests[4] = dataScales[0];
  calcval_mins[4] = origests[4] - 1.0;
  calcval_maxs[4] = origests[4] + 1.0;
  physval_mins[4] = calcval_mins[4];
  physval_maxs[4] = calcval_maxs[4];
  fixed[4] = 1;
  scls[4] = 1.0;
  origests[5] = dataZpts[1];
  calcval_mins[5] = origests[5] - 1.0;
  calcval_maxs[5] = origests[5] + 1.0;
  physval_mins[5] = calcval_mins[5];
  physval_maxs[5] = calcval_maxs[5];
  fixed[5] = 1;
  scls[5] = 1.0;
  origests[6] = dataScales[1];
  calcval_mins[6] = origests[6] - 1.0;
  calcval_maxs[6] = origests[6] + 1.0;
  physval_mins[6] = calcval_mins[6];
  physval_maxs[6] = calcval_maxs[6];
  fixed[6] = 1;
  scls[6] = 1.0;

  // Allocate data for errors and model storage
  yErrData = gsl_vector_alloc(actN);
  gsl_vector_set_all(yErrData, 1.0);
  yModel = gsl_vector_alloc(actN);

  // Load ioblk
  ioblk.nparm = NPARM;
  ioblk.origests = origests;
  ioblk.fixed = fixed;
  ioblk.physval_mins = physval_mins;
  ioblk.physval_maxs = physval_maxs;
  ioblk.physvals = physvals;
  ioblk.physvalsavs = physvalsavs;
  ioblk.bestphysvals = bestphysvals;
  ioblk.calcvals = calcvals;
  ioblk.calcvalsavs = calcvalsavs;
  ioblk.bestcalcvals = bestcalcvals;
  ioblk.calcval_mins = calcval_mins;
  ioblk.calcval_maxs = calcval_maxs;
  ioblk.scls = scls;

  // Load userdata
  userdata.ndata = actN;
  userdata.xData = xData;
  userdata.yData = yData;
  userdata.yErrData = yErrData;
  userdata.yModel = yModel;

  // copy the origests to ioblk.calcvals
  for (i=0; i<NPARM; i++) {
    ioblk.calcvals[i] = ioblk.origests[i];
  }
  // copy calcvals to physvals using the user provded func_physvals
  tmpi = func_physvals(&ioblk, &algo_params, &mcmc, &userdata);
  //printf("%d %lf %lf %lf %lf %lf %lf %lf\n", tmpi, ioblk.physval_mins[0], ioblk.physval_mins[1],
  //  ioblk.physval_mins[2], ioblk.physval_mins[3], ioblk.physval_mins[4], ioblk.physval_mins[5], ioblk.physval_mins[6]);
  //printf("%d %lf %lf %lf %lf %lf %lf %lf\n", tmpi, ioblk.physvals[0], ioblk.physvals[1],
  //  ioblk.physvals[2], ioblk.physvals[3], ioblk.physvals[4], ioblk.physvals[5], ioblk.physvals[6]);
  //printf("%d %lf %lf %lf %lf %lf %lf %lf\n", tmpi, ioblk.physval_maxs[0], ioblk.physval_maxs[1],
  //  ioblk.physval_maxs[2], ioblk.physval_maxs[3], ioblk.physval_maxs[4], ioblk.physval_maxs[5], ioblk.physval_maxs[6]);

  // Set the mcmc algorithm parameters
  algo_params.outputFilename = outputFilename; // Output hdf5 chain data filename
  algo_params.debugLevel = 2; // 3 - Debug level setting
  algo_params.maxstps = 300000; // 300000 - Maximum number of MCMC steps in run
  algo_params.fracwant = 0.25; // 0.25 - Target for fraction of steps accepted
  algo_params.initNSteps = 200; // 200 - burn in before starting to gather any stats
  algo_params.coarseNSteps = 100; // 100 - Initial Coarse solving for mcmc step scales. number of steps before calculating accepted fraction
  algo_params.coarseLowFrac = 0.1; // 0.1 - acceptable range for coarse scale solving
  algo_params.coarseHghFrac = 0.8; // 0.8
  algo_params.refineNSteps = 400; // 400 - Refined mcmc step scale solving number of steps
  algo_params.refineLowFrac = 0.15; // 0.15  - acceptabel range for refined scale solving
  algo_params.refineHghFrac = 0.42; // 0.42
  algo_params.maxPropTries = 50; // 50 - Number of tries to adjusting step sizes before giving up
  algo_params.saveNSteps = 100; // 100 - Write out results after this many steps

  // assign mcmc algorithm arrays
  mcmc.pars = pars;
  mcmc.paridx = paridx;
  mcmc.attempts = attempts;
  mcmc.accepts = accepts;
  mcmc.curp = curp;
  mcmc.sigs = sigs;
  mcmc.nexp = nexp;
  mcmc.fracs = fracs;
  // Seed the mcmc random number generation
  mcmc.ranseed = 440000033;
  mcmc.ranobj = gsl_rng_alloc(gsl_rng_mt19937); // Mersenne twister
  gsl_rng_set(mcmc.ranobj, mcmc.ranseed);
  
  // Assign the problem specific user functions
  userfuncs.func_physvals = &func_physvals;
  userfuncs.func_calcvals = &func_calcvals;
  userfuncs.func_prior = &func_prior;
  userfuncs.func_calcmodel = &func_calcmodel;
  userfuncs.func_likehood = &func_likehood;

  tmpi = pgmcmc_setup(&ioblk, &algo_params, &mcmc, &userdata, &userfuncs);
  
  // Make figure showing initial data being fit without model
  if (algo_params.debugLevel > 1) {
    func_showmodel(&ioblk, &algo_params, &mcmc, &userdata, "pgmcmc_forc_initdata.png", 0);
  }
  tmpi = pgmcmc_run_mcmc(&ioblk, &algo_params, &mcmc, &userdata, &userfuncs);
  
  // Make figure showing best fitting model in chi2 sense
  if (algo_params.debugLevel > 1) {
    func_showmodel(&ioblk, &algo_params, &mcmc, &userdata, "pgmcmc_forc_bestfit.png", 1);
  }
  
  tmpi = pgmcmc_report(&ioblk, &algo_params, &mcmc, &userdata, &userfuncs, physval_names, calcval_names);
  
  gsl_vector_free(xData);
  gsl_vector_free(yData);
  gsl_vector_free(yErrData);
  gsl_vector_free(yModel);

  return 0;
}
