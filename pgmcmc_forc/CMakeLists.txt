cmake_minimum_required(VERSION 3.0)
project(pgmcmc_forc VERSION 0.1
                  DESCRIPTION "Pretty Good MCMC (PGMCMC) C example")

set(CMAKE_BUILD_TYPE DEBUG)

# define the executables
add_executable(gen_linear_data
        src/gen_linear_data.c)

add_executable(pgmcmc_forc src/pgmcmc_forc.c src/pgmcmc_core.c src/pgmcmc_report.c)

# find the gsl library
find_library(GSL_LIBRARY gsl)
if(GSL_LIBRARY)
    # link target with gsl library
    target_link_libraries(gen_linear_data PUBLIC ${GSL_LIBRARY})
    target_link_libraries(pgmcmc_forc PUBLIC ${GSL_LIBRARY})
endif()

# find the gslcblas library
find_library(GSLCBLAS_LIBRARY gslcblas)
if(GSLCBLAS_LIBRARY)
    # link target with gslcbls library
    target_link_libraries(gen_linear_data PUBLIC ${GSLCBLAS_LIBRARY})
    target_link_libraries(pgmcmc_forc PUBLIC ${GSLCBLAS_LIBRARY})
endif()

# find the plplot library
find_library(PLPLOT_LIBRARY plplot)
if (PLPLOT_LIBRARY)
  target_link_libraries(gen_linear_data PUBLIC ${PLPLOT_LIBRARY})
  target_link_libraries(pgmcmc_forc PUBLIC ${PLPLOT_LIBRARY})
endif()

# find the hdf5 library
find_library(HDF5_LIBRARY hdf5)
if (HDF5_LIBRARY)
  target_link_libraries(pgmcmc_forc PUBLIC ${HDF5_LIBRARY})
endif()


# define header file path
target_include_directories(pgmcmc_forc PRIVATE include)
