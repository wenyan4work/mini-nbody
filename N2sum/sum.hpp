
#ifndef SUM_HPP_
#define SUM_HPP_

#include "cuda.h"
#include "cuda_runtime.h"
#include <vector>

void dn2CUDA(const std::vector<double> &posSrc,
             const std::vector<double> &posTrg,
             const std::vector<double> &valueSrc,
             std::vector<double> &valueTrg);

void fn2CUDA(const std::vector<float> &posSrc, const std::vector<float> &posTrg,
             const std::vector<float> &valueSrc, std::vector<float> &valueTrg);
#endif