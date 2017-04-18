#include "omp.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

/*
  from intel website
  https://software.intel.com/en-us/articles/cache-blocking-techniques

*/

/*
reciprocal if non zero
from stackoverflow:
http://stackoverflow.com/questions/10606483/sse-reciprocal-if-not-zero
__m128 rcp_nz_ps(__m128 input) {
    __m128 mask = _mm_cmpeq_ps(_mm_set1_ps(0.0), input);
    __m128 recip = _mm_rcp_ps(input);
    return _mm_andnot_ps(mask, recip);
}

*/
// simple implementation
template <typename Real>
void n2CPU(const std::vector<Real> &posSrc, const std::vector<Real> &posTrg,
           const std::vector<Real> &valueSrc, std::vector<Real> &valueTrg) {
  const size_t nSrc = posSrc.size() / 3;
  const size_t nTrg = posTrg.size() / 3;
#pragma omp parallel for
  for (size_t i = 0; i < nTrg; i++) {
    Real vtrg = 0;
#pragma omp simd
    for (size_t j = 0; j < nSrc; j++) {
      Real dx = posTrg[3 * i] - posSrc[3 * j];
      Real dy = posTrg[3 * i + 1] - posSrc[3 * j + 1];
      Real dz = posTrg[3 * i + 2] - posSrc[3 * j + 2];
      Real r2 = dx * dx + dy * dy + dz * dz;
      Real rinv = r2 > 1e-14 ? 1 / sqrt(r2) : 0;
      // Real rinv = 1 / sqrt(r2); // not faster
      vtrg += valueSrc[j] * rinv;
    }
    valueTrg[i] = vtrg;
  }
}



int main(int argc, char **argv) {
  int nSrc = 16384;
  int nTrg = nSrc;

  if (argc > 1) {
    int temp = atoi(argv[1]);
    nSrc = temp < 0 ? -temp : temp;
  }
  if (argc > 2) {
    int temp = atoi(argv[2]);
    nTrg = temp < 0 ? -temp : temp;
  }

  printf("N Src: %d, N Trg: %d\n", nSrc, nTrg);

  using Real = double;
  std::vector<Real> posSrc(nSrc * 3);
  std::vector<Real> posTrg(nTrg * 3);
  std::vector<Real> valueSrc(nSrc);       // simple 1/r kernel
  std::vector<Real> valueTrg(nTrg, 0);    // simple 1/r kernel, 0 initialized
  std::vector<Real> valueTrgGPU(nTrg, 0); // simple 1/r kernel, 0 initialized

  // random initialize
  for (auto &p : posSrc) {
    p = drand48();
  }
  for (auto &p : posTrg) {
    p = drand48();
  }
  for (auto &p : valueSrc) {
    p = drand48();
  }

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  n2CPU<Real>(posSrc, posTrg, valueSrc, valueTrg);

  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

  std::cout << "N2CPU: " << time_span.count() << " seconds.";
  std::cout << std::endl;

  return 0;
}
