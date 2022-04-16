#ifndef QUADRUPED_DEPLOY_INCLUDE_MATH_UTILS_HPP_
#define QUADRUPED_DEPLOY_INCLUDE_MATH_UTILS_HPP_
#include <cmath>
#include <array>
#include <Eigen/Core>

template<std::size_t N>
using fArray = Eigen::Array<float, N, 1>;

using Array3 = fArray<3>;
using Array4 = fArray<4>;
using Array8 = fArray<8>;
using Array12 = fArray<12>;
using Array24 = fArray<24>;

constexpr float PI = M_PI;
constexpr float TAU = M_PI * 2;

float ang_norm(float x) {
  if (-PI <= x and x < PI) return x;
  return x - int((x + PI) / TAU) * TAU;
}

template<typename T>
T clip(T x, T min, T max) {
  if (x > max) return max;
  if (x < min) return min;
  return x;
}

template<typename T>
inline T pow2(T n) { return n * n; }

#endif //QUADRUPED_DEPLOY_INCLUDE_MATH_UTILS_HPP_
