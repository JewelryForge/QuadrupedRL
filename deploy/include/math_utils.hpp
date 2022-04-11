#ifndef QUADRUPED_DEPLOY_INCLUDE_MATH_UTILS_HPP_
#define QUADRUPED_DEPLOY_INCLUDE_MATH_UTILS_HPP_
#include <cmath>
#include <array>
#include <Eigen/Core>

namespace mu {
using std::size_t;

template<size_t N>
using fVec = Eigen::Array<float, 1, N>;
template<size_t N>
using fArr = std::array<float, N>;


using Vec3 = fVec<3>;
using Vec4 = fVec<4>;
using Vec8 = fVec<8>;
using Vec12 = fVec<12>;
using Vec24 = fVec<24>;

constexpr float PI = M_PI;
constexpr float TAU = M_PI * 2;

template<size_t N>
float dot(const fVec<N> array1, const fVec<N> &array2) {
  float sum = 0;
  for (int i = 0; i < N; i++) {
    sum += array1[i] * array2[i];
  }
  return sum;
}

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
}

#endif //QUADRUPED_DEPLOY_INCLUDE_MATH_UTILS_HPP_
