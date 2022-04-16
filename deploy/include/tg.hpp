#ifndef QUADRUPED_DEPLOY_INCLUDE_TG_H_
#define QUADRUPED_DEPLOY_INCLUDE_TG_H_
#include <cmath>
#include <utility>
#include <vector>
#include <array>
#include "math_utils.hpp"
#include "io.hpp"

class TG {
 public:
  virtual Array12 getPriori(const Array4 &phases) const = 0;
};

class VerticalTG : public TG {
 public:
  explicit VerticalTG(float h = 0.08) : h_(h) {}

  Array12 getPriori(const Array4 &phases) const override {
    Array12 out = Array12::Zero();
    Array4 ks = phases * 2 / PI;
    for (int i = 0; i < 4; ++i) {
      float k = ks[i];
      if (0 < k and k < 2) {
        Array4 k_pow{1};
        for (int j = 1; j < 4; j++) k_pow[j] = k_pow[j - 1] * k;
        if (k <= 1) out[i * 3 + 2] = k_pow.matrix().dot(coeff1.matrix()) * h_;
        else out[i * 3 + 2] = k_pow.matrix().dot(coeff2.matrix()) * h_;
      }
    }
    return out;
  }

 private:
  float h_;
  const Array4 coeff1{0., 0., 3., -2.};
  const Array4 coeff2{-4., 12., -9., 2.};
};

class TgStateMachine {
 public:
  TgStateMachine(std::shared_ptr<TG> tg, float base_freq, Array4 init_phases,
                 float lower_freq = 0.5, float upper_freq = 3.0)
      : tg_(std::move(tg)), base_freq_(base_freq), phases_(std::move(init_phases)),
        lower_freq(lower_freq), upper_freq(upper_freq) {
    freq_.fill(base_freq);
  };

  const Array4 &freq = freq_;
  const Array4 &phases = phases_;
  const float &base_freq = base_freq_;
  const float lower_freq, upper_freq;

  const Array4 &update(float time_step) {
    phases_ += freq_ * TAU * time_step;
    for (int i = 0; i < 4; ++i) {
      phases_[i] = ang_norm(phases_[i]);
    }
    return phases_;
  };

  const Array4 &update(float time_step, const Array4 &freq_offsets) {
    for (int i = 0; i < 4; ++i) freq_[i] = clip(base_freq_ + freq_offsets[i], lower_freq, upper_freq);
    return update(time_step);
  };

  Array12 getPrioriTrajectory() const {
    return tg_->getPriori(phases);
  }

  void getSoftPhases(Eigen::Ref<Array8> out) {
    out.segment<4>(0) = phases_.sin();
    out.segment<4>(4) = phases_.cos();
  }

 private:
  std::shared_ptr<TG> tg_;
  float base_freq_;
  fArray<4> freq_{}, phases_{};
};

#endif //QUADRUPED_DEPLOY_INCLUDE_TG_H_
