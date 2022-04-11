#ifndef QUADRUPED_DEPLOY_INCLUDE_STATE_HPP_
#define QUADRUPED_DEPLOY_INCLUDE_STATE_HPP_

#include "math_utils.hpp"

struct ProprioObservation {
  static constexpr uint dim = 36;
  using array_type = mu::fVec<dim>;
  mu::Vec3 command;
  mu::Vec3 gravity_vector;
  mu::Vec3 base_linear;
  mu::Vec3 base_angular;
  mu::Vec12 joint_pos;
  mu::Vec12 joint_vel;

  std::shared_ptr<array_type> standard() const {
    auto array = std::make_shared<array_type>();
    array->segment<3>(0) = command;
    array->segment<3>(3) = gravity_vector;
    array->segment<3>(6) = base_linear;
    array->segment<3>(9) = base_angular;
    array->segment<12>(12) = joint_pos;
    array->segment<12>(24) = joint_vel;
    *array -= biases;
    *array *= weights;
    return array;
  }

  static const mu::fVec<dim> weights, biases;
};

struct ProprioInfo : public ProprioObservation {
  static constexpr uint dim = 60;
  using array_type = mu::fVec<dim>;
  mu::Vec12 joint_pos_target;
  mu::Vec8 ftg_phases;
  mu::Vec4 ftg_frequencies;
  static const mu::fVec<dim> weights, biases;

  std::shared_ptr<array_type> standard() const {
    auto array = std::make_shared<array_type>();
    array->segment<3>(0) = command;
    array->segment<3>(3) = gravity_vector;
    array->segment<3>(6) = base_linear;
    array->segment<3>(9) = base_angular;
    array->segment<12>(12) = joint_pos;
    array->segment<12>(24) = joint_vel;
    array->segment<12>(36) = joint_pos_target;
    array->segment<8>(48) = ftg_phases;
    array->segment<4>(56) = ftg_frequencies;
    *array -= biases;
    *array *= weights;
    return array;
  }
};

struct RealWorldObservation : public ProprioInfo {
  static constexpr uint dim = 60 + 73;
  using array_type = mu::fVec<dim>;
  mu::Vec12 joint_prev_pos_err;
  mu::Vec24 joint_pos_err_his;
  mu::Vec24 joint_vel_his;
  mu::Vec12 joint_prev_pos_target;
  mu::fVec<1> base_frequency;
  static const mu::fVec<dim> weights, biases;

  std::shared_ptr<array_type> standard() const {
    auto array = std::make_shared<array_type>();
    array->segment<3>(0) = command;
    array->segment<3>(3) = gravity_vector;
    array->segment<3>(6) = base_linear;
    array->segment<3>(9) = base_angular;
    array->segment<12>(12) = joint_pos;
    array->segment<12>(24) = joint_vel;
    array->segment<12>(36) = joint_pos_target;
    array->segment<8>(48) = ftg_phases;
    array->segment<4>(56) = ftg_frequencies;
    array->segment<12>(60) = joint_prev_pos_err;
    array->segment<24>(72) = joint_pos_err_his;
    array->segment<24>(96) = joint_vel_his;
    array->segment<12>(120) = joint_prev_pos_target;
    array->segment<1>(132) = base_frequency;
    *array -= biases;
    *array *= weights;
    return array;
  }
};

const mu::fVec<RealWorldObservation::dim> RealWorldObservation::weights = {
    {1., 1., 1., 5., 5., 20.,
     2., 2., 2., 2., 2., 2.,
     2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
     0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3,
     2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
     1., 1., 1., 1., 1., 1., 1., 1., 100., 100., 100., 100.,
     6.5, 4.5, 3.5, 6.5, 4.5, 3.5, 6.5, 4.5, 3.5, 6.5, 4.5, 3.5,
     5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
     5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
     0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3,
     0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3,
     2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
     1.
    }};
const mu::fVec<RealWorldObservation::dim> RealWorldObservation::biases = {
    {0., 0., 0., 0., 0., 0.99,
     0., 0., 0., 0., 0., 0.,
     0., 0.6435, -1.287, 0., 0.6435, -1.287, 0., 0.6435, -1.287, 0., 0.6435, -1.287,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0.6435, -1.287, 0., 0.6435, -1.287, 0., 0.6435, -1.287, 0., 0.6435, -1.287,
     0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0.6435, -1.287, 0., 0.6435, -1.287, 0., 0.6435, -1.287, 0., 0.6435, -1.287,
     2.
    }};

const mu::fVec<ProprioInfo::dim> ProprioInfo::weights = RealWorldObservation::weights.segment<ProprioInfo::dim>(0);
const mu::fVec<ProprioInfo::dim> ProprioInfo::biases = RealWorldObservation::biases.segment<ProprioInfo::dim>(0);
const mu::fVec<ProprioObservation::dim>
    ProprioObservation::weights = ProprioInfo::weights.segment<ProprioObservation::dim>(0);
const mu::fVec<ProprioObservation::dim>
    ProprioObservation::biases = ProprioInfo::biases.segment<ProprioObservation::dim>(0);

template<typename T, std::size_t N>
class StaticQueue {
 public:
  static constexpr std::size_t S = N + 1;
  StaticQueue() = default;
  struct QueueOverflow : public std::exception {
    const char *what() const noexcept override { return "deque overflow error"; }
  };

  struct EmptyQueue : public std::exception {
    const char *what() const noexcept override { return "empty deque error"; }
  };

  T &front() {
    assert_not_empty();
    return data_[front_];
  }
  const T &front() const {
    assert_not_empty();
    return data_[front_];
  }

  T &back() {
    assert_not_empty();
    return data_[back_ - 1];
  }
  const T &back() const {
    assert_not_empty();
    return data_[back_ - 1];
  }

  void push_back(const T &item) noexcept {
    if (is_full()) front_ = (front_ + 1) % S;
    data_[back_] = item;
    back_ = (back_ + 1) % S;
  }
  T &make_back() noexcept {
    if (is_full()) front_ = (front_ + 1) % S;
    auto &ref = data_[back_];
    back_ = (back_ + 1) % S;
    return ref;
  }

  void clear() noexcept { front_ = back_ = 0; }
  inline std::size_t size() const noexcept { return (back_ + S - front_) % S; }
  inline bool is_empty() const noexcept { return front_ == back_; }
  inline bool is_full() const noexcept { return (back_ + 1) % S == front_; }

  const T &get_padded(int idx) const noexcept {
    std::size_t size_ = size();
    if (idx < 0) idx = int(size_) + idx;
    if (idx < 0) return front();
    else if (idx >= size_) return back();
    else return data_[(front_ + idx) % S];
  }

  const T &get(int idx, const T &default_value) const noexcept {
    std::size_t size_ = size();
    if (idx < 0) idx = int(size_) + idx;
    if (idx < 0 or idx >= size_) return default_value;
    return data_[(front_ + idx) % S];
  }

  T &operator[](int idx) {
    std::size_t size_ = size();
    if (idx < 0) idx = int(size_) + idx;
    if (idx < 0 or idx >= size_) throw QueueOverflow();
    return data_[(front_ + idx) % N];
  }
 private:
  inline void assert_not_empty() const {
    if (is_empty()) throw EmptyQueue();
  }

  std::array<T, N + 1> data_;
  std::size_t front_ = 0, back_ = 0;
};

#endif //QUADRUPED_DEPLOY_INCLUDE_STATE_HPP_
