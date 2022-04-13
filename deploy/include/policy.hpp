#ifndef QUADRUPED_DEPLOY_INCLUDE_POLICY_HPP_
#define QUADRUPED_DEPLOY_INCLUDE_POLICY_HPP_

#include <string>
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h>
#include "state.hpp"

class SlidingWindow {
 public:
  SlidingWindow(int obs_dim, int max_len, int history_len, const torch::DeviceType &device_type)
      : device_(device_type), max_len_(max_len), history_len_(history_len),
        obs_buffer_(torch::zeros({1, obs_dim, max_len}, at::TensorOptions(device_))) {}

  void add_transition(const torch::Tensor &proprio_obs) {
    using namespace torch::indexing;
    if (num_transitions_ >= max_len_) {
      num_transitions_ = history_len_ - 1;
      obs_buffer_.index_put_({"...", Slice(None, num_transitions_)},
                             obs_buffer_.index({"...", Slice(-num_transitions_)}));
    }
    obs_buffer_.index_put_({"...", num_transitions_}, proprio_obs.to(device_));
    num_transitions_ += 1;
  }

  torch::Tensor get_window() {
    using namespace torch::indexing;
    namespace F = torch::nn::functional;
    int padding = history_len_ - num_transitions_;
    torch::Tensor obs_history;
    if (padding > 0) {
      obs_history = F::pad(obs_buffer_.index({"...", Slice(None, num_transitions_)}),
                           F::PadFuncOptions({padding, 0}).mode(torch::kConstant));
    } else {
      obs_history = obs_buffer_.index({"...", Slice(num_transitions_ - history_len_, num_transitions_)});
    }
    return obs_history;
  }

  void clear() { num_transitions_ = 0; }

 private:
  torch::Device device_;
  torch::Tensor obs_buffer_;
  int max_len_, history_len_;
  int num_transitions_ = 0;
};

class Policy {
 public:
  explicit Policy(const std::string &model_path, const torch::DeviceType &device_type)
      : device_(device_type), module_(torch::jit::load(model_path, device_)),
        history_(ProprioInfo::dim, 2000, 123, device_type) {
    biases = biases.to(device_);
    weights = weights.to(device_);
    for (int i = 0; i < 10; ++i) {
      module_.forward({torch::rand({1, 60, 123}).to(device_),
                       torch::rand({1, RealWorldObservation::dim}).to(device_)}).toTensor();
    }
  }

  Policy(Policy &) = delete;
  Policy(const Policy &) = delete;

  torch::Tensor biases = torch::zeros(12);
  torch::Tensor weights = torch::from_blob(new float[12]{0.25, 0.25, 0.15, 0.25, 0.25, 0.15,
                                                         0.25, 0.25, 0.15, 0.25, 0.25, 0.15}, 12);

  torch::Tensor get_action(const ProprioInfo &proprio_info, const RealWorldObservation &real_world_obs) {
    history_.add_transition(torch::from_blob(proprio_info.standard()->data(), ProprioInfo::dim));
    auto real_world_obs_tensor = torch::from_blob(real_world_obs.standard()->data(), {1, RealWorldObservation::dim});
    auto action = module_.forward({history_.get_window(), real_world_obs_tensor.to(device_)}).toTensor();
    action = (action.tanh() - biases) * weights;
    return action.to(torch::kCPU);
  }
 private:
  torch::Device device_;
  torch::jit::script::Module module_;
  SlidingWindow history_;
};
#endif //QUADRUPED_DEPLOY_INCLUDE_POLICY_HPP_
