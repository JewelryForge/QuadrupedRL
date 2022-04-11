#ifndef QUADRUPED_DEPLOY_INCLUDE_ALIENGO_HPP_
#define QUADRUPED_DEPLOY_INCLUDE_ALIENGO_HPP_

#include <iostream>
#include <array>
#include <utility>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <memory>

#include "tg.hpp"
#include "math_utils.hpp"
#include "state.hpp"
#include "policy.hpp"
#include "torch/script.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

template<typename T>
class ThreadSafe {
 public:
  ThreadSafe() = default;
  explicit ThreadSafe(const T &item) : item_(item) {}
  void copyTo(T &out) {
    lock_.lock();
    out = item_;
    lock_.unlock();
  }

  T getValue() {
    T value;
    copyTo(value);
    return value;
  }

  void setValue(const T &value) {
    lock_.lock();
    item_ = value;
    lock_.unlock();
  }

 private:
  std::mutex lock_;
  T item_{};
};

class UnitreeUDPWrapper {
 public:
  explicit UnitreeUDPWrapper(int inner_freq = 500, int outer_freq = 50)
      : num_inner_loops_(inner_freq / outer_freq), inner_freq_(inner_freq), outer_freq_(outer_freq),
        udp_pub_(UNITREE_LEGGED_SDK::LOWLEVEL),
        safe_(UNITREE_LEGGED_SDK::LeggedType::Aliengo) {
    udp_pub_.udpState.RecvCount = 0;
    udp_pub_.udpState.SendCount = 0;

    udp_pub_.InitCmdData(low_cmd_msg_);
    low_cmd_msg_.levelFlag = UNITREE_LEGGED_SDK::LOWLEVEL;
    clearCommandMsg();
    udp_pub_.SetSend(low_cmd_msg_);
    udp_pub_.Send();
  }

  virtual void startThreads() {
    loop_control_thread_ = std::thread(([this] { controlLoop(); }));
  }

  void standup() {
    //TODO: interpolate
  }

  void emergentStop() {
    status_ = false;
    clearCommandMsg();
    udp_pub_.Send();
    loop_control_thread_.join();
  }

  void controlLoop() {
    while (status_) {
      cout << "controlLoop" << endl;
      auto start = chrono::system_clock::now();
      controlLoopEvent();
      auto end = chrono::system_clock::now();
      auto sleep_time = chrono::microseconds(1000000 / inner_freq_) - (start - end);
      std::this_thread::sleep_for(sleep_time);
    }
  }

  void controlLoopEvent() {
    udp_pub_.Recv();
    udp_pub_.GetRecv(low_state_msg_);

    mu::Vec12 step_action{step_action_.getValue()};
    if (inner_loop_cnt_ == num_inner_loops_) {
      proc_action_ = step_action;
    } else {
      auto error = step_action - proc_action_;
      proc_action_ += error / (num_inner_loops_ - inner_loop_cnt_);
      ++inner_loop_cnt_;
    }

    for (int i = 0; i < 12; ++i) {
      auto &cmd = low_cmd_msg_.motorCmd[i];
      cmd.Kp = 150;
      cmd.Kd = 4;
      cmd.q = proc_action_[i];
    }
    safe_.PowerProtect(low_cmd_msg_, low_state_msg_, 5);
    udp_pub_.SetSend(low_cmd_msg_);
    udp_pub_.Send();
  }

  void applyCommand(const mu::Vec12 &cmd) {
    step_action_.setValue(cmd);
    low_cmd_history_.push_back(cmd);
    inner_loop_cnt_ = 0;
  }

 protected:
  void clearCommandMsg() {
    for (int i = 0; i < 12; i++) {
      auto &cmd = low_cmd_msg_.motorCmd[i];
      cmd.mode = 0x0A;   // motor switch to servo (PMSM) mode
      cmd.q = UNITREE_LEGGED_SDK::PosStopF;
      cmd.dq = UNITREE_LEGGED_SDK::VelStopF;
      cmd.Kp = cmd.Kd = cmd.tau = 0;
    }
  }

  template<std::size_t N, typename ARRAY1, typename ARRAY2>
  static void copy(const ARRAY1 &in, ARRAY2 &out) {
    for (int i = 0; i < N; ++i) out[i] = in[i];
  }

  int num_inner_loops_, inner_freq_, outer_freq_;
  std::thread loop_control_thread_;

  UNITREE_LEGGED_SDK::Safety safe_;
  UNITREE_LEGGED_SDK::UDP udp_pub_;
  UNITREE_LEGGED_SDK::LowCmd low_cmd_msg_{};
  UNITREE_LEGGED_SDK::LowState low_state_msg_{};

  mu::Vec12 proc_action_{};
  ThreadSafe<mu::Vec12> step_action_{};
  std::atomic<bool> status_{true};
  int inner_loop_cnt_ = 0;
  StaticQueue<mu::Vec12, 10> low_cmd_history_;
};

class AlienGo : public UnitreeUDPWrapper {
 public:
  explicit AlienGo(const std::string &model_path, int inner_freq = 500, int outer_freq = 50)
      : UnitreeUDPWrapper(inner_freq, outer_freq),
        policy_(model_path, torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        tg_(VerticalTG(0.12), 2.0, {0, -mu::PI, -mu::PI, 0}) {
    applyCommand(STANCE_POSTURE);
  }

  const float STANCE_HEIGHT = 0.4;
  const mu::Vec12 STANCE_POSTURE{0., 0.6435, -1.287, 0., 0.6435, -1.287,
                                 0., 0.6435, -1.287, 0., 0.6435, -1.287};
  const mu::Vec12 STANCE_FOOT_POSITIONS{0., 0., -STANCE_HEIGHT, 0., 0., -STANCE_HEIGHT,
                                        0., 0., -STANCE_HEIGHT, 0., 0., -STANCE_HEIGHT};
  const mu::Vec3 LINK_LENGTHS{0.083, 0.25, 0.25};

  void startThreads() override {
    reinterpret_cast<UnitreeUDPWrapper *>(this)->startThreads();
    action_thread_ = std::thread([this]() { actionLoop(); });
  }

  void setCommand(const mu::Vec3 &base_linear_cmd) {
    base_lin_cmd_.setValue(base_linear_cmd);
  }

  void actionLoop() {
    while (status_) {
      cout << "actionLoop" << endl;
      auto start = chrono::system_clock::now();
      actionLoopEvent();
      auto end = chrono::system_clock::now();
      auto sleep_time = chrono::microseconds(1000000 / inner_freq_) - (start - end);
      std::this_thread::sleep_for(sleep_time);
    }
  }

  void actionLoopEvent() {
    auto proprio_info = collectProprioInfo();
    auto realworld_obs = makeRealWorldObs();
    auto action = policy_.get_action(*proprio_info, *realworld_obs);
    tg_.update(1. / outer_freq_);
    mu::Vec12 priori, joint_cmd;
    tg_.getPrioriTrajectory(priori);
    inverseKinematicsPatch(Eigen::Map<mu::Vec12>(action.data_ptr<float>()) + priori, joint_cmd);
    step_action_.setValue(joint_cmd);
  }

  std::shared_ptr<ProprioInfo> collectProprioInfo() {
    auto obs = std::make_shared<ProprioInfo>();
    obs_history_.push_back(obs);
    obs->command = base_lin_cmd_.getValue();
    getGravityVector(low_state_msg_.imu.quaternion, obs->gravity_vector);
    getLinearVelocity(obs->base_linear);
    copy<3>(low_state_msg_.imu.gyroscope, obs->base_angular);
    for (int i = 0; i < 12; ++i) {
      obs->joint_pos[i] = low_state_msg_.motorState[i].q;
      obs->joint_vel[i] = low_state_msg_.motorState[i].dq;
    }
    obs->joint_pos_target = low_cmd_history_.back();
    obs->ftg_frequencies = tg_.freq;
    tg_.getSoftPhases(obs->ftg_phases);
    return obs;
  }

  std::shared_ptr<RealWorldObservation> makeRealWorldObs() {
    assert(not obs_history_.is_empty());
    std::shared_ptr<RealWorldObservation> obs(new RealWorldObservation);
    auto proprio_obs = obs_history_[-1];
    reinterpret_cast<ProprioInfo &>(*obs) = *proprio_obs;
    const auto &last_joint_cmd = low_cmd_history_.get(-1, STANCE_POSTURE);
    const auto &last_joint_pos = proprio_obs->joint_pos;
    obs->joint_prev_pos_err = last_joint_cmd - last_joint_pos;
    int p0_01 = -int(0.01 * outer_freq_), p0_02 = -int(0.02 * outer_freq_);
    const auto &obs_p0_01 = obs_history_.get_padded(p0_01), &obs_p0_02 = obs_history_.get_padded(p0_02);
    obs->joint_pos_err_his.segment<12>(0) = obs_p0_01->joint_pos_target - obs_p0_01->joint_pos;
    obs->joint_pos_err_his.segment<12>(12) = obs_p0_02->joint_pos_target - obs_p0_02->joint_pos;
    obs->joint_vel_his.segment<12>(0) = obs_p0_01->joint_vel;
    obs->joint_vel_his.segment<12>(12) = obs_p0_02->joint_vel;
    obs->joint_prev_pos_target = low_cmd_history_.get_padded(-2);
    obs->base_frequency = {tg_.base_freq};
    return obs;
  }

  void getLinearVelocity(mu::Vec3 &out) {
    // get linear velocity from realsense
    // Add acceleration integral
  }

  template<typename ARRAY>
  static void getGravityVector(const ARRAY &orientation, mu::Vec3 &out) {
    float w = orientation[0], x = orientation[1], y = orientation[2], z = orientation[3];
    out[0] = 2 * x * z + 2 * y * w;
    out[1] = 2 * y * z - 2 * x * w;
    out[2] = 1 - 2 * x * x - 2 * y * y;
  }

  template<typename ARRAY_3>
  void inverseKinematics(uint leg, mu::Vec3 pos, ARRAY_3 &out) {
    float l_shoulder = LINK_LENGTHS[0], l_thigh = LINK_LENGTHS[1], l_shank = LINK_LENGTHS[2];
    if (leg % 2 == 0) l_shoulder *= -1;
    const float &dx = pos[0], &dy = pos[1], &dz = pos[2];
    pos[1] += l_shoulder;
    pos += STANCE_FOOT_POSITIONS.segment<3>(leg * 3);
    while (true) {
      float l_stretch = std::sqrt(Eigen::square(pos).sum() - mu::pow2(l_shoulder));
      float a_hip_bias = std::atan2(dy, dz);
      float sum = std::asin(l_shoulder / std::hypot(dy, dz));
      if (not std::isnan(sum)) {
        float a_hip1 = mu::ang_norm(sum - a_hip_bias), a_hip2 = mu::ang_norm(mu::PI - sum - a_hip_bias);
        out[0] = std::abs(a_hip1) < std::abs(a_hip2) ? a_hip1 : a_hip2;
        float a_stretch = -std::asin(dx / l_stretch);
        if (not std::isnan(a_stretch)) {
          float a_shank = std::acos((mu::pow2(l_shank) + mu::pow2(l_thigh) - mu::pow2(l_stretch))
                                        / (2 * l_shank * l_thigh)) - mu::PI;
          if (not std::isnan(a_shank)) {
            out[2] = a_shank;
            float a_thigh = a_stretch - std::asin(l_shank * std::sin(a_shank) / l_stretch);
            out[1] = a_thigh;
            break;
          }
        }
      }
      pos *= 0.95;
    }
  }

  void inverseKinematicsPatch(const mu::Vec12 &pos, mu::Vec12 &out) {
    for (int i = 0; i < 4; ++i) {
      float *start = out.data() + i * 3;
      inverseKinematics(i, pos.segment<3>(i * 3), start);
    }
  }
 private:
  std::thread action_thread_;

  ThreadSafe<mu::fVec<3>> base_lin_cmd_{};
  TgStateMachine tg_;
  StaticQueue<std::shared_ptr<ProprioInfo>, 100> obs_history_;
  Policy policy_;
};

#endif //QUADRUPED_DEPLOY_INCLUDE_ALIENGO_HPP_
