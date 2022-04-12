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
#include "io.hpp"
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


constexpr float ALIENGO_STANCE_HEIGHT = 0.4;
constexpr std::array<float, 12> ALIENGO_STANCE_POSTURE_ARRAY{0., 0.6435, -1.287, 0., 0.6435, -1.287,
                                                             0., 0.6435, -1.287, 0., 0.6435, -1.287};
constexpr std::array<float, 12> ALIENGO_STANCE_FOOT_POSITIONS_ARRAY
    {0., 0., -ALIENGO_STANCE_HEIGHT, 0., 0., -ALIENGO_STANCE_HEIGHT,
     0., 0., -ALIENGO_STANCE_HEIGHT, 0., 0., -ALIENGO_STANCE_HEIGHT};
constexpr std::array<float, 3> ALIENGO_LINK_LENGTHS_ARRAY{0.083, 0.25, 0.25};

class UnitreeUDPWrapper {
 public:
  explicit UnitreeUDPWrapper(int inner_freq = 500, int outer_freq = 50)
      : num_inner_loops_(inner_freq / outer_freq), inner_freq_(inner_freq), outer_freq_(outer_freq),
        udp_pub_(UNITREE_LEGGED_SDK::LOWLEVEL),
        safe_(UNITREE_LEGGED_SDK::LeggedType::Aliengo) {
    udp_pub_.InitCmdData(low_cmd_msg_);
    low_cmd_msg_.levelFlag = UNITREE_LEGGED_SDK::LOWLEVEL;
    clearCommandMsg();
    udp_pub_.SetSend(low_cmd_msg_);
    udp_pub_.Send();
  }

  ~UnitreeUDPWrapper() {
    status_ = false;
    loop_control_thread_.join();
  }

  void startControlThread() {
    if (not loop_control_thread_.joinable()) {
      status_ = true;
      loop_control_thread_ = std::thread(([this] { controlLoop(); }));
    }
  }

  void emergentStop() {
    status_ = false;
    clearCommandMsg();
    udp_pub_.Send();
    loop_control_thread_.join();
  }

  void controlLoop() {
    auto period = chrono::duration<double>(1. / inner_freq_);
    while (true) {
//      cout << "controlLoop" << endl;
      auto start = chrono::system_clock::now();
      if (not status_) break;
      controlLoopEvent();
      auto end = chrono::system_clock::now();
      auto sleep_time = period - (end - start);
      std::this_thread::sleep_for(sleep_time);
//      cout << chrono::duration_cast<chrono::microseconds>(sleep_time).count() << endl;
    }
  }

  void controlLoopEvent() {
    udp_pub_.Recv();
    low_state_mutex_.lock();
    udp_pub_.GetRecv(low_state_msg_);

    if (active_) {
      if (inner_loop_cnt_ == num_inner_loops_) {
        proc_action_ = step_action_;
      } else {
         auto error = step_action_ - proc_action_;
         proc_action_ += error / (num_inner_loops_ - inner_loop_cnt_);
         ++inner_loop_cnt_;
      }
      for (int i = 0; i < 12; ++i) {
        low_cmd_msg_.motorCmd[i].Kp = 150;
        low_cmd_msg_.motorCmd[i].Kd = 4;
        low_cmd_msg_.motorCmd[i].dq = 0;
        low_cmd_msg_.motorCmd[i].q = proc_action_[i];
      }
    } else {
      clearCommandMsg();
      for (int i = 0; i < 12; ++i) {
        proc_action_[i] = low_state_msg_.motorState[i].q;
      }
    }
    safe_.PositionLimit(low_cmd_msg_);
    safe_.PowerProtect(low_cmd_msg_, low_state_msg_, 2);
    udp_pub_.SetSend(low_cmd_msg_);
    udp_pub_.Send();
    low_state_mutex_.unlock();
  }

  void applyCommand(const mu::Vec12 &cmd) {
    low_state_mutex_.lock();
    step_action_ = cmd;
    low_cmd_history_.push_back(cmd);
    inner_loop_cnt_ = 0;
    low_state_mutex_.unlock();
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
  std::mutex low_state_mutex_;
  UNITREE_LEGGED_SDK::LowState low_state_msg_{};

  mu::Vec12 proc_action_{};
  mu::Vec12 step_action_{};
  std::atomic<bool> status_{false}, active_{false};
  int inner_loop_cnt_ = 0;

  StaticQueue<mu::Vec12, 10> low_cmd_history_;
};

class AlienGo : public UnitreeUDPWrapper {
 public:
  explicit AlienGo(const std::string &model_path, int inner_freq = 500, int outer_freq = 50)
      : STANCE_POSTURE{ALIENGO_STANCE_POSTURE_ARRAY.data()},
        STANCE_FOOT_POSITIONS{ALIENGO_STANCE_FOOT_POSITIONS_ARRAY.data()},
        LINK_LENGTHS{ALIENGO_LINK_LENGTHS_ARRAY.data()},
        UnitreeUDPWrapper(inner_freq, outer_freq),
        policy_(model_path, torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        tg_(std::make_shared<VerticalTG>(0.12), 2.0, {0, -mu::PI, -mu::PI, 0}) {
    applyCommand(STANCE_POSTURE);
  }

  ~AlienGo() {
    status_ = false;
    action_thread_.join();
  }

  void startPolicyThread() {
    action_thread_ = std::thread([this]() { actionLoop(); });
  }

  void standup() {
    active_ = false;
    startControlThread();
    while (true) {
      low_state_mutex_.lock();
      auto is_empty = low_state_msg_.tick == 0;
      low_state_mutex_.unlock();
      if (not is_empty) break;
      std::cout << "NOT CONNECTED" << std::endl;
      std::this_thread::sleep_for(chrono::milliseconds(500));
    }
    low_state_mutex_.lock();
    mu::Vec12 init_cfg;
    for (int i = 0; i < 12; ++i) {
      init_cfg[i] = low_state_msg_.motorState[i].q;
    }
    low_state_mutex_.unlock();
    int num_steps = 2 * outer_freq_;
    active_ = true;
    auto period = chrono::duration<double>(1. / outer_freq_);
    for (int i = 1; i <= num_steps; ++i) {
      auto start = chrono::system_clock::now();
      applyCommand(float(num_steps - i) / num_steps * init_cfg
                       + float(i) / num_steps * STANCE_POSTURE);
      auto end = chrono::system_clock::now();
      auto sleep_time = period - (end - start);
      std::this_thread::sleep_for(sleep_time);
    }
  }

  void setCommand(const mu::Vec3 &base_linear_cmd) {
    base_lin_cmd_.setValue(base_linear_cmd);
  }

  void actionLoop() {
    auto period = chrono::duration<double>(1. / outer_freq_);
    while (true) {
//      cout << "actionLoop" << endl;
      auto start = chrono::system_clock::now();
      if (not status_) break;
      actionLoopEvent();
      auto end = chrono::system_clock::now();
      auto sleep_time = period - (end - start);
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
    inverseKinematicsPatch(/*Eigen::Map<mu::Vec12>(action.data_ptr<float>()) +*/ priori, joint_cmd);
//    print(priori);
    applyCommand(joint_cmd);
//    applyCommand(STANCE_POSTURE);
  }

  std::shared_ptr<ProprioInfo> collectProprioInfo() {
    auto obs = std::make_shared<ProprioInfo>();
    obs_history_.push_back(obs);
    obs->command = base_lin_cmd_.getValue();
    low_state_mutex_.lock();
    getGravityVector(low_state_msg_.imu.quaternion, obs->gravity_vector);
    getLinearVelocity(obs->base_linear);
    copy<3>(low_state_msg_.imu.gyroscope, obs->base_angular);
    for (int i = 0; i < 12; ++i) {
      obs->joint_pos[i] = low_state_msg_.motorState[i].q;
      obs->joint_vel[i] = low_state_msg_.motorState[i].dq;
    }
    low_state_mutex_.unlock();
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
  mu::Vec12 STANCE_POSTURE;//{ALIENGO_STANCE_POSTURE_ARRAY.data()};
  mu::Vec12 STANCE_FOOT_POSITIONS;//{ALIENGO_STANCE_FOOT_POSITIONS_ARRAY.data()};
  mu::Vec3 LINK_LENGTHS;//{ALIENGO_LINK_LENGTHS_ARRAY.data()};

  std::thread action_thread_;

  ThreadSafe<mu::fVec<3>> base_lin_cmd_{};
  TgStateMachine tg_;
  StaticQueue<std::shared_ptr<ProprioInfo>, 100> obs_history_;
  Policy policy_;
};

#endif //QUADRUPED_DEPLOY_INCLUDE_ALIENGO_HPP_
