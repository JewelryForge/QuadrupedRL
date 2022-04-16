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

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include "alienGo_deploy/FloatArray.h"
#include "alienGo_deploy/MultiFloatArray.h"

constexpr float ALIENGO_STANCE_HEIGHT = 0.4;
constexpr std::array<float, 12> ALIENGO_STANCE_POSTURE_ARRAY{0., 0.6435, -1.287, 0., 0.6435, -1.287,
                                                             0., 0.6435, -1.287, 0., 0.6435, -1.287};
constexpr std::array<float, 12> ALIENGO_LYING_POSTURE_ARRAY{
    0., 1.343, -2.696, 0., 1.343, -2.696, 0., 1.343, -2.696, 0., 1.343, -2.696,
};
constexpr std::array<float, 12> ALIENGO_STANCE_FOOT_POSITIONS_ARRAY
    {0., 0., -ALIENGO_STANCE_HEIGHT, 0., 0., -ALIENGO_STANCE_HEIGHT,
     0., 0., -ALIENGO_STANCE_HEIGHT, 0., 0., -ALIENGO_STANCE_HEIGHT};
constexpr std::array<float, 3> ALIENGO_LINK_LENGTHS_ARRAY{0.083, 0.25, 0.25};

inline std::size_t time_stamp() {
  return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count();
}

class AlienGoComm {
 public:
  explicit AlienGoComm(int inner_freq = 500, int outer_freq = 50)
      : inner_freq_(inner_freq), outer_freq_(outer_freq), num_inner_loops_(inner_freq / outer_freq),
        udp_pub_(UNITREE_LEGGED_SDK::LOWLEVEL),
        safe_(UNITREE_LEGGED_SDK::LeggedType::Aliengo) {
    udp_pub_.InitCmdData(low_cmd_msg_);
    low_cmd_msg_.levelFlag = UNITREE_LEGGED_SDK::LOWLEVEL;
    clearCommandMsgNoLock();
    udp_pub_.SetSend(low_cmd_msg_);
    udp_pub_.Send();
  }

  ~AlienGoComm() {
    status_ = false;
    control_loop_thread_.join();
  }

  void startControlThread() {
    if (not control_loop_thread_.joinable()) {
      status_ = true;
      // For thread safety, not to read inner_freq_ directly
      control_loop_thread_ = std::thread(&AlienGoComm::controlLoop, this, inner_freq_);
    }
  }

  void controlLoop(int freq) {
    auto rate = ros::Rate(freq);
    while (status_) {
      controlLoopEvent();
      rate.sleep();
    }
  }

  virtual void controlLoopEvent() {
    udp_pub_.Recv();
    low_msg_mutex_.lock();
    udp_pub_.GetRecv(low_state_msg_);

    if (active_) {
      low_state_mutex_.lock();
      if (inner_loop_cnt_ == num_inner_loops_) {
        proc_action_ = step_action_;
//        print("WAIT");
      } else {
        auto error = step_action_ - proc_action_;
        // smoother interpolation
        proc_action_ += error / (num_inner_loops_ - inner_loop_cnt_);
        ++inner_loop_cnt_;
      }
//      auto current_time_stamp = time_stamp();
//      print(current_time_stamp - last_time_stamp_, inner_loop_cnt_);
//      last_time_stamp_ = current_time_stamp;
      for (int i = 0; i < 12; ++i) {
        low_cmd_msg_.motorCmd[i].Kp = 150;
        low_cmd_msg_.motorCmd[i].Kd = 4;
        low_cmd_msg_.motorCmd[i].dq = 0;
        low_cmd_msg_.motorCmd[i].q = proc_action_[i];
      }

      low_history_mutex_.lock();
      low_cmd_history_.push_back(proc_action_);
      low_history_mutex_.unlock();
      low_state_mutex_.unlock();
    } else {
      clearCommandMsgNoLock();
      low_state_mutex_.lock();
      for (int i = 0; i < 12; ++i) {
        proc_action_[i] = low_state_msg_.motorState[i].q;
      }
      low_state_mutex_.unlock();
    }
    safe_.PositionLimit(low_cmd_msg_);
    safe_.PowerProtect(low_cmd_msg_, low_state_msg_, 7);
    udp_pub_.SetSend(low_cmd_msg_);
    low_msg_mutex_.unlock();
    udp_pub_.Send();
  }

  void applyCommand(const Array12 &cmd) {
    low_state_mutex_.lock();
    step_action_ = cmd;
    inner_loop_cnt_ = 0;
    low_state_mutex_.unlock();

    step_cmd_history_.push_back(cmd);
  }

  void emergentStop() {
    status_ = false;
    clearCommandMsgNoLock();
    udp_pub_.Send();
    control_loop_thread_.join();
  }
 protected:
  void clearCommandMsgNoLock() {
    // lock outside the function if needed
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
  std::thread control_loop_thread_;
  // AlienGo communication relevant
  UNITREE_LEGGED_SDK::Safety safe_;
  UNITREE_LEGGED_SDK::UDP udp_pub_;
  UNITREE_LEGGED_SDK::LowCmd low_cmd_msg_{};
  UNITREE_LEGGED_SDK::LowState low_state_msg_{};
  std::mutex low_msg_mutex_; // for low_cmd_msg_ & low_state_msg_

  Array12 proc_action_{};
  Array12 step_action_{};
  int inner_loop_cnt_ = 0;
  std::mutex low_state_mutex_; // for proc_action & step_action & inner_loop_cnt_

  std::atomic<bool> status_{false}, active_{false};

  StaticQueue<Array12, 100> low_cmd_history_;
  std::mutex low_history_mutex_; // for low_cmd_history_
  StaticQueue<Array12, 10> step_cmd_history_; // Not used inside inner loop

//  std::size_t last_time_stamp_ = 0;
};

class AlienGo : public AlienGoComm {
 public:
  explicit AlienGo(const std::string &model_path, int inner_freq = 500, int outer_freq = 50)
      : AlienGoComm(inner_freq, outer_freq),
        policy_(model_path, torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        tg_(std::make_shared<VerticalTG>(0.12), 2.0, {0, -PI, -PI, 0}) {
    robot_vel_sub_ = nh_.subscribe<nav_msgs::Odometry>("/camera/odom/sample", 5, &AlienGo::velocityUpdate, this);
    cmd_vel_sub = nh_.subscribe<alienGo_deploy::FloatArray>("/cmd_vel", 1, &AlienGo::cmdVelUpdate, this);
    data_tester_ = nh_.advertise<alienGo_deploy::MultiFloatArray>("/test_data", 1);
    applyCommand(STANCE_POSTURE);
  }

  ~AlienGo() {
    status_ = false;
    action_loop_thread_.join();
  }

  void standup() {
    // start inner loop and control robot to stand up
    active_ = false;
    startControlThread();
    std::this_thread::sleep_for(chrono::milliseconds(100));
    while (true) {
      low_msg_mutex_.lock();
      bool is_empty = low_state_msg_.tick == 0;
      low_msg_mutex_.unlock();
      if (not is_empty) break;
      std::cout << "NOT CONNECTED" << std::endl;
      std::this_thread::sleep_for(chrono::milliseconds(500));
    }

    Array12 init_cfg;
    low_msg_mutex_.lock();
    for (int i = 0; i < 12; ++i) {
      init_cfg[i] = low_state_msg_.motorState[i].q;
    }
    low_msg_mutex_.unlock();

    int num_steps = int(1 * outer_freq_);
    auto rate = ros::Rate(outer_freq_);
    active_ = true;
    for (int i = 1; i <= num_steps; ++i) {
      applyCommand(float(num_steps - i) / num_steps * init_cfg
                       + float(i) / num_steps * LYING_POSTURE);
      rate.sleep();
    }
    num_steps = int(1.5 * outer_freq_);
    for (int i = 1; i <= num_steps; ++i) {
      applyCommand(float(num_steps - i) / num_steps * LYING_POSTURE
                       + float(i) / num_steps * STANCE_POSTURE);
      rate.sleep();
    }
  }

  void startPolicyThread() {
    if (not action_loop_thread_.joinable() and status_) {
      action_loop_thread_ = std::thread(&AlienGo::actionLoop, this, outer_freq_);
    }
  }

  void setCommand(const Array3 &cmd_vel) {
    high_state_mutex_.lock();
    cmd_vel_ = cmd_vel;
    high_state_mutex_.unlock();
  }

//  template <typename Derived>
  void inverseKinematics(uint leg, Array3 pos, Eigen::Ref<Array3> out) {
    // leg: 0 = forward right; 1 = forward left;
    //      2 = rear right;    3 = rear left.
    // pos: relative to correspondent foot on standing
    float l_shoulder = LINK_LENGTHS[0], l_thigh = LINK_LENGTHS[1], l_shank = LINK_LENGTHS[2];
    if (leg % 2 == 0) l_shoulder *= -1;
    const float &dx = pos[0], &dy = pos[1], &dz = pos[2];
    pos[1] += l_shoulder;
    pos += STANCE_FOOT_POSITIONS.segment<3>(leg * 3);
    while (true) {
      float l_stretch = std::sqrt(Eigen::square(pos).sum() - pow2(l_shoulder));
      float a_hip_bias = std::atan2(dy, dz);
      float sum = std::asin(l_shoulder / std::hypot(dy, dz));
      if (not std::isnan(sum)) {
        float a_hip1 = ang_norm(sum - a_hip_bias), a_hip2 = ang_norm(PI - sum - a_hip_bias);
        out[0] = std::abs(a_hip1) < std::abs(a_hip2) ? a_hip1 : a_hip2;
        float a_stretch = -std::asin(dx / l_stretch);
        if (not std::isnan(a_stretch)) {
          float a_shank = std::acos((pow2(l_shank) + pow2(l_thigh) - pow2(l_stretch))
                                        / (2 * l_shank * l_thigh)) - PI;
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

  void inverseKinematicsPatch(const Array12 &pos, Eigen::Ref<Array12> out) {
    for (int i = 0; i < 4; ++i) {
      inverseKinematics(i, pos.segment<3>(i * 3), out.segment<3>(i * 3));
    }
  }

 private:
  void actionLoop(int freq) {
    auto rate = ros::Rate(freq);
    while (status_) {
      actionLoopEvent();
      rate.sleep();
    }
  }

  void controlLoopEvent() override {
    AlienGoComm::controlLoopEvent();
    auto data = collectProprioInfo();
    alienGo_deploy::MultiFloatArray multi_array;

//    alienGo_deploy::FloatArray array;
//    for (int i = 0; i < 12; ++i) {
//      array.data.push_back(low_state_msg_.motorState[i].q);
//    }
//    multi_array.data.push_back(array);

    multi_array.data.push_back(*makeFloatArray(data->command));
    multi_array.data.push_back(*makeFloatArray(data->gravity_vector));
    multi_array.data.push_back(*makeFloatArray(data->base_linear));
    multi_array.data.push_back(*makeFloatArray(data->base_angular));
    multi_array.data.push_back(*makeFloatArray(data->joint_pos));
    multi_array.data.push_back(*makeFloatArray(data->joint_vel));
    multi_array.data.push_back(*makeFloatArray(data->joint_pos_target));
    data_tester_.publish(multi_array);
  }

  template<int N>
  static alienGo_deploy::FloatArray::Ptr makeFloatArray(const fArray<N> &data) {
    alienGo_deploy::FloatArray::Ptr array(new alienGo_deploy::FloatArray);
    for (int i = 0; i < N; ++i) array->data.push_back(data[i]);
    return array;
  }

  void actionLoopEvent() {
    low_state_mutex_.lock();
    // copy to make sure obs_history_ is not locked when calculating action
    auto proprio_info = *obs_history_.back();
    low_state_mutex_.unlock();
    auto realworld_obs = makeRealWorldObs();
    auto action = policy_.get_action(proprio_info, *realworld_obs);
    Array12 action_array = Eigen::Map<Array12>(action.data_ptr<float>());

    tg_.update(1. / outer_freq_);
    Array12 joint_cmd;
    inverseKinematicsPatch(action_array + tg_.getPrioriTrajectory(), joint_cmd);
    applyCommand(joint_cmd);
  }

  void velocityUpdate(const nav_msgs::Odometry::ConstPtr &odom) {
    // in main thread
    cam_state_mutex_.lock();
    cam_lin_vel_ = {float(odom->twist.twist.linear.x),
                    float(odom->twist.twist.linear.y),
                    float(odom->twist.twist.linear.z)};
    cam_ang_vel_ = {float(odom->twist.twist.angular.x),
                    float(odom->twist.twist.angular.y),
                    float(odom->twist.twist.angular.z)};
    cam_state_mutex_.unlock();
  }

  void cmdVelUpdate(const alienGo_deploy::FloatArray::ConstPtr &cmd_vel) {
    high_state_mutex_.lock();
    copy<3>(cmd_vel->data, cmd_vel_);
    high_state_mutex_.unlock();
  }

  std::shared_ptr<ProprioInfo> collectProprioInfo() {
    Eigen::Matrix3f w_R_b;
    low_msg_mutex_.lock();
    const auto &orn = low_state_msg_.imu.quaternion;
    float w = orn[0], x = orn[1], y = orn[2], z = orn[3];
    low_msg_mutex_.unlock();
    float xx = x * x, xy = x * y, xz = x * z, xw = x * w;
    float yy = y * y, yz = y * z, yw = y * w, zz = z * z, zw = z * w;
    w_R_b << 1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw,
        2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw,
        2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy;

    auto obs = std::make_shared<ProprioInfo>();
    high_state_mutex_.lock();
    obs->command = cmd_vel_;
    high_state_mutex_.unlock();
    getLinearVelocity(w_R_b, obs->base_linear);

    low_msg_mutex_.lock();
    getGravityVector(low_state_msg_.imu.quaternion, obs->gravity_vector);
    copy<3>(low_state_msg_.imu.gyroscope, obs->base_angular);
    for (int i = 0; i < 12; ++i) {
      obs->joint_pos[i] = low_state_msg_.motorState[i].q;
      obs->joint_vel[i] = low_state_msg_.motorState[i].dq;
    }
    low_msg_mutex_.unlock();

//    obs->joint_pos_target = step_cmd_history_.back();
    low_state_mutex_.lock();
    obs->joint_pos_target = proc_action_;
    low_state_mutex_.unlock();

    obs->ftg_frequencies = tg_.freq;
    tg_.getSoftPhases(obs->ftg_phases);

    obs_history_mutex_.lock();
    obs_history_.push_back(obs);
    obs_history_mutex_.unlock();
    return obs;
  }

  std::shared_ptr<RealWorldObservation> makeRealWorldObs() {
    std::shared_ptr<RealWorldObservation> obs(new RealWorldObservation);
    int p0_01 = -int(0.01 * inner_freq_) - 1, p0_02 = -int(0.02 * inner_freq_) - 1;

    obs_history_mutex_.lock();
    assert(not obs_history_.is_empty());
    auto proprio_obs = obs_history_[-1];
    reinterpret_cast<ProprioInfo &>(*obs) = *proprio_obs;
    low_state_mutex_.lock();
    obs->joint_prev_pos_err = proc_action_ - proprio_obs->joint_pos;
    low_state_mutex_.unlock();
//    obs_history_mutex_.unlock();

    low_history_mutex_.lock();
    // in simulation, apply command -> step simulation -> get observation
    // in real world, apply command -> low loop period -> get observation
    auto low_cmd_p0_01 = low_cmd_history_.get_padded(p0_01 - 1),
        low_cmd_p0_02 = low_cmd_history_.get_padded(p0_02 - 1);
//    obs_history_mutex_.lock();
    const auto obs_p0_01 = obs_history_.get_padded(p0_01),
        obs_p0_02 = obs_history_.get_padded(p0_02);
    obs->joint_pos_err_his.segment<12>(0) = low_cmd_p0_01 - obs_p0_01->joint_pos;
    obs->joint_pos_err_his.segment<12>(12) = low_cmd_p0_02 - obs_p0_02->joint_pos;
    low_history_mutex_.unlock();
    obs->joint_vel_his.segment<12>(0) = obs_p0_01->joint_vel;
    obs->joint_vel_his.segment<12>(12) = obs_p0_02->joint_vel;
    obs_history_mutex_.unlock();

    obs->joint_prev_pos_target = step_cmd_history_.get_padded(-2);
    obs->base_frequency = {tg_.base_freq};
    return obs;
  }

  void getLinearVelocity(const Eigen::Matrix3f &w_R_b, Eigen::Ref<Array3> out) {
    // get base linear velocity in BASE frame
    // w for world, b for base and c for camera
    // w_V_b = w_V_c + w_Ω_c x w_R_c · c_Q_b, so
    // b_V_b = b_V_c + b_Ω_c x w_R_c · c_Q_b, then
    // b_V_b = b_R_c · c_V_c + b_R_c · c_Ω_c x w_R_c · c_Q_b, in this case
    // b_R_c = [[1, 0, 0]
    //          [0, 1, 0]
    //          [0, 0, 1]], so w_R_c = w_R_b, let c_Ω_c x = c_S_c, then simplified
    // b_V_b = c_V_c + c_S_c · w_R_b · c_Q_b
    // c_V_c, i.e. cam_lin_vel_; c_Ω_c, i.e. cam_ang_vel_

    const Eigen::Vector3f c_Q_b(-0.332, 0, 0);
    cam_state_mutex_.lock();
    float Wx = cam_ang_vel_.x(), Wy = cam_ang_vel_.y(), Wz = cam_ang_vel_.z();
    Eigen::Matrix3f c_S_c;
    c_S_c << 0, -Wz, Wy,
        Wz, 0, -Wx,
        -Wy, Wx, 0;
    out = cam_lin_vel_ + (c_S_c * w_R_b * c_Q_b).array();
    cam_state_mutex_.unlock();
  }

  template<typename ARRAY>
  void getGravityVector(const ARRAY &orientation, Eigen::Ref<Array3> out) {
    float w = orientation[0], x = orientation[1], y = orientation[2], z = orientation[3];
    out[0] = 2 * x * z + 2 * y * w;
    out[1] = 2 * y * z - 2 * x * w;
    out[2] = 1 - 2 * x * x - 2 * y * y;
  }

 private:
  Array12 STANCE_POSTURE{ALIENGO_STANCE_POSTURE_ARRAY.data()};
  Array12 LYING_POSTURE{ALIENGO_LYING_POSTURE_ARRAY.data()};
  Array12 STANCE_FOOT_POSITIONS{ALIENGO_STANCE_FOOT_POSITIONS_ARRAY.data()};
  Array3 LINK_LENGTHS{ALIENGO_LINK_LENGTHS_ARRAY.data()};

  std::thread action_loop_thread_;

  // velocity calculation relevant
  Array3 cam_lin_vel_ = Array3::Zero(), cam_ang_vel_ = Array3::Zero();
  std::mutex cam_state_mutex_; // for cam_lin_vel_ & cam_ang_vel_
  // high velocity command relevant
  Array3 cmd_vel_ = Array3::Zero();
  std::mutex high_state_mutex_; // for cmd_vel_
  // previous observations relavant
  StaticQueue<std::shared_ptr<ProprioInfo>, 100> obs_history_;
  std::mutex obs_history_mutex_; // for obs_history_
  // ros relevant
  ros::NodeHandle nh_;
  ros::Subscriber robot_vel_sub_, cmd_vel_sub;
  ros::Publisher data_tester_;
  // action relevant
  TgStateMachine tg_;
  Policy policy_;
};

#endif //QUADRUPED_DEPLOY_INCLUDE_ALIENGO_HPP_
