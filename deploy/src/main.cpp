#include "aliengo.hpp"
#include "io.hpp"
//using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 1) {
    print("usage: ./alienGo_policy <model_path>");
    return 0;
  }
  ros::init(argc, argv, "alienGo_policy");
  std::string model_path("/home/jewel/Workspaces/teacher-student/log/student/script_model.pt");
  AlienGo robot(model_path);
  robot.standup();
  robot.startPolicyThread();
  ros::spin();
//  while (ros::ok());
}