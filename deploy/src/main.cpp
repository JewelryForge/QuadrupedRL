#include "aliengo.hpp"
#include "io.hpp"
//using namespace std;

int main(int argc, char *argv[]) {
//  if (argc < 1) {
//    print("usage: ./alienGo_policy <model_path>");
//    return 0;
//  }
  std::string model_path("/home/jewel/Workspaces/teacher-student/log/student/traced_model.pt");
  AlienGo robot(model_path);
  robot.standup();
  robot.startPolicyThread();
  while (true);
}