#include "aliengo.hpp"
#include "io.hpp"
using namespace std;

int main(int argc, char *argv[]) {
  if (argc > 1) {
    AlienGo robot(argv[1]);
  } else {
    print("usage: ./alienGo_policy <model_path>");
    return 0;
  }
  while (true);
}