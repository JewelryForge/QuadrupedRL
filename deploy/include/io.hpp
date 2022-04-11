#ifndef IO_HPP_
#define IO_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <iomanip>

template<typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  if (v.empty()) return os << "[]";
  os << '[' << v.front();
  for (auto iter = v.begin() + 1; iter != v.end(); ++iter) os << ' ' << *iter;
  return os << ']';
}

template<typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const std::array<T, N> &array) {
  if (N == 0) return os << "[]";
  os << "[" << array[0];
  for (auto iter = array.begin() + 1; iter != array.end(); ++iter) os << ' ' << *iter;
  return os << "]";
}

template<typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &p) {
  return os << '<' << p.first << ", " << p.second << '>';
}

namespace {
template<typename T, size_t N>
struct _print_tuple {
  static std::ostream &print(std::ostream &os, const T &t) {
    return _print_tuple<T, N - 1>::print(os, t) << " " << std::get<N - 1>(t);
  }
};
template<typename T>
struct _print_tuple<T, 1> {
  static std::ostream &print(std::ostream &os, const T &t) { return os << std::get<0>(t); }
};

template<typename T>
struct _print_tuple<T, 0> {
  static std::ostream &print(std::ostream &os, const T &t) { return os; }
};
}

template<typename ...Args>
std::ostream &operator<<(std::ostream &os, std::tuple<Args...> &t) {
  return _print_tuple<decltype(t), sizeof...(Args)>::print(os << '(', t) << ')';
}

namespace {
#include <bits/stl_iterator.h>

class Print {
 public:
  Print() = default;
  explicit Print(std::string sep) : sep_(std::move(sep)) {};

  template<typename T1, typename T2>
  std::ostream &operator()(std::ostream &os,
                           const __gnu_cxx::__normal_iterator<T1, T2> &t1,
                           const __gnu_cxx::__normal_iterator<T1, T2> &t2) const {
    if (t1 == t2) return os;
    return t1 + 1 == t2 ? os << *t1 : operator()(os << *t1 << sep_, t1 + 1, t2);
  }
  template<typename T>
  std::ostream &operator()(std::ostream &os, const T &t) const { return os << t; }

  template<typename T, typename ...Args>
  std::ostream &operator()(std::ostream &os, const T &t, const Args &...rest) const {
    os << t << sep_;
    return operator()(os, rest...);
  }

  template<typename ...Args>
  std::ostream &operator()(const Args &...rest) const {
    return operator()(std::cout, rest...) << std::endl;
  }

 private:
  std::string sep_{" "};
};

class Input {
 public:
  Input() = default;

  std::string operator()() const {
    std::string str;
    std::getline(std::cin, str);
    return str;
  }

  void operator()(std::string &str) const {
    std::getline(std::cin, str);
  }
};
}

static const Print print(" ");
static const Input input;
#endif //IO_HPP_