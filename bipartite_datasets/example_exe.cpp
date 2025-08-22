#include <iostream>
#include <Eigen/Sparse>
 
int main() {
  Eigen::SparseMatrix<double, Eigen::ColMajor> sm(10,10);
  sm.coeffRef(0, 0) = 3;
  sm.coeffRef(1, 0) = 2.5;
  sm.coeffRef(0, 1) = -1;
  sm.coeffRef(1, 1) = sm.coeffRef(1, 0) + sm.coeffRef(0, 1);
  std::cout << sm << std::endl;
}
