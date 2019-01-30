
#include <vector>
#include "Sparse"
#include "Eigen"
#include "SpecialFunctions"

typedef Eigen::MappedSparseMatrix<double> MSpMat;
typedef MSpMat::InnerIterator InIterMat;
typedef Eigen::SparseVector<double> SpVec;
typedef SpVec::InnerIterator InIterVec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef SpMat::InnerIterator SpMatiter;

typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<double> > Mat;
typedef std::vector<double> dVec;
typedef std::vector<int> iVec;





void pf_cpp(Mat const&X, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, dVec &K_r, dVec &T_r, int maxiter);
void hpf_cpp(Mat const&X, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, dVec &K_r, dVec &T_r, int maxiter);