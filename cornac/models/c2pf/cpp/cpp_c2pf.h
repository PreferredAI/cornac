
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





void c2pf_cpp(Mat const&X, Mat const&C, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &L2_s, Mat &L2_r, Mat &L3_s, Mat &L3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt);
void c2pf_cpp2(Mat const&tX, Mat const&tC, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &L2_s, Mat &L2_r, Mat &tL3_s, Mat &tL3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter,double at,double bt);
void tc2pf_cpp(Mat const&tX, Mat const&tC, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &tL3_s, Mat &tL3_r, dVec &T2_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt);
void rc2pf_cpp(Mat const&tX, Mat const&tC, int const&g, Mat &G_s, Mat &G_r,Mat &L2_s, Mat &L2_r, Mat tL3_s, Mat tL3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt);