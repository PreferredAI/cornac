
#include "./cpp_hpf.h"






//Compute expectations of Sparse matrices contaning logGamma elements
SpMat E_SpMat_logGamma(SpMat const& G_s, SpMat const& G_r)
{
  
    SpMat logG_r(G_r.rows(),G_r.cols());
    logG_r = G_r;
    logG_r.coeffs() = logG_r.coeffs().log();
  
    SpMat digaG_s(G_s.rows(),G_s.cols());
    digaG_s = G_s;
    digaG_s.coeffs() = digaG_s.coeffs().digamma();
  
    return (digaG_s - logG_r);
}


void pf_cpp(Mat const&tX, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, dVec &K_r, dVec &T_r, int maxiter = 100){
  
	//data shape
	int n = X.rows();
	int d = X.cols();
	
	//create sparse matrices from triplet
    SpMat X = triplet_to_csc_sparse(tX,n,d);
  
	//Hyper parameter setting
	double a  = 0.3;
	double a_ = 3.;
	double att = 1.;
	double c  = 0.3;
	double c_ = 2.;
	double b_ = 1.0;
	double d_ = 1.0;
	double k_s = a ;
	double t_s = c;
	//double k_s = a_ + g*a;
	//double t_s = a_ + g*c;
	//double eps = pow(2.0,-52);
	
	//Util variables declaration
	SpMat Lt(n,g);
	SpMat Lb(d,g);
  
  
	//Learning 
	for(int iter = 0;iter<maxiter;++iter){
    
		Lt = E_SpMat_logGamma(G_s,G_r);
		Lt.coeffs() = Lt.coeffs().exp();
    
		Lb = E_SpMat_logGamma(L_s,L_r);
		Lb.coeffs() = Lb.coeffs().exp();
    
		//Update user related parameters
		
		//updare Gamma_S
		set_coeffs_to(G_s,a);
		update_gamma_s(G_s,X,Lt,Lb);
    
		//update Gamma_R
		update_gamma_r(G_r,L_s,L_r,K_r,k_s,att);
    
		//Update Kappa_R
    	//update_kappa_r(K_r,G_s,G_r,a_,b_);
    
    
		///Update item related parameters///
    
		//updare Lambda_S
		set_coeffs_to(L_s,c);
		update_lambda_s(L_s,X,Lt,Lb);
    
		//update Lambda_R
		update_gamma_r(L_r,G_s,G_r,T_r,t_s,att);
    
		//Update Tau_R
		//update_kappa_r(T_r,L_s,L_r,c_,d_);
    
		// End of learning 
		if((iter%10)==0)
			Rcout << "iter: " << iter <<std::endl;
    
	}

}