
#include "./cpp_hpf.h"


// Update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r(MSpMat &G_r, MSpMat &L_s,MSpMat &L_r,NumericVector &K_r, double k_s, double att = 1.0){
  
	for(int k = 0;k<G_r.cols();++k){
		double Sk = 0.0;
		for (InIterMat j_(L_r, k); j_; ++j_){
			Sk += L_s.coeff(j_.row(),k)/L_r.coeff(j_.row(),k);
		}
    
		for(int i = 0;i<G_r.rows();++i){
			G_r.coeffRef(i,k) = att*k_s/K_r[i] + Sk;
		}
	}
}


//Update the shape parameters of Gamma 
void update_gamma_s(MSpMat &G_s,MSpMat const& X, SpMat const&Lt, SpMat const&Lb){
  
	double eps = pow(2,-52);
	for(int nz = 0;nz<X.outerSize();++nz){
		for (InIterMat i_(X,nz); i_; ++i_){
			double dk = eps;
      
			for(int k = 0;k<Lt.cols();++k){
				dk += Lt.coeff(i_.row(),k)*Lb.coeff(i_.col(),k);           
			} 
			for(int k = 0;k<G_s.cols();++k){
				G_s.coeffRef(i_.row(),k) += Lt.coeff(i_.row(),k)*Lb.coeff(i_.col(),k)*X.coeff(i_.row(),i_.col())/dk;
				//L_s.coeffRef(i_.col(),k) += Lt.coeff(i_.row(),k)*Lb.coeff(i_.col(),k)*X.coeff(i_.row(),i_.col())/dk;
			} 
		}
	}
}

// Set all entries of sparse matrix into a particular value
void set_coeffs_to(Mat &L_s,double c_)
{
	for(int j = 0;j<L_s.size();++j){
    	for(int k = 0;k<L_s[0].size();++k)
    	{
			L_s[j][k] = c_;
		}
	}
}


// Compute expectations of Sparse matrices contaning logGamma elements
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