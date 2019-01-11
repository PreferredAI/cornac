
#include "./cpp_hpf.h"



//Update Kappa_R, can be user to update Tau_r as well
void update_kappa_r(dVec &K_r,  Mat &G_s, Mat &G_r,double a_,double b_)
{  
    for(int i = 0; i<K_r.size(); ++i)
    {
        double Sk = 0.0;
        for(int k = 0;k<G_s[0].size();++k)
        {
          //if(G_r.coeff(i,k)!=0.0)
          Sk += G_s[i][k]/G_r[i][k];
        }
        K_r[i] = a_/b_ + Sk;
    }  
}

//update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r(Mat &G_r, Mat &L_s, Mat &L_r, dVec &K_r, double k_s, double att)
{
	for(int k = 0;k<G_r[0].size();++k)
	{
		double Sk = 0.0;
		for (int j_ = 0; j_<L_r.size(); ++j_)
		{
    		if(L_r[j_][k]>0.0)
    			Sk += L_s[j_][k]/L_r[j_][k];
		}
		for(int i = 0;i<G_r.size();++i)
		{
			G_r[i][k] = k_s/K_r[i] + Sk;
		}   
	}
}


//Update the shape parameters of Gamma 
void update_gamma_s(Mat &G_s, SpMat const& X, SpMat const&Lt, SpMat const&Lb){
  
	double eps = pow(2,-52);
	for(int nz = 0;nz<X.outerSize();++nz)
	{
		for (SpMatiter x_(X,nz); x_; ++x_)
		{
			double dk = eps;
      
			for(int k = 0;k<Lt.cols();++k)
			{
				dk += Lt.coeff(x_.row(),k)*Lb.coeff(x_.col(),k);           
			} 
			for(int k = 0;k<Lt.cols();++k)
			{
				G_s[x_.row()][k] += Lt.coeff(x_.row(),k)*Lb.coeff(x_.col(),k)*X.coeff(x_.row(),x_.col())/dk;
				//L_s.coeffRef(i_.col(),k) += Lt.coeff(i_.row(),k)*Lb.coeff(i_.col(),k)*X.coeff(i_.row(),i_.col())/dk;
			} 
		}
	}
}


//Update the shape parameters of Lambda 
void update_lambda_s(Mat &L_s, SpMat const& X, SpMat const&Lt, SpMat const&Lb)
{
	double eps = pow(2,-52);
	
	for(int nz = 0;nz<X.outerSize();++nz)
	{
		for (SpMatiter x_(X,nz); x_; ++x_)
		{
			double dk = eps;
			for(int k = 0;k<Lt.cols();++k)
			{
				dk += Lt.coeff(x_.row(),k)*Lb.coeff(x_.col(),k);           
			} 
			for(int k = 0;k<Lt.cols();++k)
			{
				L_s[x_.col()][k] += Lt.coeff(x_.row(),k)*Lb.coeff(x_.col(),k)*X.coeff(x_.row(),x_.col())/dk;
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

SpMat E_SpMat_logGamma(Mat const& G_s, Mat const& G_r)
{  
    SpMat logG_r(G_r.size(),G_r[0].size());
    SpMat digaG_s(G_s.size(),G_s[0].size());
    
    for(int u = 0; u<G_s.size();++u)
    {
        for(int k = 0; k<G_s[0].size();++k)
        {
            if(G_r[u][k]>0.0)
                logG_r.coeffRef(u,k) = G_r[u][k];
            if(G_s[u][k]>0.0)
                digaG_s.coeffRef(u,k) = G_s[u][k];
        }
    }
    logG_r.prune(0.0);
    digaG_s.prune(0.0);
    logG_r.coeffs() = logG_r.coeffs().log();
    digaG_s.coeffs() = digaG_s.coeffs().digamma();
  
    return (digaG_s - logG_r);
}


// Build a csc sparse matrix from triplet data
SpMat triplet_to_csc_sparse(Mat const& M, int n_row, int n_col)
{
    SpMat spM(n_row, n_col);
    for(int i = 0;i<M.size();++i)
    {
        spM.coeffRef(M[i][0],M[i][1]) = M[i][2];
    }
      
    return spM;
}


void pf_cpp(Mat const&tX, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, dVec &K_r, dVec &T_r, int maxiter){
  
	//data shape
    int n  = G_s.size();
    int d  = L_s.size();
	
	//create sparse matrices from triplet
    SpMat X = triplet_to_csc_sparse(tX,n,d);
  
	//Hyper parameter setting
	double a_  = 0.3;
	double att = 1.;
	double c_  = 0.3;
	double b_ = 1.0;
	double d_ = 1.0;
	double k_s = a_;
	double t_s = c_;
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
		set_coeffs_to(G_s,a_);
		update_gamma_s(G_s,X,Lt,Lb);
    
		//update Gamma_R
		update_gamma_r(G_r,L_s,L_r,K_r,k_s,att);
    
		//Update Kappa_R
    	//update_kappa_r(K_r,G_s,G_r,a_,b_);
    
    
		///Update item related parameters///
    
		//updare Lambda_S
		set_coeffs_to(L_s,c_);
		update_lambda_s(L_s,X,Lt,Lb);
    
		//update Lambda_R
		update_gamma_r(L_r,G_s,G_r,T_r,t_s,att);
    
		//Update Tau_R
		//update_kappa_r(T_r,L_s,L_r,c_,d_);
    
		// End of learning 
    
	}

}




void hpf_cpp(Mat const&tX, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, dVec &K_r, dVec &T_r, int maxiter){
  
	//data shape
    int n  = G_s.size();
    int d  = L_s.size();
	
	//create sparse matrices from triplet
    SpMat X = triplet_to_csc_sparse(tX,n,d);
  
	//Hyper parameter setting
	double a_  = 0.3;
	double att = 1.;
	double b_ = 0.3;
	double c_ = 1.;
	double k_s = a_ + g*a_;
	double t_s = b_ + g*b_;
	//double eps = pow(2.0,-52);
	
	//Util variables declaration
	SpMat Lt(n,g);
	SpMat Lb(d,g);
  
  	//Update Kappa_R
    update_kappa_r(K_r,G_s,G_r,a_,c_);
    
	//Update Tau_R
	update_kappa_r(T_r,L_s,L_r,b_,c_);
    
    
	//Learning 
	for(int iter = 0;iter<maxiter;++iter){
    
		Lt = E_SpMat_logGamma(G_s,G_r);
		Lt.coeffs() = Lt.coeffs().exp();
    
		Lb = E_SpMat_logGamma(L_s,L_r);
		Lb.coeffs() = Lb.coeffs().exp();
    
		//Update user related parameters
		
		//updare Gamma_S
		set_coeffs_to(G_s,a_);
		update_gamma_s(G_s,X,Lt,Lb);
    
		//update Gamma_R
		update_gamma_r(G_r,L_s,L_r,K_r,k_s,att);
    
		//Update Kappa_R
    	update_kappa_r(K_r,G_s,G_r,a_,c_);
    
    
		///Update item related parameters///
    
		//updare Lambda_S
		set_coeffs_to(L_s,b_);
		update_lambda_s(L_s,X,Lt,Lb);
    
		//update Lambda_R
		update_gamma_r(L_r,G_s,G_r,T_r,t_s,att);
    
		//Update Tau_R
		update_kappa_r(T_r,L_s,L_r,b_,c_);
    
		// End of learning 
    
	}

}