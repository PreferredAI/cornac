
#include "./cpp_c2pf.h"
 
  
 //Update the shape parameters of Gamma, contextual pf 
void update_gamma_s_context(Mat &G_s, SpMat const& X, SpMat const&Lt, SpMat const&Lb, SpMat const&Lb2){
	double eps = pow(2,-52);
	for(int nz = 0;nz<X.outerSize();++nz)
	{
		for (SpMatiter x_(X,nz); x_; ++x_)
		{
			double dk = eps; 
			for(int k = 0;k<Lt.cols();++k)
			{
				dk += Lt.coeff(x_.row(),k)*(Lb.coeff(x_.col(),k)+Lb2.coeff(x_.col(),k));           
			} 
			for(int k = 0;k<G_s[0].size();++k)
			{
				G_s[x_.row()][k] += Lt.coeff(x_.row(),k)*(Lb.coeff(x_.col(),k)+Lb2.coeff(x_.col(),k))*X.coeff(x_.row(),x_.col())/dk;
				//L_s.coeffRef(i_.col(),k) += Lt.coeff(i_.row(),k)*Lb.coeff(i_.col(),k)*X.coeff(i_.row(),i_.col())/dk;
			}   
		}
	}
}


void update_gamma_s_context_r(Mat &G_s, SpMat const& X, SpMat const&Lt, SpMat const&Lb2)
{
    double eps = pow(2,-52);
    for(int nz = 0;nz<X.outerSize();++nz)
    {
        for (SpMatiter x_(X,nz); x_; ++x_)
        {
            double dk = eps; 
            for(int k = 0;k<Lt.cols();++k)
            {
                dk += Lt.coeff(x_.row(),k)*Lb2.coeff(x_.col(),k);           
            } 
            for(int k = 0;k<G_s[0].size();++k)
            {
                G_s[x_.row()][k] += Lt.coeff(x_.row(),k)*Lb2.coeff(x_.col(),k)*X.coeff(x_.row(),x_.col())/dk;
                //L_s.coeffRef(i_.col(),k) += Lt.coeff(i_.row(),k)*Lb.coeff(i_.col(),k)*X.coeff(i_.row(),i_.col())/dk;
    	     }   
		}
	}
}


//Update the shape parameters of Gamma, contextual pf 
void update_lambda_s_context(Mat &L_s, SpMat const& X, SpMat const&Lt, SpMat const&Lb, SpMat const&Lb2)
{ 
	double eps = pow(2,-52);
	for(int nz = 0;nz<X.outerSize();++nz)
	{
		for (SpMatiter x_(X,nz); x_; ++x_)
		{
			double dk = eps; 
			for(int k = 0;k<Lt.cols();++k){
				dk += Lt.coeff(x_.row(),k)*(Lb.coeff(x_.col(),k)+Lb2.coeff(x_.col(),k));           
			} 
			for(int k = 0;k<L_s[0].size();++k)
			{
				L_s[x_.col()][k] += Lt.coeff(x_.row(),k)*Lb.coeff(x_.col(),k)*X.coeff(x_.row(),x_.col())/dk;
			} 
		}
	}
}


//Update the shape parameters of Gamma, contextual pf 
void update_gamma_s_context_2_n(Mat &L2_s, SpMat const& X, SpMat const& C, SpMat const&Lt,SpMat const&Lb,SpMat const&L2b,SpMat const&L3b, SpMat const&Lb2)
{  
	double eps = pow(2,-52);
	SpMat Lb_u(X.cols(),L2_s[0].size());  
	for(int i = 0;i<X.cols();++i)
	{
		for(int k = 0;k<Lt.cols();++k)
		{
			Lb_u.coeffRef(i,k) = 0.0;
		}
		for(SpMatiter u_(X, i); u_; ++u_)
		{
			double dk = eps;
			for(int k = 0;k<Lt.cols();++k)
			{
				dk += Lt.coeff(u_.row(),k)*(Lb.coeff(i,k)+Lb2.coeff(i,k));           
			} 
			for(int k = 0;k<Lt.cols();++k)
			{
				Lb_u.coeffRef(i,k) += X.coeff(u_.row(),i)*Lt.coeff(u_.row(),k)/dk;
			}
		}
		for(SpMatiter j_(C, i); j_; ++j_)
		{
			for(int k = 0;k<Lt.cols();++k)
			{
				L2_s[j_.row()][k] += L2b.coeff(j_.row(),k)*L3b.coeff(i,j_.row())*Lb_u.coeff(i,k);
			}  
		}
	}
}




void update_gamma_s_context_2_n_r(Mat &L2_s,SpMat const& X, SpMat const& C, SpMat const&Lt, SpMat const&L2b, SpMat const&L3b, SpMat const&Lb2)
{  
    double eps = pow(2,-52);
    SpMat Lb_u(X.cols(),L2_s[0].size());  
    for(int i = 0;i<X.cols();++i)
    {
        for(int k = 0;k<Lt.cols();++k)
        {
            Lb_u.coeffRef(i,k) = 0.0;
        }
        for(SpMatiter u_(X, i); u_; ++u_)
        {
            double dk = eps;
            for(int k = 0;k<Lt.cols();++k)
            {
                dk += Lt.coeff(u_.row(),k)*Lb2.coeff(i,k);           
            } 
            for(int k = 0;k<Lt.cols();++k)
            {
                Lb_u.coeffRef(i,k) += X.coeff(u_.row(),i)*Lt.coeff(u_.row(),k)/dk;
            }
        }
        for(SpMatiter j_(C, i); j_; ++j_)
        {
            for(int k = 0;k<Lt.cols();++k)
            {
                L2_s[j_.row()][k] += L2b.coeff(j_.row(),k)*L3b.coeff(i,j_.row())*Lb_u.coeff(i,k);
            }  
        }
    }
}


//Update the shape parameters of Gamma, contextual pf 
void update_gamma_s_context_3_n(SpMat &L3_s, SpMat const& X, SpMat const& C, SpMat const&Lt,SpMat const&Lb,SpMat const&L2b,SpMat const&L3b, SpMat const&Lb2)
{
    double eps = pow(2,-52);
    SpMat Lb_u(X.cols(),Lt.cols());
    for(int i = 0;i<X.cols();++i)
    {
        for(int k = 0;k<Lt.cols();++k)
        {
            Lb_u.coeffRef(i,k) = 0.0;
        }
        for(SpMatiter u_(X,i); u_; ++u_){
            double dk = eps;
            for(int k = 0;k<Lt.cols();++k)
            {
                dk += Lt.coeff(u_.row(),k)*(Lb.coeff(i,k)+Lb2.coeff(i,k));           
            } 
            for(int k = 0;k<Lt.cols();++k)
            {
                Lb_u.coeffRef(i,k) += X.coeff(u_.row(),i)*Lt.coeff(u_.row(),k)/dk;
            }
        }
          
        for(SpMatiter j_(C,i); j_; ++j_)
        {
    		 for(int k = 0;k<Lt.cols();++k)
    		 {
    			 L3_s.coeffRef(i,j_.row()) += L2b.coeff(j_.row(),k)*L3b.coeff(i,j_.row())*Lb_u.coeff(i,k);
			 }
        }
    }
}


void update_gamma_s_context_3_n_r(SpMat &L3_s,SpMat const& X, SpMat const& C, SpMat const&Lt, SpMat const&L2b, SpMat const&L3b, SpMat const&Lb2)
{
    double eps = pow(2,-52);
    SpMat Lb_u(X.cols(),Lt.cols());
    for(int i = 0;i<X.cols();++i)
    {
        for(int k = 0;k<Lt.cols();++k)
        {
            Lb_u.coeffRef(i,k) = 0.0;
        }
        for(SpMatiter u_(X,i); u_; ++u_)
        {
            double dk = eps;
            for(int k = 0;k<Lt.cols();++k)
            {
                dk += Lt.coeff(u_.row(),k)*Lb2.coeff(i,k);           
            } 
            for(int k = 0;k<Lt.cols();++k)
            {
                Lb_u.coeffRef(i,k) += X.coeff(u_.row(),i)*Lt.coeff(u_.row(),k)/dk;
            }
        }         
        for(SpMatiter j_(C,i); j_; ++j_)
        {
			for(int k = 0;k<Lt.cols();++k)
			{
				L3_s.coeffRef(i,j_.row()) += L2b.coeff(j_.row(),k)*L3b.coeff(i,j_.row())*Lb_u.coeff(i,k);
			}
        }
    }
}



//Update the shape parameters of Gamma, contextual pf 
void update_gamma_s_context_2(MSpMat &L2_s,MSpMat const& X,MSpMat const& C, SpMat const&Lt,SpMat const&Lb,SpMat const&L2b, SpMat const&Lb2){
  
	double eps = pow(2,-52);
	SpMat Lb_u(X.cols(),L2_s.cols());
  
	for(int i = 0;i<X.cols();++i){
		for(int k = 0;k<Lt.cols();++k){
			Lb_u.coeffRef(i,k) = 0.0;
		}
		for(InIterMat u_(X, i); u_; ++u_){
			double dk = eps;
			for(int k = 0;k<Lt.cols();++k){
				//Lb_u.coeffRef(i,k) = 0.0;
				dk += Lt.coeff(u_.row(),k)*(Lb.coeff(i,k)+Lb2.coeff(i,k));           
			} 
			for(int k = 0;k<Lt.cols();++k){
				Lb_u.coeffRef(i,k) += X.coeff(u_.row(),i)*Lt.coeff(u_.row(),k)/dk;
			}
		}
		for(InIterMat j_(C, i); j_; ++j_){
			for(int k = 0;k<Lt.cols();++k){
				L2_s.coeffRef(j_.row(),k) += L2b.coeff(j_.row(),k)*Lb_u.coeff(i,k);
			}  
		}
	}
}











//update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r_context(MSpMat &G_r, MSpMat &L_s,MSpMat &L_r, MSpMat &L2_s,MSpMat &L2_r, MSpMat &C, dVec &K_r, double k_s,double att){  
	for(int k = 0;k<G_r.cols();++k){
		double Sk = 0.0;
		for (InIterMat j_(L_r, k); j_; ++j_){
			Sk += (L_s.coeff(j_.row(),k)/L_r.coeff(j_.row(),k));
			for(InIterMat o_(C, j_.row()); o_; ++o_){
				Sk += (L2_s.coeff(o_.row(),k)/L2_r.coeff(o_.row(),k)); 
			}
		}   
		for(int u = 0;u<G_r.rows();++u){
			G_r.coeffRef(u,k) = att*k_s/K_r[u] + Sk;
		}
	}
}

//update the Gamma rate parameter
void update_gamma_r_context_n(Mat &G_r, Mat &L_s, Mat &L_r, Mat &L2_s, Mat &L2_r, SpMat &L3_s, SpMat &L3_r, SpMat &C, double k_s)
{  
	for(int k = 0;k<G_r[0].size();++k)
	{
		double Sk = 0.0;
		for (int i_ = 0; i_<L_r.size(); ++i_)
		{
    		if (L_r[i_][k]>0.0)
    		{
    			Sk += L_s[i_][k]/L_r[i_][k];
    			for(SpMatiter c_(C, i_); c_; ++c_)
    			{
    				Sk += (L2_s[c_.row()][k]/L2_r[c_.row()][k])*(L3_s.coeff(i_,c_.row())/L3_r.coeff(i_,c_.row()));
    			}
    		}
		}
    
		for(int u = 0;u<G_r.size();++u)
		{
			G_r[u][k] = k_s + Sk;
		}
	}
}


void update_gamma_r_context_n_r(Mat &G_r, Mat &L2_s, Mat &L2_r, SpMat &L3_s, SpMat &L3_r, SpMat const&C, double k_s)
{  
	for(int k = 0;k<G_r[0].size();++k)
	{
		double Sk = 0.0;
		for(int i = 0; i<L2_s.size(); ++i)
		{
    		for(SpMatiter c_(C, i); c_; ++c_)
    		{
    			Sk += (L2_s[c_.row()][k]/L2_r[c_.row()][k])*(L3_s.coeff(i,c_.row())/L3_r.coeff(i,c_.row()));
			}
		}
    
		for(int u = 0;u<G_r.size();++u)
		{
			G_r[u][k] = k_s + Sk;
		}
	}
}



//update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r(Mat &G_r, Mat &L_s, Mat &L_r, double k_s)
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
			G_r[i][k] = k_s + Sk;
		}   
	}
}




//update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r_context_2(MSpMat &G_r, MSpMat &L_s,MSpMat &L_r,dVec &K_r, double k_s,dVec c_sum_c,double att){
	for(int k = 0;k<G_r.cols();++k){
		double Sk = 0.0;
		for (InIterMat j_(L_r, k); j_; ++j_){
			Sk += L_s.coeff(j_.row(),k)/L_r.coeff(j_.row(),k);
		} 
		for(int i = 0;i<G_r.rows();++i){
			G_r.coeffRef(i,k) = att*k_s/K_r[i] + c_sum_c[i]*Sk;  
		} 
	}
}


//update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r_context_2_n(Mat &L2_r, Mat &G_s,Mat &G_r, SpMat &L3_s, SpMat &L3_r, double k_s, SpMat &C)
{  
	for(int k = 0;k<G_r[0].size();++k)
	{
		double Sk = 0.0;
		for (int u_ = 0; u_<G_r.size(); ++u_)
		{
    		if(G_r[u_][k]>0)
    			Sk += G_s[u_][k]/G_r[u_][k];
		}
		for(int j = 0;j<L2_r.size();++j)
		{
			double Sj = 0.0;
			for(SpMatiter i_(C, j); i_; ++i_)
			{
				Sj += (L3_s.coeff(i_.row(),j)/L3_r.coeff(i_.row(),j));
			}
			L2_r[j][k] = k_s + Sj*Sk;
		}
	}
}



//update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r_context_2_n_tied(Mat &L2_r, Mat &G_s, Mat &G_r, SpMat &L3_s, SpMat &L3_r, double k_s, SpMat &C)
{      
    update_gamma_r(L2_r,G_s,G_r,k_s);
	
    for(int k = 0;k<G_r[0].size();++k)
    {
        double Sk = 0.0;
        for (int u_ = 0; u_<G_r.size(); ++u_)
        {
            if(G_r[u_][k]>0)
                Sk += G_s[u_][k]/G_r[u_][k];
        }
        for(int j = 0;j<L2_r.size();++j)
        {
            double Sj = 0.0;
            for(SpMatiter i_(C, j); i_; ++i_)
            {
                Sj += (L3_s.coeff(i_.row(),j)/L3_r.coeff(i_.row(),j));
            }
            L2_r[j][k] += Sj*Sk;
        }
    }
}



// update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r_context_3_n(SpMat &L3_r, Mat &G_s, Mat &G_r, Mat &L2_s, Mat &L2_r, dVec &K_r, dVec &col_sum, double k_s, SpMat &C, SpMat &X, double att){  
    dVec Sk(G_r[0].size());
    for(int k = 0;k<G_r[0].size();++k)
    {
        Sk[k] = 0.0;
        for(int u_ = 0; u_<X.rows(); ++u_)
        {
            if(G_r[u_][k]>0.0)
                Sk[k] += G_s[u_][k]/G_r[u_][k];
        }
    }
    for(int j = 0;j<C.cols();++j)
    {
        double Sj = 0.0; 
        for(int k = 0;k<L2_r[0].size();++k)
        {
            if(L2_r[j][k]>0.0)
                Sj += (L2_s[j][k]/L2_r[j][k])*Sk[k];
        }
        for(SpMatiter c_(C, j); c_; ++c_)
        {
			L3_r.coeffRef(c_.row(),j) = att*(k_s+att*col_sum[c_.row()])/K_r[c_.row()] + Sj;
			//L3_r.coeffRef(c_.row(),j) =k_s/K_r[c_.row()] + Sj;
        }
    }
}



// update the Gamma rate parameter, can be used to update Lambda rate parameter as well 
void update_gamma_r_context_3_n_2(SpMat &L3_r, Mat &G_s,Mat &G_r, Mat &L2_s, Mat &L2_r, dVec &K_r, dVec &col_sum, double k_s, SpMat &C,  SpMat &X)
{  
    dVec Sk(G_r[0].size());
    for(int k = 0;k<G_r[0].size();++k)
    {
        Sk[k] = 0.0;
        for(int u_ = 0; u_<G_r.size(); ++u_)
        {
            if (G_r[u_][k]>0.0)
            Sk[k] += G_s[u_][k]/G_r[u_][k];
        }
    }
    for(int j = 0;j<L2_r.size();++j)
    {
        double Sj = 0.0; 
        for(int k = 0;k<L2_r[0].size();++k)
        {
            if(L2_r[j][k]>0.0)
                Sj += (L2_s[j][k]/L2_r[j][k])*Sk[k];
        }
        for(SpMatiter c_(C, j); c_; ++c_)
        {
			L3_r.coeffRef(c_.row(),j) =k_s/K_r[c_.row()] + Sj;
        }
    }
}





void update_kappa_r_inv(dVec &K_r,  MSpMat &G_s, MSpMat &G_r,double a_,double b_,double att){
	for(int i = 0;i<K_r.size();++i){
		double Sk = 0.0;
		for(int k = 0;k<G_s.cols();++k){
			Sk += G_s.coeff(i,k)/G_r.coeff(i,k);         
		}
		K_r[i] = a_/b_ + att*Sk;
	}  
}



void update_kappa_r_inv_kappa(dVec &K_r,  SpMat &L3_s, SpMat &L3_r, SpMat &C,double a_,double b_,double att){
    for(int i = 0;i<L3_r.rows();++i)
    {
        double Si = 0.0;
        for(SpMatiter j_(C, i); j_; ++j_)
        {
        	 Si += (L3_s.coeff(i,j_.row())/L3_r.coeff(i,j_.row()));
        }
        K_r[i] = a_/b_ + att*Si;
	}
}


/************* Util functions *************/ 
 
void set_coeffs_to(Mat &L_s,double c_)
{
	for(int j = 0;j<L_s.size();++j){
    	for(int k = 0;k<L_s[0].size();++k)
    	{
			L_s[j][k] = c_;
		}
	}
}


// Set all entries of sparse matrix into a particular value
void set_coeffs_to_sparse(SpMat &L_s, double c_)
{
    for(int nz = 0;nz<L_s.outerSize();++nz)
    {
        for (SpMatiter i_(L_s,nz); i_; ++i_)
        {
    		 L_s.coeffRef(i_.row(),i_.col()) = c_;
        }
	}
}

//Compute the expectation of a Sparse matrix contaning logGamma elements 
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

SpMat triplet_to_csc_sparse(Mat const& M, int n_row, int n_col)
{
    SpMat spM(n_row, n_col);
    for(int i = 0;i<M.size();++i)
    {
        spM.coeffRef(M[i][0],M[i][1]) = M[i][2];
    }
      
    return spM;
}

void csc_sparse_to_triplet(SpMat const&spM, Mat &M)
{
    int i = 0; 
    for(int nz = 0;nz<spM.outerSize();++nz)
    {
        for (SpMatiter i_(spM,nz); i_; ++i_)
        {   
            M[i][0] = i_.row();
            M[i][1] = i_.col();
            M[i][2] = spM.coeff(i_.row(),i_.col());
            i+=1;
        }
    }
}
  
 
/*============================================================================================================*/

void c2pf_cpp(Mat const&tX, Mat const&tC, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &L2_s, Mat &L2_r, Mat &tL3_s, Mat &tL3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt)  
{	
    //data shape
    int n  = G_s.size();
    int d  = L_s.size();
    int d2 = L2_s.size();
    
    //create sparse matrices from triplet
    SpMat X = triplet_to_csc_sparse(tX,n,d);
    SpMat C = triplet_to_csc_sparse(tC,d,d2);
    SpMat L3_s = triplet_to_csc_sparse(tL3_s,d,d2);
    SpMat L3_r = triplet_to_csc_sparse(tL3_r,d,d2);
  
    //Hyper parameter setting
    double att = 1.0;
    double aa   = 0.3;
    //double a_  = 6.;
    double a1_ = 5.;
    double a_t = at;
    double b_t = bt;
    double cc   = 0.3;
    double c1  = 5.;
    double c2 = 2.;
    double b_ = 1.0;
    double d_ = 1.0;
    double ee  = 0.3;
    double e_ = 0.3;
    //double k_s = a_ + g*a;
    //double t_s = a1 + g*c;
    double k_s = aa;
    double t_s = aa;
    double t2_s = t_s;
  
    //Util variables declaration
    SpMat Lt(n,g);
    SpMat Lb(d,g);
    SpMat L2b(d2,g);
    SpMat Lb2(d,g);
    SpMat L3b(d,d2);
	
   
    //Rcout << "c2pf_cpp: " <<std::endl;
  
    update_kappa_r_inv_kappa(T3_r,L3_s,L3_r,C,b_t,b_,a_t);
  
    //Learning 
  
    Lt = E_SpMat_logGamma(G_s,G_r);
    Lt.coeffs() = Lt.coeffs().exp();
    
    Lb = E_SpMat_logGamma(L_s,L_r);
    Lb.coeffs() = Lb.coeffs().exp();
    
    L2b = E_SpMat_logGamma(L2_s,L2_r);
    L2b.coeffs() = L2b.coeffs().exp();
    
    L3b = E_SpMat_logGamma(L3_s,L3_r);
    L3b.coeffs() = L3b.coeffs().exp();
    
    //compute agregated statistics of item context
    for(int i = 0;i<d;++i)
    {
        for(int k=0;k<g;++k)
        {
            Lb2.coeffRef(i,k) = 0.0;
            for(SpMatiter c_(C, i); c_; ++c_)
            {
                Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
            }
        }
    }

    for(int iter = 0;iter<maxiter;++iter)
    {  
    	 ///Update item influence factor kappa_ij///
    
        //update Lambda_S for item influence factors
        set_coeffs_to_sparse(L3_s,a_t);
        update_gamma_s_context_3_n(L3_s,X,C,Lt,Lb,L2b,L3b,Lb2);
    
        //update Lambda_R
        //update_gamma_r_context_3_n(L3_r,G_s,G_r,L2_s,L2_r,T3_r,util_sum,b_t,C,X,att);   
        update_gamma_r_context_3_n(L3_r,G_s,G_r,L2_s,L2_r,T3_r,util_sum,a1_,C,X,a_t);   
    
        L3b = E_SpMat_logGamma(L3_s,L3_r);
        L3b.coeffs() = L3b.coeffs().exp();
    
        //compute agregated statistics of item context
        for(int i = 0;i<d;++i)
        {
            for(int k=0;k<g;++k){
                Lb2.coeffRef(i,k) = 0.0;
                for(SpMatiter c_(C, i); c_; ++c_)
                {
                    Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
                }
            }
        }

        update_kappa_r_inv_kappa(T3_r,L3_s,L3_r,C,b_t,b_,a_t);
    
        ///Update user related parameters////
   
        //updare Gamma_S
        set_coeffs_to(G_s,aa);
        update_gamma_s_context(G_s,X,Lt,Lb,Lb2);
     
        //update Gamma_R
        update_gamma_r_context_n(G_r,L_s,L_r,L2_s,L2_r,L3_s,L3_r,C,k_s);
	
        Lt = E_SpMat_logGamma(G_s,G_r);
        Lt.coeffs() = Lt.coeffs().exp();
    
    
        ///Update item related parameters///
    
        //updare Lambda_S
        set_coeffs_to(L_s,cc);
        update_lambda_s_context(L_s,X,Lt,Lb,Lb2);
     
        //update Lambda_R
        update_gamma_r(L_r,G_s,G_r,t_s);

		
        Lb = E_SpMat_logGamma(L_s,L_r);
        Lb.coeffs() = Lb.coeffs().exp();    
    
  
        ///Update items' context factors ///
    
        //update Lambda_S for context items
        set_coeffs_to(L2_s,ee);
        update_gamma_s_context_2_n(L2_s,X,C,Lt,Lb,L2b,L3b,Lb2);
    
        //update Lambda_R
        update_gamma_r_context_2_n(L2_r,G_s,G_r,L3_s,L3_r,t2_s,C);
	
        L2b = E_SpMat_logGamma(L2_s,L2_r);
        L2b.coeffs() = L2b.coeffs().exp();
    
        //compute agregated statistics of item context
        for(int i = 0;i<d;++i)
        {
            for(int k=0;k<g;++k)
            {
                Lb2.coeffRef(i,k) = 0.0;
                for(SpMatiter c_(C, i); c_; ++c_)
                {
                    Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
                }
            }
        }
     
        //if((iter%10)==0)
        //Rcout << "iter: " << iter <<std::endl;
    
    //////////////////////// End of learning /////////////////////////// 
    
    }
    //update the triplet matrices, this step is necessary to capture the changes in python
    csc_sparse_to_triplet(L3_s, tL3_s);
    csc_sparse_to_triplet(L3_r, tL3_r);
    //X = NULL;
    //C = NULL;
    //L3_s = NULL; 
    //L3_r = NULL;
    
}



void c2pf_cpp2(Mat const&tX, Mat const&tC, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &L2_s, Mat &L2_r, Mat &tL3_s, Mat &tL3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter,double at,double bt)
{
  
    //data shape
    int n  = G_s.size();
    int d  = L_s.size();
    int d2 = L2_s.size();
    
    //create sparse matrices from triplet
    SpMat X = triplet_to_csc_sparse(tX,n,d);
    SpMat C = triplet_to_csc_sparse(tC,d,d2);
    SpMat L3_s = triplet_to_csc_sparse(tL3_s,d,d2);
    SpMat L3_r = triplet_to_csc_sparse(tL3_r,d,d2);     
  
	//Hyper parameter setting
    double att = 1.0;
    double aa   = 0.3;
    double a_  = 6.;
    double a1_ = 5.;
    double a_t = at;
    double b_t = bt;
    double cc   = 0.3;
    double c1  = 5.;
    double c2 = 2.;
    double b_ = 1.0;
    double d_ = 1.0;
    double ee  = 0.3;
    double e_ = 0.3;
    //double k_s = a_ + g*a;
    //double t_s = a1 + g*c;
    double k_s = aa;
    double t_s = aa;
    double t2_s = t_s;
  
    //Util variables declaration
    SpMat Lt(n,g);
    SpMat Lb(d,g);
    SpMat L2b(d2,g);
    SpMat Lb2(d,g);
    SpMat L3b(d,d2);
  
    //update_kappa_r_inv_kappa(T3_r,L3_s,L3_r,C,c2,b_,a_t);
  
    //Learning 
    Lt = E_SpMat_logGamma(G_s,G_r);
    Lt.coeffs() = Lt.coeffs().exp();
    
    Lb = E_SpMat_logGamma(L_s,L_r);
    Lb.coeffs() = Lb.coeffs().exp();
    
    L2b = E_SpMat_logGamma(L2_s,L2_r);
    L2b.coeffs() = L2b.coeffs().exp();
    
    L3b = E_SpMat_logGamma(L3_s,L3_r);
    L3b.coeffs() = L3b.coeffs().exp();
    
    //compute agregated statistics of item context
    for(int i = 0;i<d;++i)
    {
        for(int k=0;k<g;++k)
        {
            Lb2.coeffRef(i,k) = 0.0;
            for(SpMatiter c_(C, i); c_; ++c_)
            {
                Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
            }
        }
    }
    

    for(int iter = 0;iter<maxiter;++iter)
    {  
        ///Update item influence factor kappa_ij///
        //update Lambda_S for item influence factors
        set_coeffs_to_sparse(L3_s,a_t);
        update_gamma_s_context_3_n(L3_s,X,C,Lt,Lb,L2b,L3b,Lb2);
    
        //update Lambda_R
        update_gamma_r_context_3_n_2(L3_r,G_s,G_r,L2_s,L2_r,T3_r,util_sum,b_t,C,X);   
        //update_gamma_r_context_3_n(L3_r,G_s,G_r,L2_s,L2_r,T3_r,util_sum,a1_,C,X,a_t);   
        
        L3b = E_SpMat_logGamma(L3_s,L3_r);
        L3b.coeffs() = L3b.coeffs().exp();
    
        //compute agregated statistics of item context
        for(int i = 0;i<d;++i)
        {
            for(int k=0;k<g;++k)
            {
                Lb2.coeffRef(i,k) = 0.0;
                for(SpMatiter c_(C, i); c_; ++c_)
                {
                    Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
                }
            }
        }

        //update_kappa_r_inv_kappa(T3_r,L3_s,L3_r,C,c2,b_,a_t);
	  
    
        ///Update user related parameters////
   
        //updare Gamma_S
        set_coeffs_to(G_s, aa);
        update_gamma_s_context(G_s,X,Lt,Lb,Lb2);
     
        //update Gamma_R
        update_gamma_r_context_n(G_r,L_s,L_r,L2_s,L2_r,L3_s,L3_r,C,k_s);
	
        Lt = E_SpMat_logGamma(G_s,G_r);
        Lt.coeffs() = Lt.coeffs().exp();
    
    
        ///Update item related parameters///
    
        //updare Lambda_S
        set_coeffs_to(L_s,cc);
        update_lambda_s_context(L_s,X,Lt,Lb,Lb2);
    
        //update Lambda_R
        update_gamma_r(L_r,G_s,G_r,t_s);
	
        Lb = E_SpMat_logGamma(L_s,L_r);
        Lb.coeffs() = Lb.coeffs().exp();

    
        ///Update items' context factors ///
    
        //update Lambda_S for context items
        set_coeffs_to(L2_s,ee);
        update_gamma_s_context_2_n(L2_s,X,C,Lt,Lb,L2b,L3b,Lb2);
    
        //update Lambda_R
        update_gamma_r_context_2_n(L2_r,G_s,G_r,L3_s,L3_r,t2_s,C);
	
        L2b = E_SpMat_logGamma(L2_s,L2_r);
        L2b.coeffs() = L2b.coeffs().exp();
    
        //compute agregated statistics of item context
        for(int i = 0;i<d;++i)
        {
            for(int k=0;k<g;++k)
            {
                Lb2.coeffRef(i,k) = 0.0;
                for(SpMatiter c_(C, i); c_; ++c_)
                {
                    Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
                }
            }
        }

    
    //////////////////////// End of learning /////////////////////////// 
    }
    //update the triplet matrices, this step is necessary to capture the changes in python
    csc_sparse_to_triplet(L3_s, tL3_s);
    csc_sparse_to_triplet(L3_r, tL3_r);
}



//tied-C2PF
void tc2pf_cpp(Mat const&tX, Mat const&tC, int const&g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &tL3_s, Mat &tL3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt)
{
    
    //data shape
    int n  = G_s.size();
    int d  = L_s.size();
    int d2 = L_s.size();
    
    //create sparse matrices from triplet
    SpMat X = triplet_to_csc_sparse(tX,n,d);
    SpMat C = triplet_to_csc_sparse(tC,d,d2);
    SpMat L3_s = triplet_to_csc_sparse(tL3_s,d,d2);
    SpMat L3_r = triplet_to_csc_sparse(tL3_r,d,d2); 

  
    //Hyper parameter setting
    double aa  = 0.3;
    double a_ = 0.3;
    double att = 1.;
    double a_t = at;
    double b_t = bt;
    double cc  = 0.3;
    double c_ = 5.;
    double b_ = 1.0;
    double d_ = 1.0;
    double ee  = 0.3;
    double k_s = a_;
    double t_s = c_;
	//double eps = pow(2.0,-52);
  
    //Util variables declaration
    SpMat Lt(n,g);
    SpMat Lb(d,g);
    SpMat Lb2(d,g);
    SpMat L3b(d,d2);
    
    //Learning
	
    Lt = E_SpMat_logGamma(G_s,G_r);
    Lt.coeffs() = Lt.coeffs().exp();

    Lb = E_SpMat_logGamma(L_s,L_r);
    Lb.coeffs() = Lb.coeffs().exp();

    L3b = E_SpMat_logGamma(L3_s,L3_r);
    L3b.coeffs() = L3b.coeffs().exp();

	//compute agregated statistics of item context
    for(int i = 0;i<d;++i)
    {
        for(int k=0;k<g;++k)
        {
            Lb2.coeffRef(i,k) = 0.0;
            for(SpMatiter c_(C, i); c_; ++c_)
            {
                Lb2.coeffRef(i,k)+= Lb.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
            }
        }
    }
  
	for(int iter = 0;iter<maxiter;++iter){
    
		///Update item influence factor kappa_ij///
    
		//update Lambda_S for item influence factors
		set_coeffs_to_sparse(L3_s,a_t);
		update_gamma_s_context_3_n(L3_s,X,C,Lt,Lb,Lb,L3b,Lb2);
    
    
		//update Lambda_R
		update_gamma_r_context_3_n_2(L3_r,G_s,G_r,L_s,L_r,T3_r,util_sum,b_t,C,X); 
		
		L3b = E_SpMat_logGamma(L3_s,L3_r);
		L3b.coeffs() = L3b.coeffs().exp();

		//compute agregated statistics of item context
       for(int i = 0;i<d;++i)
       {
           for(int k=0;k<g;++k)
           {
               Lb2.coeffRef(i,k) = 0.0;
               for(SpMatiter c_(C, i); c_; ++c_)
               {
                    Lb2.coeffRef(i,k)+= Lb.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
               }
           }
       }
		
    
		///Update user related parameters////
    
		//updare Gamma_S
		set_coeffs_to(G_s,aa);
		update_gamma_s_context(G_s,X,Lt,Lb,Lb2);
    
		//update Gamma_R
		update_gamma_r_context_n(G_r,L_s,L_r,L_s,L_r,L3_s,L3_r,C,k_s);
		
		Lt = E_SpMat_logGamma(G_s,G_r);
		Lt.coeffs() = Lt.coeffs().exp();
    
		///Update item related parameters///
    
		//updare Lambda_S
		set_coeffs_to(L_s,cc);
		update_lambda_s_context(L_s,X,Lt,Lb,Lb2);
		update_gamma_s_context_2_n(L_s,X,C,Lt,Lb,Lb,L3b,Lb2);
    
		//update Lambda_R
		update_gamma_r_context_2_n_tied(L_r,G_s,G_r,L3_s,L3_r,ee,C);
    
		Lb = E_SpMat_logGamma(L_s,L_r);
		Lb.coeffs() = Lb.coeffs().exp();
		
		//compute agregated statistics of item context
       for(int i = 0;i<d;++i)
       {
           for(int k=0;k<g;++k)
           {
               Lb2.coeffRef(i,k) = 0.0;
               for(SpMatiter c_(C, i); c_; ++c_)
               {
                    Lb2.coeffRef(i,k)+= Lb.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
               }
           }
       }       
		//////////////////////// End of learning /////////////////////////// 
	}
    //update the triplet matrices, this step is necessary to capture the changes in python
    csc_sparse_to_triplet(L3_s, tL3_s);
    csc_sparse_to_triplet(L3_r, tL3_r);
}



//reduced C2PF
void rc2pf_cpp(Mat const&tX, Mat const&tC, int const&g, Mat &G_s, Mat &G_r,Mat &L2_s, Mat &L2_r, Mat tL3_s, Mat tL3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt)
{  
    //data shape
    int n  = G_s.size();
    int d  = L2_s.size();
    int d2 = L2_s.size();
	
    //create sparse matrices from triplet
    SpMat X = triplet_to_csc_sparse(tX,n,d);
    SpMat C = triplet_to_csc_sparse(tC,d,d2);
    SpMat L3_s = triplet_to_csc_sparse(tL3_s,d,d2);
    SpMat L3_r = triplet_to_csc_sparse(tL3_r,d,d2); 
  
    //Hyper parameter setting
    double att = 1.0;
    double aa   = 0.3;
    double a_  = 3.;
    double a_t = at;
    double b_t = bt;
    double cc   = 0.3;
    double c_  = 2.;
    double b_ = 1.0;
    double d_ = 1.0;
    double ee  = 0.3;
    double e_ = 0.3;
	//double k_s = a1 + g*a;
	//double t_s = a1 + g*c;
	double k_s = aa;
	double t_s = cc;
  
    //Util variables declaration
    SpMat Lt(n,g);
    SpMat L2b(d2,g);
    SpMat Lb2(d,g);
    SpMat L3b(d,d2);
    
  
    //Learning 
    Lt = E_SpMat_logGamma(G_s,G_r);
    Lt.coeffs() = Lt.coeffs().exp();
    
    L2b = E_SpMat_logGamma(L2_s,L2_r);
    L2b.coeffs() = L2b.coeffs().exp();
    
    L3b = E_SpMat_logGamma(L3_s,L3_r);
    L3b.coeffs() = L3b.coeffs().exp();
    
    //compute agregated statistics of item context
    for(int i = 0;i<d;++i)
    {
        for(int k=0;k<g;++k)
        {
            Lb2.coeffRef(i,k) = 0.0;
            for(SpMatiter c_(C, i); c_; ++c_)
            {
                Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
            }
        }
    }

	for(int iter = 0;iter<maxiter;++iter){
    
        ///Update item influence factor kappa_ij///
    
        //update Lambda_S for item influence factors
        set_coeffs_to_sparse(L3_s,a_t);
        update_gamma_s_context_3_n_r(L3_s,X,C,Lt,L2b,L3b,Lb2);
 
        //update Lambda_R
        update_gamma_r_context_3_n_2(L3_r,G_s,G_r,L2_s,L2_r,T3_r,util_sum,b_t,C,X);   
        //update_gamma_r_context_3_n(L3_r,G_s,G_r,L2_s,L2_r,T3_r,util_sum,a_,C,X,a_t);   
        //update_kappa_r_inv_kappa(T3_r,L3_s,L3_r,C,c_,b_,at);
        L3b = E_SpMat_logGamma(L3_s,L3_r);
        L3b.coeffs() = L3b.coeffs().exp();
    
		//compute agregated statistics of item context
        for(int i = 0;i<d;++i)
        {
            for(int k=0;k<g;++k)
            {
                Lb2.coeffRef(i,k) = 0.0;
                for(SpMatiter c_(C, i); c_; ++c_)
                {
                    Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
                }
            }
        }

		///Update user related parameters////
   
		//updare Gamma_S
		set_coeffs_to(G_s,aa);
		update_gamma_s_context_r(G_s,X,Lt,Lb2);
     
		//update Gamma_R
		update_gamma_r_context_n_r(G_r,L2_s,L2_r,L3_s,L3_r,C,k_s);
	
		
		Lt = E_SpMat_logGamma(G_s,G_r);
		Lt.coeffs() = Lt.coeffs().exp();
    
    
		///Update item related parameters///
    
		//nothing to be done
   
		///Update items' context factors ///
    
		//update Lambda_S for context items
		set_coeffs_to(L2_s,ee);
		update_gamma_s_context_2_n_r(L2_s,X,C,Lt,L2b,L3b,Lb2);
    
		//update Lambda_R
		update_gamma_r_context_2_n(L2_r,G_s,G_r,L3_s,L3_r,t_s,C);
	
		
		L2b = E_SpMat_logGamma(L2_s,L2_r);
		L2b.coeffs() = L2b.coeffs().exp();
    
		//compute agregated statistics of item context
        for(int i = 0;i<d;++i)
        {
            for(int k=0;k<g;++k)
            {
                Lb2.coeffRef(i,k) = 0.0;
                for(SpMatiter c_(C, i); c_; ++c_)
                {
                    Lb2.coeffRef(i,k)+= L2b.coeff(c_.row(),k)*L3b.coeff(i,c_.row());
                }
            }
        }
		
		//////////////////////// End of learning /////////////////////////// 
    }
    //update the triplet matrices, this step is necessary to capture the changes in python
    csc_sparse_to_triplet(L3_s, tL3_s);
    csc_sparse_to_triplet(L3_r, tL3_r);
}