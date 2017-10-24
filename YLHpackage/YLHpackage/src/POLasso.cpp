#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
//'Two method for Lasso
//'
//'@param y the response vector
//'@param X the design matrix
//'@param lambda the penalty parameter
//'@param e the threhold for convergence
//'@return the estimate for beta
//'@example
//'require(POLasso)
//'X=matrix(c(1,1,-1,-1),nrow=2)
//'y=c(2,-2)
//'lambda=1
//'e=0.01
//'POLasso(y,X,lambda,e)
// [[Rcpp::export]]
arma::vec POLasso(arma::vec& y,arma::mat& X,double lambda,double e){
  arma::vec beta_hat;
  arma::vec r;
  int p=X.n_cols;
  int N=X.n_rows;
  beta_hat.zeros(p);
  r=y-X*beta_hat;
  double M;
  arma::vec eigen_value=eig_sym(X.t()*X);
  M=max(eigen_value)/N;
  arma::vec tempbeta=beta_hat;
  tempbeta(0)=1;
  
  while(sum((tempbeta-beta_hat)%(tempbeta-beta_hat))>e);
  {
    tempbeta=beta_hat;
    arma::vec SFbeta=beta_hat+(X.t()*(y-X*beta_hat))/(N*M);
    for(int i=0;i<p;i++)
    {
      if(SFbeta(i)>(lambda/M))
      {
        beta_hat(i)=SFbeta(i)-(lambda/M);
      }
      else if(SFbeta(i)<(-(lambda/M)))
      {
        beta_hat(i)=SFbeta(i)+(lambda/M);
      }
      else
      {
        beta_hat(i)=0;
      }
    }
  }
  
  return beta_hat;
}