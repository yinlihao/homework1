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
//'require(CDLasso)
//'X=matrix(c(1,1,-1,-1),nrow=2)
//'y=c(2,-2)
//'lambda=1
//'e=0.01
//'CDLasso(y,X,lambda,c)
// [[Rcpp::export]]
arma::vec CDLasso(const arma::vec& y,const arma::mat& X,double lambda,double e){
  arma::vec beta_hat;
  arma::vec r;
  int p=X.n_cols;
  int N=X.n_rows;
  beta_hat.zeros(p);
  r=y-X*beta_hat;
  arma::vec tempbeta=beta_hat;
  tempbeta(0)=1;
  
  while(sum((tempbeta-beta_hat)%(tempbeta-beta_hat))>e) 
  {
    tempbeta=beta_hat;
    for(int i=0;i<p;i++)
    {
      double z=sum(r%X.col(i))/N+beta_hat(i);
      double beta_plus;
      if(z>lambda)
      {
        beta_plus=z-lambda;
      }
      else if(z<(-lambda))
      {
        beta_plus=z+lambda;
      }
      else
      {
        beta_plus=0;
      }
      r=r+X.col(i)*(-beta_plus+beta_hat(i));
      beta_hat(i)=beta_plus;
    }
  }
  
  return beta_hat;
}
