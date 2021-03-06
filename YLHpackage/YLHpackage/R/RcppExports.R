# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#'Two method for Lasso
#'
#'@param y the response vector
#'@param X the design matrix
#'@param lambda the penalty parameter
#'@param e the threhold for convergence
#'@return the estimate for beta
#'@example
#'require(CDLasso)
#'X=matrix(c(1,1,-1,-1),nrow=2)
#'y=c(2,-2)
#'lambda=1
#'e=0.01
#'CDLasso(y,X,lambda,c)
CDLasso <- function(y, X, lambda, e) {
    .Call('_YLHpackage_CDLasso', PACKAGE = 'YLHpackage', y, X, lambda, e)
}

#'Two method for Lasso
#'
#'@param y the response vector
#'@param X the design matrix
#'@param lambda the penalty parameter
#'@param e the threhold for convergence
#'@return the estimate for beta
#'@example
#'require(POLasso)
#'X=matrix(c(1,1,-1,-1),nrow=2)
#'y=c(2,-2)
#'lambda=1
#'e=0.01
#'POLasso(y,X,lambda,e)
POLasso <- function(y, X, lambda, e) {
    .Call('_YLHpackage_POLasso', PACKAGE = 'YLHpackage', y, X, lambda, e)
}

