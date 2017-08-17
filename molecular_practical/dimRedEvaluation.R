# Trustworthinesss & continuity
#
# X: a high-dimensional nxd matrix.
# Y: a low-dimensional nxm (m<d) representation of X.
# k: number of neighbors as integer. Default = 12
# k2: number of neighbors as integer. (OPTIONAL)
# This defines the upper bound for neighbors so that the measurements are done with all k in range [k,k2].
# similarity.matrix: if TRUE, X and Y are taken to be similarity matrices. Default = FALSE
# If FALSE, similarity matrices are calculated from X and Y with Euclidean distance.

trustworthiness <- function(X, Y, k = 12, k2 = 0, similarity.matrix = FALSE) {
  n = dim(X)[1]
  d = dim(X)[2]
  N = dim(Y)[1]
  m = dim(Y)[2]
  if (n != N) stop("X and Y have different number of observations.")
  if(similarity.matrix) {
    if(n != d) stop("X is not symmetric.")
    if(N != m) stop("Y is not symmetric.")
  }
  if(abs(k - round(k)) > .Machine$double.eps^0.5) stop("k is not an integer.")
  if(k >= n) stop("k can't be equal or more than the number of observations.")
  if(k < 1) stop("k can't be less than 1.")
  if(k2 != 0 && k2 < k) stop("k2 can't be less than k.")
  if(k2 == 0) k2 = k
  
  if(!similarity.matrix) {
    if(require(fields)) {
      X = rdist(X)
      Y = rdist(Y)
    } else {
      X = as.matrix(dist(X))
      Y = as.matrix(dist(Y))
    } 
  }
  
  X_ind = apply(X,1,order)
  Y_ind = apply(Y,1,order)
  
  T1 = vector()
  
  for(k1 in k:k2) {
    trust = 0
    ranks = c()
    for(i in 1:n) {
      for(j in 1:k1) {
        ind = X_ind[,i] == Y_ind[j+1,i]
        ranks[j] = which(ind != 0)
      }
      ranks = ranks - k1 - 1
      trust = trust + sum(ranks[ranks > 0])
    }
    if (k1 < n/2) A = n*k1*(2*n - 3*k1 - 1)
    else A = n*(n - k1)*(n - k1 - 1)
    trust = 1 - ((2 / A) * trust)
    T1[k1-(k-1)] = trust
  }
  return(T1)
}

continuity <- function(X, Y, k = 12, k2 = 0, similarity.matrix = FALSE) {
  n = dim(X)[1]
  d = dim(X)[2]
  N = dim(Y)[1]
  m = dim(Y)[2]
  if (n != N) stop("X and Y have different number of observations.")
  if(similarity.matrix) {
    if(n != d) stop("X is not symmetric.")
    if(N != m) stop("Y is not symmetric.")
  }
  if(abs(k - round(k)) > .Machine$double.eps^0.5) stop("k is not an integer.")
  if(k >= n) stop("k can't be equal or more than the number of observations.")
  if(k < 1) stop("k can't be less than 1.")
  if(k2 != 0 && k2 < k) stop("k2 can't be less than k.")
  if(k2 == 0) k2 = k
  
  if(!similarity.matrix) {
    if(require(fields)) {
      X = rdist(X)
      Y = rdist(Y)
    } else {
      X = as.matrix(dist(X))
      Y = as.matrix(dist(Y))
    } 
  }
  
  X_ind = apply(X,1,order)
  Y_ind = apply(Y,1,order)
  
  C1 = vector()
  
  for(k1 in k:k2) {
    conti = 0
    ranks = c()
    for(i in 1:n) {
      for(j in 1:k1) {
        ind = Y_ind[,i] == X_ind[j+1,i]
        ranks[j] = which(ind != 0)
      }
      ranks = ranks - k1 - 1
      conti = conti + sum(ranks[ranks > 0])
    }
    if (k1 < n/2) A = n*k1*(2*n - 3*k1 - 1)
    else A = n*(n - k1)*(n - k1 - 1)
    conti = 1 - ((2 / A) * conti)
    C1[k1-(k-1)] = conti
  }
  return(C1)
}