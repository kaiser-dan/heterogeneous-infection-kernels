data {
  int<lower=1> kmax;
  int<lower=1> V;
  int<lower=1> K;
  array[V, kmax] real<lower=0> M;
  array[V, kmax] real<lower=0> N;
}

parameters {
  ordered[K] logit_rho;
  simplex[K] pi;
}

transformed parameters {
  vector<lower=0, upper=1>[K] rho = inv_logit(logit_rho);
}

model {
  // Priors
  logit_rho ~ normal(-4, 2); 
  pi ~ dirichlet(rep_vector(1.0, K));
  
  for (i in 1:V) {
    vector[K] lps = log(pi);
    for (k in 1:K) {
        for (nu in 1:kmax) {
            if (M[i, nu] > 0 || N[i, nu] > 0) {
                real p = 1.0 - pow(1.0 - rho[k], nu);
                p = fmax(1e-12, fmin(1.0 - 1e-12, p));
                lps[k] += M[i, nu] * log(p) + N[i, nu] * log(1.0 - p);
            }
        }
    }
    target += log_sum_exp(lps);
  }
}

generated quantities {
    // Predicted p_k for each component
    array[K] vector[kmax] mu;
    for (k in 1:K) {
        for (nu in 1:kmax) {
            mu[k, nu] = 1.0 - pow(1.0 - rho[k], nu);
        }
    }
    
    // Posterior prob of each component for each node
    array[V] simplex[K] post_prob;
    for (i in 1:V) {
        vector[K] lps = log(pi);
        for (k in 1:K) {
            for (nu in 1:kmax) {
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    real p = mu[k, nu];
                    p = fmax(1e-12, fmin(1.0 - 1e-12, p));
                    lps[k] += M[i, nu] * log(p) + N[i, nu] * log(1.0 - p);
                }
            }
        }
        post_prob[i] = softmax(lps);
    }
}
