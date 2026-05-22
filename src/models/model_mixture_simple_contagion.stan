data {
  int<lower=1> kmax;
  int<lower=1> V;
  int<lower=1> K;
  array[V, kmax] real<lower=0> M;
  array[V, kmax] real<lower=0> N;
  // HYPERPARAMETERS - logit-scale Normal prior on rho
  real LOGIT_RHO_LOCATION;
  real<lower=0> LOGIT_RHO_SCALE;
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
  logit_rho ~ normal(LOGIT_RHO_LOCATION, LOGIT_RHO_SCALE);
  pi ~ dirichlet(rep_vector(1.0, K));
  
  for (i in 1:V) {
    vector[K] lps = log(pi);
    for (k in 1:K) {
        for (nu in 1:kmax) {
            if (M[i, nu] > 0 || N[i, nu] > 0) {
                // p(nu) = 1 - (1-rho)^nu, evaluated in log space for stability
                real log1m_rho = log_inv_logit(-logit_rho[k]);      // log(1-rho)
                real log_survival_nu = nu * log1m_rho;              // log((1-rho)^nu)
                real log_p_nu = log1m_exp(log_survival_nu);         // log(1-(1-rho)^nu)
                lps[k] += M[i, nu] * log_p_nu + N[i, nu] * log_survival_nu;
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
            real log1m_rho = log_inv_logit(-logit_rho[k]);          // log(1-rho)
            real log_survival_nu = nu * log1m_rho;                  // log((1-rho)^nu)
            mu[k, nu] = -expm1(log_survival_nu);                    // 1 - exp(log_survival_nu)
        }
    }
    
    // Posterior prob of each component for each node
    array[V] simplex[K] post_prob;
    vector[V] log_lik;
    for (i in 1:V) {
        vector[K] lps = log(pi);
        for (k in 1:K) {
            for (nu in 1:kmax) {
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    real log1m_rho = log_inv_logit(-logit_rho[k]);  // log(1-rho)
                    real log_survival_nu = nu * log1m_rho;          // log((1-rho)^nu)
                    real log_p_nu = log1m_exp(log_survival_nu);     // log(1-(1-rho)^nu)
                    lps[k] += M[i, nu] * log_p_nu + N[i, nu] * log_survival_nu;
                }
            }
        }
        log_lik[i] = log_sum_exp(lps);
        post_prob[i] = softmax(lps);
    }
}
