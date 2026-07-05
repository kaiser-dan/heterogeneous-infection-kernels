/*
 * Non-parametric monotonic mixture model with EMPIRICAL BAYES priors
 *
 * KEY CHANGE: logit_p prior now centered at empirical scale from data
 * instead of at logit(0.5) = 0
 *
 * This fixes the issue where prior at rho = 0.5 dominates likelihood at rho = 0.0001
 */

data {
	// number of individuals in population
    int<lower=2> V;

    // max number of neighbors
	int<lower=1, upper=V> kmax;

    // number of kernel modes (mixture components)
    int<lower=1, upper=V> K;

    // M[i,j] = number of infections (expected) when node i has j exposures
    array[V, kmax] real<lower=0> M;

    // N[i,j] = number of infections failing to occur (expected) when node i has j exposures
    array[V, kmax] real<lower=0> N;

    // EMPIRICAL BAYES: prior location for logit_p (computed from data)
    real LOGIT_P_LOCATION;

    // EMPIRICAL BAYES: prior scale for logit_p
    real<lower=0> LOGIT_P_SCALE;

    // HYPERPARAMETER - Dirichlet concentration for increments
    real<lower=0> DELTA_CONCENTRATION;

    // HYPERPARAMETER - HalfNormal(0, SIGMA_SCALE) prior on sigma
    real<lower=0> SIGMA_SCALE;

    // HYPERPARAMETER - Dirichlet(PI_CONCENTRATION) prior on mixture weights
    real<lower=0> PI_CONCENTRATION;
}

parameters {
    // scaling factors for kernel modes on logit scale
    ordered[K] logit_p;

    // increments for kernel modes
    array[K] simplex[kmax] Delta;

    // per-user deviation scale (logit scale)
    real<lower=1e-6> sigma;

    // non-centered per-user deviations
    array[V, K] vector[kmax] z_alpha;

    // mixture weights
    simplex[K] pi;
}

transformed parameters {
    // scaling factors for kernel modes
    vector[K] p = inv_logit(logit_p);

    // nonparametric kernel modes on logit scale
    array[K] vector[kmax] logit_mu;
    for (k in 1:K){
        vector[kmax] arg = p[k] * cumulative_sum(Delta[k]);
        logit_mu[k] = logit( fmin(fmax(arg, 1e-12), 1 - 1e-12) );
    }

    // per-user kernels on logit scale
    array[V] matrix[kmax, K] alpha; // transposed for computation
    for (i in 1:V){
        for (k in 1:K){
            alpha[i][,k] = logit_mu[k] + sigma * z_alpha[i, k];
        }
    }
}

model {
    // empirical Bayes priors
    logit_p ~ normal(LOGIT_P_LOCATION, LOGIT_P_SCALE);

    Delta ~ dirichlet(rep_vector(DELTA_CONCENTRATION, kmax));

    sigma ~ normal(0, SIGMA_SCALE)T[0,];

    pi ~ dirichlet(rep_vector(PI_CONCENTRATION, K));

    // component-specific noise for each node
    for (i in 1:V){
        for (k in 1:K){
            z_alpha[i, k] ~ normal(0, 1);
        }
    }

    // Marginalized mixture likelihood
    for (i in 1:V){
        vector[K] lps;
        for (k in 1:K){
            lps[k] = log(pi[k]);
            // Bernoulli logit likelihood weighted by expected counts M and N
            for (nu in 1:kmax){
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    lps[k] += M[i, nu] * log_inv_logit(alpha[i][nu, k]) + N[i, nu] * log1m_inv_logit(alpha[i][nu, k]);
                }
            }
        }

        target += log_sum_exp(lps);
    }
}

generated quantities {
    // kernel modes on probability scale
    array[K] vector[kmax] mu;
    for (k in 1:K){
        mu[k] = inv_logit( logit_mu[k] );
    }

    // per-user inferred kernels on probability scale
    array[V] matrix[K, kmax] c;
    for (i in 1:V){
        c[i] = inv_logit( alpha[i]' );
    }

    // posterior mixture assignment
    array[V, K] real<lower=0, upper=1> post_prob;
    array[V] int<lower=1, upper=K> z;

    // per-node marginal log-likelihood (for WAIC/LOO/model comparison)
    array[V] real log_lik;

    for (i in 1:V){
        vector[K] lps;
        for (k in 1:K){
            lps[k] = log(pi[k]);
            for (nu in 1:kmax){
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    lps[k] += M[i, nu] * log_inv_logit(alpha[i][nu, k]) 
                            + N[i, nu] * log1m_inv_logit(alpha[i][nu, k]);
                }
            }
        }

        real log_denom = log_sum_exp(lps);
        log_lik[i] = log_denom;
        
        for (k in 1:K){
            post_prob[i, k] = exp(lps[k] - log_denom);
        }

        z[i] = categorical_rng(to_vector(post_prob[i]));
    }
}
