/*
 * Mixture Simple Contagion Model
 *
 * Kernel: geometric (independent cascade)
 *   P(adopt | nu exposures) = 1 - (1 - rho)^nu
 *
 * Shared mixture scaffolding: K components, per-user heterogeneity via
 * non-centered logit-scale deviations, all priors passed as DATA.
 */

data {
    int<lower=2> V;                          // number of individuals
    int<lower=1, upper=V> kmax;              // max exposure level
    int<lower=1, upper=V> K;                 // number of mixture components

    array[V, kmax] real<lower=0> M;          // adoption counts per (user, exposure)
    array[V, kmax] real<lower=0> N;          // non-adoption counts per (user, exposure)

    // --- Prior hyperparameters (all configurable, nothing hardcoded) ---
    real LOGIT_RHO_LOCATION;                 // logit-Normal prior on per-contact infection prob
    real<lower=0> LOGIT_RHO_SCALE;

    real<lower=0> SIGMA_SCALE;               // HalfNormal(0, SIGMA_SCALE) prior on sigma
    real<lower=0> PI_CONCENTRATION;          // Dirichlet(PI_CONCENTRATION) prior on mixture weights
}

parameters {
    ordered[K] logit_rho;                    // per-contact infection probability (logit scale)
    real<lower=1e-6> sigma;                  // per-user deviation scale (logit scale)
    array[V, K] vector[kmax] z_alpha;        // non-centered per-user deviations
    simplex[K] pi;                           // mixture weights
}

transformed parameters {
    vector<lower=0, upper=1>[K] rho = inv_logit(logit_rho);

    // Kernel modes on logit scale: geometric kernel
    array[K] vector[kmax] logit_mu;
    for (k in 1:K) {
        for (nu in 1:kmax) {
            real p_nu = 1.0 - pow(1.0 - rho[k], nu);
            logit_mu[k][nu] = logit(fmin(fmax(p_nu, 1e-12), 1 - 1e-12));
        }
    }

    // Per-user kernels on logit scale
    array[V] matrix[kmax, K] alpha;
    for (i in 1:V) {
        for (k in 1:K) {
            alpha[i][,k] = logit_mu[k] + sigma * z_alpha[i, k];
        }
    }
}

model {
    // Priors
    logit_rho ~ normal(LOGIT_RHO_LOCATION, LOGIT_RHO_SCALE);
    sigma ~ normal(0, SIGMA_SCALE)T[0,];
    pi ~ dirichlet(rep_vector(PI_CONCENTRATION, K));

    // Per-user noise
    for (i in 1:V) {
        for (k in 1:K) {
            z_alpha[i, k] ~ normal(0, 1);
        }
    }

    // Marginalized mixture likelihood
    for (i in 1:V) {
        vector[K] lps;
        for (k in 1:K) {
            lps[k] = log(pi[k]);
            for (nu in 1:kmax) {
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    lps[k] += M[i, nu] * log_inv_logit(alpha[i][nu, k])
                            + N[i, nu] * log1m_inv_logit(alpha[i][nu, k]);
                }
            }
        }
        target += log_sum_exp(lps);
    }
}

generated quantities {
    // Kernel modes on probability scale
    array[K] vector[kmax] mu;
    for (k in 1:K) {
        mu[k] = inv_logit(logit_mu[k]);
    }

    // Per-user inferred kernels
    array[V] matrix[K, kmax] c;
    for (i in 1:V) {
        c[i] = inv_logit(alpha[i]');
    }

    // Posterior mixture assignment
    array[V, K] real<lower=0, upper=1> post_prob;
    array[V] int<lower=1, upper=K> z;

    // Log-likelihood for WAIC/LOO
    vector[V] log_lik;

    for (i in 1:V) {
        vector[K] lps;
        for (k in 1:K) {
            lps[k] = log(pi[k]);
            for (nu in 1:kmax) {
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    lps[k] += M[i, nu] * log_inv_logit(alpha[i][nu, k])
                            + N[i, nu] * log1m_inv_logit(alpha[i][nu, k]);
                }
            }
        }

        real log_denom = log_sum_exp(lps);
        log_lik[i] = log_denom;

        for (k in 1:K) {
            post_prob[i, k] = exp(lps[k] - log_denom);
        }
        z[i] = categorical_rng(to_vector(post_prob[i]));
    }
}
