/*
 * Non-parametric monotonic mixture model with EMPIRICAL BAYES priors (optimized)
 *
 * logit_p prior centered at empirical scale from data.
 */

data {
    int<lower=2> V;                          // number of individuals
    int<lower=1, upper=V> kmax;              // max number of neighbors
    int<lower=1, upper=V> K;                 // number of kernel modes

    array[V, kmax] real<lower=0> M;          // expected infections at j exposures
    array[V, kmax] real<lower=0> N;          // expected non-infections at j exposures

    real LOGIT_P_LOCATION;                   // empirical Bayes prior location
    real<lower=0> LOGIT_P_SCALE;             // empirical Bayes prior scale
    real<lower=0> DELTA_CONCENTRATION;       // Dirichlet conc. for increments
    real<lower=1e-6> SIGMA_SCALE;
    real<lower=0> PI_CONCENTRATION;
}

transformed data {
    array[V] row_vector[kmax] Mv;
    array[V] row_vector[kmax] Nv;
    for (i in 1:V) {
        Mv[i] = to_row_vector(M[i]);
        Nv[i] = to_row_vector(N[i]);
    }
}

parameters {
    ordered[K] logit_p;                      // kernel-mode scaling (logit)
    array[K] simplex[kmax] Delta;            // increments for kernel modes
    real<lower=1e-6> sigma;                  // per-user deviation scale
    array[V, K] vector[kmax] z_alpha;        // non-centered per-user, per-component, per-exposure deviations
    simplex[K] pi;                           // mixture weights
}

transformed parameters {
    vector[K] p = inv_logit(logit_p);

    // nonparametric monotonic kernel modes on logit scale
    array[K] vector[kmax] logit_mu;
    for (k in 1:K) {
        vector[kmax] arg = p[k] * cumulative_sum(Delta[k]);
        logit_mu[k] = logit(fmin(fmax(arg, 1e-12), 1 - 1e-12));
    }
}

model {
    // priors
    logit_p ~ normal(LOGIT_P_LOCATION, LOGIT_P_SCALE);
    Delta ~ dirichlet(rep_vector(DELTA_CONCENTRATION, kmax));
    sigma ~ normal(0, SIGMA_SCALE);
    pi ~ dirichlet(rep_vector(PI_CONCENTRATION, K));

    for (i in 1:V) {
        for (k in 1:K) {
            z_alpha[i, k] ~ std_normal();
        }
    }

    // marginalized mixture likelihood (vectorized)
    vector[K] log_pi = log(pi);
    for (i in 1:V) {
        matrix[kmax, K] A; // per-user kernel on logit scale
        for (k in 1:K) {
            A[, k] = logit_mu[k] + sigma * z_alpha[i, k];
        }

        row_vector[K] ll = Mv[i] * log_inv_logit(A)
                         + Nv[i] * log1m_inv_logit(A);

        target += log_sum_exp(log_pi + ll');
    }
}

generated quantities {
    array[K] vector[kmax] mu; // kernel modes (probability scale)
    for (k in 1:K) {
        mu[k] = inv_logit(logit_mu[k]);
    }

    array[V, K] real<lower=0, upper=1> post_prob; // posterior over mixture indicator
    array[V] int<lower=1, upper=K> z;             // mixture indicator
    array[V] real log_lik;                        // log-likelihood
    array[V] matrix[K, kmax] c;                   // per-user kernels (probability scale)

    vector[K] log_pi = log(pi);
    for (i in 1:V) {
        matrix[kmax, K] A;
        for (k in 1:K) {
            A[, k] = logit_mu[k] + sigma * z_alpha[i, k];
        }

        row_vector[K] ll = Mv[i] * log_inv_logit(A)
                         + Nv[i] * log1m_inv_logit(A);
        vector[K] lps = log_pi + ll';

        real log_denom = log_sum_exp(lps);
        log_lik[i] = log_denom;

        for (k in 1:K) {
            post_prob[i, k] = exp(lps[k] - log_denom);
        }
        z[i] = categorical_rng(to_vector(post_prob[i]));

        c[i] = inv_logit(A');
    }
}
