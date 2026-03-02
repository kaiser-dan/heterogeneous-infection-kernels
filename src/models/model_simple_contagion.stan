data {
    // number of individuals in population
    int<lower=2> V;

    // max number of neighbors
    int<lower=1, upper=V> kmax;

    // number of kernel modes
    int<lower=1, upper=V> K;

    // M[i,j] = number of infections (expected) when node i has j infected neighbors
    array[V, kmax] real<lower=0> M;

    // N[i,j] = number of infections failing to occur (expected) when node i has j infected neighbors
    array[V, kmax] real<lower=0> N;

    // HYPERPARAMETER - exponential distribution scale for kernel mode deviations
    real<lower=1e-6> DEVIATION_SCALE;

    // HYPERPARAMETER - prior location for logit_p (not used directly here but kept for consistency)
    real LOGIT_P_LOCATION;

    // HYPERPARAMETER - prior scale for logit_p
    real<lower=0> LOGIT_P_SCALE;
}

parameters {
    // geometric infection probability parameters for each mode
    vector<lower=0, upper=1>[K] rho;

    // std for kernel mode deviations
    real<lower=1e-6> sigma;

    // for non-centered parameterizations of kernels(indexed by node AND component)
    array[V, K] vector[kmax] z_alpha;

    // global mixture weights over components
    simplex[K] pi;
}

transformed parameters {
    // kernel modes on logit scale
    array[K] vector[kmax] logit_mu;
    for (k in 1:K){
        for (nu in 1:kmax){
            // p(nu) = 1 - (1 - rho)^nu
            real p_nu = 1.0 - pow(1.0 - rho[k], nu);
            logit_mu[k][nu] = logit( fmin(fmax(p_nu, 1e-12), 1 - 1e-12) );
        }
    }

    // kernels on logit scale 
    array[V] matrix[kmax, K] alpha; // transposed for computation
    for (i in 1:V){
        for (k in 1:K){
            alpha[i][,k] = logit_mu[k] + sigma * z_alpha[i, k];
        }
    }
}

model {
    // Parametric prior
    rho ~ beta(1, 10); // slightly more informative than 1,100 but still supports small values

    sigma ~ exponential(DEVIATION_SCALE);

    // global mixture weights
    pi ~ dirichlet(rep_vector(1.0, K));

    // component-specific noise for each node
    for (i in 1:V){
        for (k in 1:K){
            z_alpha[i, k] ~ normal(0, 1);
        }
    }

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
    // kernel modes
    array[K] vector[kmax] mu;
    for (k in 1:K){
        mu[k] = inv_logit( logit_mu[k] );
    }

    // inferred kernels
    array[V] matrix[K, kmax] c;
    for (i in 1:V){
        c[i] = inv_logit( alpha[i]' );
    }

    // posterior over mixture indicator
    array[V, K] real<lower=0, upper=1> post_prob;

    // mixture indicator
    array[V] int<lower=1, upper=K> z;

    // log_lik for WAIC/LOO
    vector[V] log_lik;

    for (i in 1:V){
        vector[K] lps;
        for (k in 1:K){
            lps[k] = log(pi[k]);
            for (nu in 1:kmax){
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    lps[k] += M[i, nu] * log_inv_logit(alpha[i][nu, k]) + N[i, nu] * log1m_inv_logit(alpha[i][nu, k]);
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
