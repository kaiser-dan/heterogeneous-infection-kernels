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

    // HYPERPARAMETER - Dirichlet concentration for increments
    real<lower=0> CONCENTRATION;

    // HYPERPARAMETER - prior location for logit_p (empirical Bayes)
    real LOGIT_P_LOCATION;

    // HYPERPARAMETER - prior scale for logit_p
    real<lower=0> LOGIT_P_SCALE;
}

transformed data {
}

parameters {
    // scaling factors for kernel modes on logit scale
    ordered[K] logit_p;

    // increments for kernel modes
    array[K] simplex[kmax] Delta;

    // std for kernel mode deviations
    real<lower=1e-6> sigma;

    // for non-centered parameterizations of kernels(indexed by node AND component)
    array[V, K] vector[kmax] z_alpha;

    // global mixture weights over components
    simplex[K] pi;
}

transformed parameters {
    // scaling factors for kernel modes
    vector[K] p = inv_logit(logit_p);

    // kernel modes on logit scale
    array[K] vector[kmax] logit_mu;
    for (k in 1:K){
        vector[kmax] arg = p[k] * cumulative_sum(Delta[k]);
        logit_mu[k] = logit( fmin(fmax(arg, 1e-12), 1 - 1e-12) );
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
    logit_p ~ normal(LOGIT_P_LOCATION, LOGIT_P_SCALE);

    Delta ~ dirichlet(rep_vector(CONCENTRATION, kmax));

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

    for (i in 1:V){
        vector[K] lps;
        for (k in 1:K){
            lps[k] = log(pi[k]);
            for (nu in 1:kmax){
                if (M[i, nu] > 0 || N[i, nu] > 0) {
                    lps[k] += M[i, nu] * log(c[i][k, nu]) + N[i, nu] * log(1 - c[i][k, nu]);
                }
            }
        }

        real log_denom = log_sum_exp(lps);
        for (k in 1:K){
            post_prob[i, k] = exp(lps[k] - log_denom);
        }

        z[i] = categorical_rng(to_vector(post_prob[i]));
    }
}

