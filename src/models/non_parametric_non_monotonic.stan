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
    vector<lower=1e-6>[K] DEVIATION_SCALE;

    // HYPERPARAMETER - Dirichlet concentration for mixture components
    real<lower=0> CONCENTRATION;
}

transformed data {
}

parameters {
    // kernel mode means (non-monotonic - no ordering constraint)
    array[K] simplex[kmax] mu;

    // kernel mode variances (scalar per component)
    array[K] real<lower=1e-6> sigma;

    // for non-centered parameterizations of kernels (indexed by node AND component)
    array[V, K] vector[kmax] z_alpha;

    // global mixing weights over components
    simplex[K] pi;
}

transformed parameters {
    // kernel modes on logit scale (non-monotonic)
    array[K] vector[kmax] logit_mu;
    for (k in 1:K){
        logit_mu[k] = logit( fmin(fmax(to_vector(mu[k]), 1e-12), 1 - 1e-12) );
    }

    // kernels on logit scale (with component-specific noise)
    array[V] matrix[kmax, K] alpha; // transposed for computation
    for (i in 1:V){
        for (k in 1:K){
            alpha[i][,k] = logit_mu[k] + sigma[k] * z_alpha[i, k];
        }
    }
}

model {
    // kernel mode means
    for (k in 1:K){
        mu[k] ~ dirichlet(rep_vector(CONCENTRATION, kmax));
    }

    // kernel mode variances
    for (k in 1:K){
        sigma[k] ~ exponential(DEVIATION_SCALE[k]);
    }

    // global mixture weights
    pi ~ dirichlet(rep_vector(1.0, K));

    // component-specific noise for each node
    for (i in 1:V){
        for (k in 1:K){
            z_alpha[i, k] ~ normal(0, 1);
        }
    }

    // marginalized mixture likelihood
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
    array[K] vector[kmax] mu_prob;
    for (k in 1:K){
        mu_prob[k] = to_vector(mu[k]);
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

