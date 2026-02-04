data {
	// number of individuals in population
    int<lower=2> V;

    // max number of neighbors
	int<lower=1, upper=V> kmax;

    // number of kernel modes
    int<lower=1, upper=V> K;

    // M[i,j] = number of infections when node i has j infected neighbors
    array[V, kmax] int<lower=0> M;

    // N[i,j] = number of infections failing to occur when node i has j infected neighbors
    array[V, kmax] int<lower=0> N;

    // HYPERPARAMETER - exponential distribution scale for kernel mode deviations
    real<lower=1e-6> DEVIATION_SCALE;
}

transformed data {
    // events[i,j] = number of times node i has j infected neighbors
    array[V, kmax] int<lower=0> events;
    for (i in 1:V){
        for (nu in 1:kmax){
            events[i, nu] = M[i, nu] + N[i, nu];
        }
    }
}

parameters {
    // kernel modes (non-monotonic)
    array[K] simplex[kmax] mu;

    // std for kernel mode deviations
    real<lower=1e-6> sigma;

    // for non-centered parameterizations of kernels
    array[V] vector[kmax] z_alpha;

    // per-node mixing weights over components
    array[V] simplex[K] pi;
}

transformed parameters {
    // kernel modes on logit scale
    array[K] vector[kmax] logit_mu;
    for (k in 1:K){
        logit_mu[k] = logit( fmin(fmax(mu[k], 1e-12), 1 - 1e-12) );
    }

    // kernels on logit scale
    array[V] matrix[kmax, K] alpha; // transposed for computation
    for (i in 1:V){
        for (k in 1:K){
            alpha[i][,k] = logit_mu[k] + sigma * z_alpha[i];
        }
    }
}

model {
    // Prior on kernel modes
    for (k in 1:K){
        mu[k] ~ dirichlet(rep_vector(1.0, kmax));
    }

    sigma ~ exponential(DEVIATION_SCALE);

    for (i in 1:V){
        pi[i] ~ dirichlet(rep_vector(1.0, K));
        z_alpha[i] ~ normal(0, 1);
    }

    for (i in 1:V){
        vector[K] lps;
        for (k in 1:K){
            lps[k] = log(pi[i][k]) + binomial_logit_lupmf(M[i,] | events[i,], alpha[i][,k]);
        }

        target += log_sum_exp(lps);
    }
}

generated quantities {
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
            lps[k] = log(pi[i][k]) + dot_product(to_vector(M[i,]), log(c[i][k,])) + dot_product(to_vector(N[i,]), log(1 - c[i][k,]));
        }

        real log_denom = log_sum_exp(lps);
        for (k in 1:K){
            post_prob[i, k] = exp(lps[k] - log_denom);
        }

        z[i] = categorical_rng(to_vector(post_prob[i]));
    }
}
