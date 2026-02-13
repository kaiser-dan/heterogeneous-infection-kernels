data {
	// number of individuals in population
    int<lower=1> V; 

    // max number of neighbors
	int<lower=1> kmax;

    // number of kernel modes
    int<lower=1> K;

    // M[i,j] = number of infections when node i has j infected neighbors
    array[V, kmax] int<lower=0> M;

    // N[i,j] = number of infections failing to occur when node i has j infected neighbors
    array[V, kmax] int<lower=0> N;

    // HYPERPARAMETER - exponential distribution scale for kernel mode deviations
    vector<lower=1e-6>[K] DEVIATION_SCALE;
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
    // mixing proportions
    array[V] simplex[K] pi;

    // scaling factors for kernel modes on logit scale
    ordered[K] logit_p;

    // increments for kernel modes
    array[K] simplex[kmax] Delta;

    // stds for kernel mode deviations
    array[K] real<lower=1e-6> sigma;

    // for non-centered parameterization of kernels
    array[K] vector[kmax] z_alpha;
}

transformed parameters {
    // scaling factors for kernel modes
    vector[K] p = inv_logit(logit_p);

    // kernel modes on logit scale
    matrix[kmax, K] logit_mu; // transposed for easier computation
    for (k in 1:K){
        vector[kmax] arg = p[k] * cumulative_sum(Delta[k]);
        logit_mu[, k] = logit( fmin(fmax( arg , 1e-12), 1 - 1e-12) );
    }

    // kernels on logit scale
    matrix[kmax, K] alpha_k; // transposed for easier computation
    for (k in 1:K){
        alpha_k[, k] = logit_mu[, k] + sigma[k] * z_alpha[k];
    }

    // individual kernel mixtures on logit scale
    matrix[kmax, V] alpha; // transposed for easier computation
    for (i in 1:V){
        alpha[, i] = alpha_k * pi[i];
    }
}

model {
    pi ~ dirichlet(rep_vector(1.0, K));
    
    logit_p ~ normal(0, 2);
    
    Delta ~ dirichlet(rep_vector(1.0, kmax));

    for (k in 1:K){
        sigma[k] ~ exponential(DEVIATION_SCALE[k]);
        z_alpha[k] ~ normal(0,1);
    }

    for (i in 1:V){
        target += binomial_logit_lupmf(M[i, ] | events[i, ], alpha[, i]);
    }
}

generated quantities {
    // kernel modes
    matrix[K, kmax] mu = inv_logit(logit_mu');

    // individual kernel mixtures
    matrix [V, kmax] c = inv_logit(alpha');
}