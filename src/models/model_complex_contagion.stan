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

    // HYPERPARAMETER - prior location for logit_p
    real LOGIT_P_LOCATION;

    // HYPERPARAMETER - prior scale for logit_p
    real<lower=0> LOGIT_P_SCALE;
}

parameters {
    // Sigmoid parameters for each mode
    vector<lower=0, upper=1>[K] rho_l;
    vector<lower=0, upper=1>[K] rho_h;
    vector<lower=0>[K] w;
    vector<lower=0>[K] k0;

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
            // Sigmoid: rho_l + (rho_h - rho_l) / (1 + exp(-w*(nu - k0)))
            real p_nu = rho_l[k] + (rho_h[k] - rho_l[k]) / (1.0 + exp(-w[k] * (nu - k0[k])));
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
    // Sigmoid priors
    rho_l ~ beta(1, 10);
    rho_h ~ beta(1, 1); // allow for higher infection probabilities
    w ~ exponential(0.5);
    k0 ~ normal(kmax/2.0, kmax/2.0);

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
