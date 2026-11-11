/*
 * Non-parametric monotonic mixture model with EMPIRICAL BAYES priors, on observed cells
 *
 * Same model as mixture_nonparametric_eb.stan. Per-node deviations are sampled only for
 * (node, nu) cells with data: the others are independent N(0, 1) and never touch the likelihood.
 * For large empirical networks (e.g. WeChat) where most cells are empty.
 */

data {
	// number of individuals in population
    int<lower=2> V;

    // max number of neighbors
	int<lower=1> kmax;

    // number of kernel modes (mixture components)
    int<lower=1, upper=V> K;

    // observed (node, nu) cells
    int<lower=1> n_cells;
    array[n_cells] int<lower=1, upper=V> cell_node;
    array[n_cells] int<lower=1, upper=kmax> cell_nu;
    array[n_cells] real<lower=0> cell_M;
    array[n_cells] real<lower=0> cell_N;

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

    // non-centered per-cell deviations
    array[n_cells] vector[K] z_alpha;

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
}

model {
    // empirical Bayes priors
    logit_p ~ normal(LOGIT_P_LOCATION, LOGIT_P_SCALE);

    Delta ~ dirichlet(rep_vector(DELTA_CONCENTRATION, kmax));

    sigma ~ normal(0, SIGMA_SCALE)T[0,];

    pi ~ dirichlet(rep_vector(PI_CONCENTRATION, K));

    for (j in 1:n_cells){
        z_alpha[j] ~ normal(0, 1);
    }

    // per-node log-likelihood under each kernel mode
    array[V] vector[K] ll = rep_array(rep_vector(0, K), V);
    for (j in 1:n_cells){
        for (k in 1:K){
            real alpha = logit_mu[k][cell_nu[j]] + sigma * z_alpha[j][k];
            ll[cell_node[j]][k] += cell_M[j] * log_inv_logit(alpha) + cell_N[j] * log1m_inv_logit(alpha);
        }
    }

    // Marginalized mixture likelihood
    for (i in 1:V){
        target += log_sum_exp(log(pi) + ll[i]);
    }
}

generated quantities {
    // kernel modes on probability scale
    array[K] vector[kmax] mu;
    for (k in 1:K){
        mu[k] = inv_logit( logit_mu[k] );
    }

    // posterior mixture assignment
    array[V, K] real<lower=0, upper=1> post_prob;
    {
        array[V] vector[K] ll = rep_array(rep_vector(0, K), V);
        for (j in 1:n_cells){
            for (k in 1:K){
                real alpha = logit_mu[k][cell_nu[j]] + sigma * z_alpha[j][k];
                ll[cell_node[j]][k] += cell_M[j] * log_inv_logit(alpha) + cell_N[j] * log1m_inv_logit(alpha);
            }
        }
        for (i in 1:V){
            post_prob[i] = to_array_1d(softmax(log(pi) + ll[i]));
        }
    }
}
