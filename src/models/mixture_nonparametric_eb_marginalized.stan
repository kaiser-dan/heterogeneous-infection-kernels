/*
 * Non-parametric monotonic mixture of contagion kernels, EMPIRICAL BAYES priors.
 *
 * MARGINALIZED DEVIATIONS:
 *   The per-node, per-exposure logit-scale deviations
 *          c~_i(nu) ~ Normal( mu~_k(nu), sigma )   (independent across nu)
 *   are INTEGRATED OUT analytically-per-dimension via Gauss-Hermite quadrature,
 *   rather than sampled. Because the deviations are independent across nu, the
 *   node/class marginal likelihood factorizes into 1-D Gaussian integrals:
 *
 *     P(M_i, N_i | k) = prod_nu INT Binom(M,N | logit^{-1}(x)) Normal(x | mu~_k(nu), sigma) dx
 */

data {
    int<lower=2> V;                       // number of individuals
    int<lower=1, upper=V> kmax;           // max number of neighbours (exposure levels)
    int<lower=1, upper=V> K;              // number of mixture components

    array[V, kmax] real<lower=0> M;       // expected infection events
    array[V, kmax] real<lower=0> N;       // expected non-infection events

    real LOGIT_P_LOCATION;                // empirical-Bayes prior mean for logit ceiling
    real<lower=0> LOGIT_P_SCALE;          // empirical-Bayes prior scale
    real<lower=0> DELTA_CONCENTRATION;    // Dirichlet concentration for increments
    real<lower=0> SIGMA_SCALE;            // HalfNormal scale for sigma
    real<lower=0> PI_CONCENTRATION;       // Dirichlet concentration for mixture weights

    // Gauss-Hermite quadrature (PHYSICISTS' version): nodes x_g and weights w_g such that
    //   INT exp(-t^2) f(t) dt  ~=  sum_g w_g f(x_g).
    // in numpy:  gh_node, gh_weight = numpy.polynomial.hermite.hermgauss(G)
    int<lower=1> G;
    vector[G] gh_node;
    vector[G] gh_weight;
}

transformed data {
    real sqrt2 = sqrt(2.0);

    // pack counts into matrices for row-vector access per node
    matrix[V, kmax] Mmat;
    matrix[V, kmax] Nmat;
    for (i in 1:V) {
        Mmat[i] = to_row_vector(M[i]);
        Nmat[i] = to_row_vector(N[i]);
    }

    // log of GH weights normalized for a standard Gaussian: log(w_g) - 0.5*log(pi)
    vector[G] gh_logw;
    for (g in 1:G)
        gh_logw[g] = log(gh_weight[g]) - 0.5 * log(pi());
}

parameters {
    ordered[K] logit_p;                       // component ceilings (logit), ordered
    array[K] simplex[kmax] Delta;             // monotone increments per component
    real<lower=1e-6> sigma;                   // per-node deviation scale (logit)
    simplex[K] pi;                            // mixture weights
}

transformed parameters {
    vector[K] p = inv_logit(logit_p);
    vector[K] log_pi = log(pi);

    // component logit-scale kernel modes Tilde mu_k(nu); columns = components
    matrix[kmax, K] logit_mu;
    for (k in 1:K) {
        vector[kmax] arg = p[k] * cumulative_sum(Delta[k]);
        logit_mu[, k] = logit(fmin(fmax(arg, 1e-12), 1 - 1e-12));
    }
}

model {
    real sd_scale = sqrt2 * sigma;

    // ---- priors ----
    logit_p ~ normal(LOGIT_P_LOCATION, LOGIT_P_SCALE);
    Delta   ~ dirichlet(rep_vector(DELTA_CONCENTRATION, kmax));
    pi      ~ dirichlet(rep_vector(PI_CONCENTRATION, K));
    sigma   ~ normal(0, SIGMA_SCALE);         // half-normal via the lower bound

    // ---- precompute integrand pieces at the quadrature points ----
    array[K, kmax] vector[G] A;   // log inv_logit(x_g)
    array[K, kmax] vector[G] B;   // log1m inv_logit(x_g)
    for (k in 1:K)
        for (nu in 1:kmax) {
            vector[G] xg = logit_mu[nu, k] + sd_scale * gh_node;   // integration points
            A[k, nu] = log_inv_logit(xg);
            B[k, nu] = log1m_inv_logit(xg);
        }

    // ---- marginal mixture likelihood ----
    for (i in 1:V) {
        vector[K] lps = log_pi;
        for (nu in 1:kmax) {
            real Mv = Mmat[i, nu];
            real Nv = Nmat[i, nu];
            if (Mv + Nv > 0)
                // log INT Binom(M,N | logit^{-1}(c)) Normal(c | Tilde mu_k(nu), sigma) dc
                for (k in 1:K)
                    lps[k] += log_sum_exp(gh_logw + Mv * A[k, nu] + Nv * B[k, nu]);
        }
        target += log_sum_exp(lps);
    }
}

generated quantities {
    // component kernels on probability scale (CV uses these + pi)
    array[K] vector[kmax] mu;
    for (k in 1:K)
        mu[k] = inv_logit(logit_mu[, k]);

    // per-node marginal log-likelihood of the fitted (sigma-integrated) model,
    // for PSIS-LOO / WAIC diagnostics on the TRAINING set
    array[V] real log_lik;
    {
        real sd_scale = sqrt2 * sigma;

        array[K, kmax] vector[G] A;
        array[K, kmax] vector[G] B;
        for (k in 1:K)
            for (nu in 1:kmax) {
                vector[G] xg = logit_mu[nu, k] + sd_scale * gh_node;
                A[k, nu] = log_inv_logit(xg);
                B[k, nu] = log1m_inv_logit(xg);
            }

        for (i in 1:V) {
            vector[K] lps = log_pi;
            for (k in 1:K) {
                real ll = 0;
                for (nu in 1:kmax)
                    ll += log_sum_exp(gh_logw
                                      + Mmat[i, nu] * A[k, nu]
                                      + Nmat[i, nu] * B[k, nu]);
                lps[k] += ll;
            }
            log_lik[i] = log_sum_exp(lps);
        }
    }
}
