/* Helper functions */
// functions {
//     real<lower=0, upper=1> psum(vector v, int pos) {
//         return 
//     }
// }

/* Input data from user */
data {
	/* Population constants */
		// Number of individuals in the population
		int<lower=2> V;
	
		// Maximum number of neighbors (and hence number of infected neighbors)
		int<lower=1, upper=V> kmax;

        // Number of kernel modes
        int<lower=1, upper=V> K;  // NOTE: Should this be a parameter? Inferred according to some information criterion.

    /* Node states */
        // Infection events, M_ij = number of infection events when node i has j infected neighbors
        array[V, kmax] int<lower=0> M;

        // Infections failing to occur, N_ij = number of infection events that did *not* occur when node i has j infected neighbors
        array[V, kmax] int<lower=0> N;

    /* Hyperparameters */
        // Kernel average dirichlet distribution location scaling
        vector<lower=0>[K] AVERAGE_SCALE;

        // Kernel deviation exponential distribution scale
        vector<lower=0>[K] DEVIATION_SCALE;
}

/* Model parameters we wish to infer */
parameters {
    /* Population parameters */
        // Kernel mode mixtures for each individual
        array[V] simplex[K] pi; // NOTE: K is not first index in pi

    /* Kernel parameters */
        // Variances (constant w.r.t. nu)
        array[K] real<lower=0> sigma;
        
        // Modes
        matrix[K, kmax] alpha_k;
        // alpha_k ~ mu(nu) + f(k);
        // alpha ~ mu(nu) + pi_1*f(1) + (1-pi_1)*f(2);

	/* Hyperparameters */
        // Kernel averages
        simplex[kmax] delta; 
}
transformed parameters {
    /* Kernel parameters */
        // Averages
        matrix[K, kmax] mu;
        for (k in 1:K) {
            for (nu in 1:kmax) {
                mu[k][nu] = 0;
                for (idx in 1:nu) {
                    mu[k][nu] += delta[idx];
                }
                mu[k][nu] = logit(mu[k][nu]);
            }
        }

        // Individual kernel intercepts
        matrix[V, kmax] alpha;
        for (i in 1:V) {
            for (nu in 1:kmax) {
                alpha[i][nu] = 0;
                // Mixing kernel modes with weight vector pi[i]
                // NOTE: Can swap some indices/transpose and compute as a dot product
                for (k in 1:K) {
                    alpha[i][nu] += pi[i][k] * alpha_k[k][nu];
                }
            }
        }

    /* Population parameters */
        // Individual kernels
        matrix[V, kmax] c = inv_logit(alpha);
        // for (i in 1:V) {
        //     for (nu in 1:kmax) {
        //         c[i][i] = inv_logit(alpha[i][nu]); // + REGRESSION;
        //     }
        // }
}

/* Building posterior distribution */
model {
    /* Hyperparameter priors */
        // Kernel averages
        delta ~ dirichlet(rep_vector(1.0, kmax));

    /* Parameter priors */
        // Kernel mixing
        for (i in 1:V) {
            pi[i] ~ dirichlet(rep_vector(1.0, K));  // NOTE: Should this be exported to its own hyperprior?
        }
        // Multinomial to force more separability?

        // Kernel modes
        for (k in 1:K) {
            // Kernel variance
            sigma[k] ~ exponential(DEVIATION_SCALE[k]); 
        
            // Kernel modes
            alpha_k[k] ~ normal(mu[k], sigma[k]);
        }

    /* Likelihood given data */
        // (log) likelihood contributions from infection events
        target += to_matrix(M) .* log(c);

        // (log) likelihood contributions from failed infection events
        target += to_matrix(N) .* log(1-c);
}