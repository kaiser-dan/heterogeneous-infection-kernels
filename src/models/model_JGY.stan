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

	/* Hyperparameters */
        // Kernel averages (before sum)
        array[K] simplex[kmax] delta; 
}
transformed parameters {
    /* Kernel parameters */
        // Mix kernel modes
        matrix[V, kmax] alpha;
        for (i in 1:V) {
            for (nu in 1:kmax) {
                alpha[i][nu] = 0;
                // Mixing kernel modes with weight vector pi[i]
                for (k in 1:K) {
                    // NOTE: Could swap some indices/transpose and compute as a dot product?
                    alpha[i][nu] += pi[i][k] * alpha_k[k][nu];
                }
            }
        }
            
        // Calculate kernel means as function of nu
        matrix[K, kmax] mu;
        for (k in 1:K) {
            mu[k] = rep_row_vector(0.0, kmax);
            for (nu in 1:kmax) {
                mu[k][nu] = sum(head(delta[k], nu));
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
        pi ~ dirichlet(rep_vector(1.0, K));  // NOTE: Should this be exported to its own hyperprior?

        // Kernel modes
        for (k in 1:K) {
            // Kernel variance
            sigma[k] ~ exponential(DEVIATION_SCALE[k]); 
        
            // Kernel modes
            alpha_k[k] ~ normal(logit(mu[k]), sigma[k]);
        }

    /* Likelihood given data */
        // (log) likelihood contributions from infection events
        target += to_matrix(M) .* log(c);

        // (log) likelihood contributions from failed infection events
        target += to_matrix(N) .* log(1-c);
}