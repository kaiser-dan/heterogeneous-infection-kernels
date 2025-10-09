/* Input data from user */
data {
	/* Population constants */
		// Number of individuals in the population
		int<lower=2> V;
	
		// Maximum number of neighbors (and hence number of infected neighbors)
		int<lower=1, upper=V> kmax;

    /* Node states */
        // Infection events, M_ij = number of infection events when node i has j infected neighbors
        array[V, kmax] int<lower=0, upper=kmax> M;

        // Infections failing to occur, N_ij = number of infection events that did *not* occur when node i has j infected neighbors
        array[V, kmax] int<lower=0, upper=kmax> N;

    /* Hyperparameters */
        // Kernel average dirichlet distribution location scaling
        real<lower=1> AVERAGE_SCALE;

        // Kernel deviation exponential distribution scale
        real<lower=0> DEVIATION_SCALE;
}

/* Model parameters we wish to infer */
// NOTE: Assumes a single group kernel (alpha)
parameters {
	// Kernel averages (as function of nu)
	simplex[kmax] mu;
	
	// Kernel variance (constant w.r.t. nu)
	real<lower=0> sigma;
	
	// Individual kernels
	matrix[V, kmax] alpha;
}
transformed parameters {
    // Logit c = alpha
	matrix<lower=0, upper=1>[V, kmax] c;
    for (i in 1:V) {
        for (nu in 1:kmax) {
            c[i][nu] = inv_logit(alpha[i][nu]);
        }
    }
}

/* Building posterior distribution */
model {
    /* Parameter priors */
        // Kernel average
        vector[kmax] dirichlet_position_vec;
        /*
        By sampling from a Dirichlet prior with increasing shape parameter components,
        we enforce a convexity condition on mu with respect to index.
        This is slightly different from JGY's notes,
        where mu is a partial sum of a Dirichlet r.v.
        That formulation would force mu[kmax] = 1;
        instead, the below form only forces mu[k] > mu[k'] when k' < k.
        */
        for (nu in 1:kmax) {
            dirichlet_position_vec[nu] = AVERAGE_SCALE*nu;
        }
        mu ~ dirichlet(dirichlet_position_vec);
        
        // Kernel variance
        sigma ~ exponential(DEVIATION_SCALE); 

        // Individual kernels
        for (i in 1:V) {
            alpha[i] ~ normal(mu, sigma); // NOTE: c is defined in transformed parameters block
        }
        
    /* Likelihood given data */
        // (log) likelihood contributions from infection events
        target += to_matrix(M) .* log(c);

        // (log) likelihood contributions from failed infection events
        target += to_matrix(N) .* log(1-c);
}