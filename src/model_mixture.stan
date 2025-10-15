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
        vector<lower=1>[K] AVERAGE_SCALE;

        // Kernel deviation exponential distribution scale
        vector<lower=0>[K] DEVIATION_SCALE;

        // Kernel mixture strength
        // array[K] simplex[kmax] MIXTURE_STRENGTHS;
}

/* Model parameters we wish to infer */
parameters {
	// Kernel averages (as function of nu)
	array[K] simplex[kmax] mu;
	
	// Kernel variances (constant w.r.t. nu)
	array[K] real<lower=0> sigma;
	
	// Kernel modes
	array[K] matrix<lower=0, upper=1>[V, kmax] alpha_k;

    // Kernel mode mixtures (for each individual)
    array[V] simplex[K] pi;
}
transformed parameters {
	// Individual kernel intercepts
	matrix[V, kmax] alpha;
    for (i in 1:V) {
        for (nu in 1:kmax) {
            alpha[i][nu] = 0;
            // >>> DEBUG >>>
            print(">>>");
            print("i ", i, " nu ", nu);
            // <<< DEBUG <<<

            // Mixing kernel modes with weight vector pi[i]
            for (k in 1:K) {
                // >>> DEBUG >>>
                print("k ", k);
                print("pi[i][k] ", pi[i][k]);
                print("alpha_k[k][i][nu] ", alpha_k[k][i][nu]);
                // <<< DEBUG <<<
                alpha[i][nu] += pi[i][k] * alpha_k[k][i][nu];
            }
            print("alpha[i][nu] ", alpha[i][nu]);
            // >>> DEBUG >>>
            print("<<<");
            // <<< DEBUG <<<
        }
    }

    // Individual kernels
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
        for (k in 1:K) {
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
                dirichlet_position_vec[nu] = AVERAGE_SCALE[k]*nu;
            }
            mu[k] ~ dirichlet(dirichlet_position_vec);
        
            // Kernel variance
            sigma[k] ~ exponential(DEVIATION_SCALE[k]); 
        
            // Kernel modes
            for (i in 1:V) {
                for (nu in 1:kmax) {
                    alpha_k[k][i][nu] ~ normal(mu[k][nu], sigma[k]);
                }
            }
        }

        // Kernel mixing
        for (i in 1:V) {
            pi[i] ~ dirichlet(rep_vector(1, K));  // NOTE: Should this be exported to its own hyperprior?
        }

    /* Likelihood given data */
        // (log) likelihood contributions from infection events
        target += to_matrix(M) .* log(c);

        // (log) likelihood contributions from failed infection events
        target += to_matrix(N) .* log(1-c);
}