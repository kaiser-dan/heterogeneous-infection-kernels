/* Input data from user */
data {
	/* Population constants */
		// Number of individuals in the population
		int<lower=2> V;
	
		// Maximum number of neighbors (and hence number of infected neighbors)
		int<lower=1, upper=V> kmax;

    /* Hyperparameters */
        // Kernel average dirichlet distribution location scaling
        real<lower=1> AVERAGE_SCALE;

        // Kernel deviation exponential distribution scale
        real<lower=0> DEVIATION_SCALE;
}

/* Sampled from specified sampling distributions */
generated quantities {
	// Kernel averages (as function of nu)
	vector[kmax] dirichlet_position_vec;
	for (nu in 1:kmax) {
		dirichlet_position_vec[nu] = AVERAGE_SCALE*nu;
	}

	simplex[kmax] mu;
	mu = dirichlet_rng(dirichlet_position_vec);

	// Kernel variance (constant w.r.t. nu)
	real<lower=0> sigma;
	sigma = exponential_rng(DEVIATION_SCALE);
	
	// Individual kernels
	matrix[V, kmax] alpha;
	matrix<lower=0, upper=1> [V, kmax] c;
	for (i in 1:V) {
		alpha[i] = to_row_vector(normal_rng(mu, sigma));
		c[i] = inv_logit(alpha[i]);
	}
}
