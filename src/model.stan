/* Input data from user */
data {
	/* Population constants */
		// Number of individuals in the population
		int<lower=2> N;
	
		// Maximum number of neighbors (and hence number of infected neighbors)
		int<lower=1, upper=N> kmax;
}

/* Sampled from specified sampling distributions */
generated quantities {
	// Population average kernel
	array[kmax] real<lower=0, upper=1> mu;
	for (nu in 1:kmax) {
		mu[nu] = uniform_rng(0,1); // NOTE: Debugging prior
	}
	
	// Individual deviations
	array[N, kmax] real epsilon;
	for (i in 1:N) {
		for (nu in 1:kmax) {
			epsilon[i][nu] = normal_rng(0, 10); // NOTE: Debugging prior
		}
	}
	
	// Individual kernels
	array[N, kmax] real kernel;
	for (i in 1:N) {
		for (nu in 1:kmax) {
			kernel[i][nu] = inv_logit(mu[nu] + epsilon[i][nu]);
		}
	}
}
