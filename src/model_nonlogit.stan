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
}

/* Model parameters we wish to infer */
parameters {
    /* Population parameters */
        // Kernel mode mixtures for each individual
        array[V] simplex[K] pi; // NOTE: K is not first index in pi

    /* Contagion parameters */
        // Individual infection kernels
        array[K, V, kmax] real<lower=0, upper=1> c_;

    /* Hyperparameters */
        // Prior on succesful infection events
        array[K, kmax] real<lower=0> a;

        // Prior on failed infection events
        array[K, kmax] real<lower=0> b;

}
transformed parameters {
    array[V, kmax] real<lower=0, upper=1> c;
    for (i in 1:V) {
        for (nu in 1:kmax) {
            c[i][nu] = 0;
            for (k in 1:K) {
                c[i][nu] += pi[i][k] * c_[k][i][nu];
            }
        }
    }
}

/* Building posterior distribution */
model {
    /* Hyperpriors */
    for (nu in 1:kmax) {
        for (k in 1:K) {
            a[k][nu] ~ exponential(2);
            b[k][nu] ~ exponential(2);
        }
    }
    print(a);
    print(b);

    /* Priors */
        // Contagion kernels
        for (i in 1:V) {
            // Kernel mixing
            pi[i] ~ dirichlet(rep_vector(1.0, K));  // NOTE: Should this be exported to its own hyperprior?
            // Multinomial to force more separability?

            // Individual kernels
            for (nu in 1:kmax) {
                for (k in 1:K) {
                    c_[k][i][nu] ~ beta(M[i][nu] + a[k][nu], N[i][nu] + b[k][nu]);
                }
            }
        }

    /* Likelihood given data */
        // (log) likelihood contributions from infection events
        target += to_matrix(M) .* log(to_matrix(c));

        // (log) likelihood contributions from failed infection events
        target += to_matrix(N) .* log(1-to_matrix(c));
}
