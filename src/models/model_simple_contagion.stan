data {
  int<lower=1> kmax;
  array[kmax] real<lower=0> M;
  array[kmax] real<lower=0> N;
}

parameters {
  real<lower=0, upper=1> rho;
}

transformed parameters {
  vector[kmax] c;
  for (nu in 1:kmax) {
    c[nu] = 1.0 - pow(1.0 - rho, nu);
  }
}

model {
  // Prior
  rho ~ beta(1, 100); // Low expectation for per-exposure infection probability
  
  // Likelihood
  for (nu in 1:kmax) {
    if (M[nu] > 0) target += M[nu] * log(fmax(c[nu], 1e-12));
    if (N[nu] > 0) target += N[nu] * log(fmax(1.0 - c[nu], 1e-12));
  }
}
