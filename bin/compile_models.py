import os
import sys

import cmdstanpy as csp


if __name__ == '__main__':
    dir_models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "models")
    dir_build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")

    if len(sys.argv) > 1:
        stan_file_basename = sys.argv[1]
        model = csp.CmdStanModel(stan_file=os.path.join(dir_models, stan_file_basename), force_compile=True)
        filename = os.path.splitext(os.path.basename(stan_file_basename))[0]

        # NOTE: Assumed ran through justfile job or from root
        os.system(f"mv {os.path.join(dir_models, filename)} {os.path.join(dir_build)}")
    else:
        for stan_file_basename in os.listdir(dir_models):
            model = csp.CmdStanModel(stan_file=os.path.join(dir_models, stan_file_basename), force_compile=True)
            filename = os.path.splitext(os.path.basename(stan_file_basename))[0]

            # NOTE: Assumed ran through justfile job or from root
            os.system(f"mv {os.path.join(dir_models, filename)} {os.path.join(dir_build)}")
