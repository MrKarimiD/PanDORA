import numpy as np
import argparse

def calculate_exposures(min_exposure, max_exposure, nb_exposures):
    return 2 ** np.linspace(np.log2(min_exposure), np.log2(max_exposure), nb_exposures)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--min", type=float, required=True, default=0.00004)
    argparser.add_argument("--max", type=float, required=True, default=0.004)
    args = argparser.parse_args()

    min_exposure = args.min
    max_exposure = args.max
    nb_exposures = 11       # min and max exposures included

    exposures = calculate_exposures(min_exposure, max_exposure, nb_exposures)
    print("Calculated exposures:")
    for exposure in reversed(exposures):
        denominator = round(1 / exposure)
        print(f"1/{denominator}")