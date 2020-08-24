# Multiobjective Tree-structured Parzen Estimator (MOTPE)

## Dependencies

Please install the following Python packages.

- [ConfigSpace](https://automl.github.io/ConfigSpace/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [optproblems](https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/)
- [pandas](https://pandas.pydata.org/)
- [pyDOE2](https://github.com/clicumu/pyDOE2)
- [pygmo](https://esa.github.io/pygmo/)
- [scipy](https://www.scipy.org/)


## Usage

Run MOTPE with the default settings on WFG4.

```sh
python main.py
```

Several commandline options are available.

```sh
python main.py --benchmark_name WFG4 --num_objectives 2 --num_variables 9 --k 1 --num_max_evals 250 --num_initial_samples 98 --init_method lhs --num_candidates 24 --gamma 0.10 --seed 128
```

## Code Contributors
[Yoshihiko Ozaki](https://github.com/y0z)
[Shuhei Watanabe](https://github.com/nabenabe0928)
