import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import motpe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_name', default='WFG4', type=str)
    parser.add_argument('--num_objectives', default=2, type=int)
    parser.add_argument('--num_variables', default=9, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--num_max_evals', default=250, type=int)
    parser.add_argument('--num_initial_samples', default=98, type=int)
    parser.add_argument('--init_method', default='lhs', type=str)
    parser.add_argument('--num_candidates', default=24, type=int)
    parser.add_argument('--gamma', default=0.10, type=float)
    parser.add_argument('--seed', default=128, type=int)
    args = parser.parse_args()

    seed = args.seed
    num_initial_samples = args.num_initial_samples
    num_max_evals = args.num_max_evals
    num_objectives = args.num_objectives
    num_variables = args.num_variables
    k = args.k
    num_candidates = args.num_candidates
    init_method = args.init_method
    gamma = args.gamma
    base_configuration = {
        'num_objectives': num_objectives,
        'num_variables': num_variables,
        'k': k,
        'seed': seed}
    benchmark_name = args.benchmark_name
    f = motpe.WFG(benchmark_name, base_configuration)
    cs = f.make_cs(CS.ConfigurationSpace(seed=seed))
    problem = motpe.Problem(f, cs)
    solver = motpe.MOTPE(seed=seed)

    history = solver.solve(
        problem,
        {'num_initial_samples': num_initial_samples,
         'num_max_evals': num_max_evals,
         'init_method': init_method,
         'num_candidates': num_candidates,
         'gamma': gamma})
    if num_objectives == 2:
        fig = plt.figure(figsize=(8, 6))
        f1s = [fs['f1'] for fs in history['f']]
        f2s = [fs['f2'] for fs in history['f']]
        plt.scatter(f1s, f2s)
        plt.title(benchmark_name)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.grid()
        plt.show()
    else:
        print(history)
