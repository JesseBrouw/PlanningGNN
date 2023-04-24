import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

def ratio_OverThreshold(y, predictions):
    y = [y[i:i + 100] for i in range(0, len(y), 100)]
    predictions = [predictions[i:i + 100] for i in range(0, len(predictions), 100)]

    over_threshold = []
    for problem_y, problem_pred in zip(y, predictions):
        val1 = problem_y.index(1)
        val2 = problem_pred.index(1)
        over = 1 if val1 <= val2 else 0
        over_threshold.append(over)

    return sum(over_threshold)/len(over_threshold)

def visualize_class_results(y, predictions, domains, filename):
    y = [y[i:i + 100] for i in range(0, len(y), 100)]
    predictions = [predictions[i:i + 100] for i in range(0, len(predictions), 100)]

    domain_y = defaultdict(list)
    domain_pred = defaultdict(list)
    maximum = 0
    for i, (problem_y, problem_pred) in enumerate(zip(y, predictions)):
        domain_y[domains[i*100]].append(problem_y.index(1))
        domain_pred[domains[i*100]].append(problem_pred.index(1))
        maximum = max(domain_y[domains[i*100]]) if max(domain_y[domains[i*100]]) > maximum else maximum
        maximum = max(domain_pred[domains[i*100]]) if max(domain_pred[domains[i*100]]) > maximum else maximum


    fig, ax = plt.subplots()
    ax.plot([0,maximum],[0,maximum], '--')
    ax.set_ylabel('Predicted threshold')
    ax.set_xlabel('True threshold')
    ax.set_title(f'Threshold predictions {filename.split(".")[0]}.')
    for key in domain_y.keys():
        ax.scatter(domain_y[key], domain_pred[key], label=key)
    ax.legend()
    path = os.path.join(os.pardir(), 'Results', 'Results_classification', filename)
    fig.savefig(path, dpi=300)

def visualize_regr_results(y, predictions, domains, filename):

    domain_y = defaultdict(list)
    domain_pred = defaultdict(list)
    maximum = 0
    for i, (problem_y, problem_pred) in enumerate(zip(y, predictions)):
        domain_y[domains[i]].append(problem_y)
        domain_pred[domains[i]].append(problem_pred)
        maximum = max(domain_y[domains[i]]) if max(domain_y[domains[i]]) > maximum else maximum
        maximum = max(domain_pred[domains[i]]) if max(domain_pred[domains[i]]) > maximum else maximum

    fig, ax = plt.subplots()
    ax.plot([0,maximum],[0,maximum], '--')
    ax.set_ylabel('Predicted threshold')
    ax.set_xlabel('True threshold')
    ax.set_title(f'Threshold predictions {filename.split(".")[0]}.')
    for key in domain_y.keys():
        ax.scatter(domain_y[key], domain_pred[key], label=key)
    ax.legend()
    path = os.path.join(os.pardir, 'Results', filename)
    fig.savefig(path, dpi=300)

def MSE(y, predictions, domains):
    y = [y[i:i + 100] for i in range(0, len(y), 100)]
    predictions = [predictions[i:i + 100] for i in range(0, len(predictions), 100)]

    results = defaultdict(list)
    for i, (problem_y, problem_pred) in enumerate(zip(y, predictions)):
        val1 = problem_y.index(1)
        val2 = problem_pred.index(1)
        results[domains[i*100]].append((val1 - val2)**2)

    for key, value in results.items():
        results[key] = sum(value)/len(value)
        print(f'MSE {key} : {results[key]}')
    
    results['total'] = sum(results.values())/len(results.values())
    print(f'Total MSE : {results["total"]}')

    return results


def MAE(y, predictions, domains):
    y = [y[i:i + 100] for i in range(0, len(y), 100)]
    predictions = [predictions[i:i + 100] for i in range(0, len(predictions), 100)]

    results = defaultdict(list)
    for i, (problem_y, problem_pred) in enumerate(zip(y, predictions)):
        val1 = problem_y.index(1)
        val2 = problem_pred.index(1)
        results[domains[i*100]].append(abs(val1 - val2))

    for key, value in results.items():
        results[key] = sum(value)/len(value)
        print(f'MAE {key} : {results[key]}')
    
    results['total'] = sum(results.values())/len(results.values())
    print(f'Total MAE : {results["total"]}')
    
    return results

def runtime_graph(y, predictions, df, filename):
    problems = df.problem.unique()
    problem_y = defaultdict(list)
    problem_pred = defaultdict(list)
    counter = 0
    for problem in problems:
        subset = df[df.problem == problem]
        for i, row in subset.iterrows():
            problem_y[problem].append((row.horizon, y[counter]))
            problem_pred[problem].append((row.horizon, predictions[counter]))
            counter += 1
    
    for i, key in enumerate(problem_y.keys()):
        domain = list(df[df.problem == key].domain.unique())[0]
        fig, ax = plt.subplots(1, 2)
        x_real = [i for i,j in problem_y[key]]
        y_real = [j for i,j in problem_y[key]]
        x_pred = [i for i,j in problem_pred[key]]
        y_pred = [j for i,j in problem_pred[key]]
        ax[0].scatter(x_real, y_real)
        ax[1].scatter(x_pred, y_pred)
        ax[0].set_xlabel('horizon')
        ax[0].set_ylabel('runtime')
        ax[0].set_title('Actual runtimes')
        ax[1].set_xlabel('horizon')
        ax[1].set_ylabel('runtime')
        ax[1].set_title('Predicted runtimes')

        path = os.path.join(os.pardir, 'Results','Results_runtime', f'{domain}_{problem}_{filename}')
        fig.savefig(path, dpi=300)


def visualize_loss(train_loss, val_loss, test_loss, filename):
    fig, ax = plt.subplots()
    x = list(range(len(train_loss)))
    ax.plot(x, train_loss, label='Train loss')
    ax.plot(x, val_loss, label='Validation loss')
    ax.plot(x, [test_loss]*len(x), '--', label='Test loss')
    ax.legend()

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss per epochs.')
    path = os.path.join(os.pardir, 'Results', f'{filename}_lossGraph.png')
    fig.savefig(path, dpi=300)




    




