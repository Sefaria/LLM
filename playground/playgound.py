import optuna


my_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

def objective(trial):
    # Suggest a permutation of indices as the trial
    permutation = [trial.suggest_int(f'index_{i}', 0, len(my_array)-1) for i in range(len(my_array))]
    # n = len(my_array)  # You can adjust this value as needed
    # lower_bound = 0
    # upper_bound = len(my_array)-1

    # Suggest n unique integers using suggest_int in a loop
    # unique_integers = []

    # while len(unique_integers) < n:
    #     unique_int = trial.suggest_int(f'index_{len(unique_integers)}', lower_bound, upper_bound)
    #     if unique_int not in unique_integers:
    #         unique_integers.append(unique_int)


    # # Suggest one of the unique integers using suggest_categorical
    # selected_integers = []
    # for i in range(0, upper_bound+1):
    #     selected_integers += [trial.suggest_categorical(f'index_{i}', indexes_to_choose-set(selected_integers))]
    #
    # permutation = selected_integers
    # Use the suggested permutation to sort the array
    sorted_array = [my_array[i] for i in permutation]

    if len(permutation) != len(set(permutation)):
        score = 99999999
    else:
        # Evaluate the quality of the sorting (lower is better, as we want ascending order)
        score = sum(sorted_array[i] > sorted_array[i + 1] for i in range(len(sorted_array) - 1))

    return score

if __name__ == "__main__":
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)

    # Get the best trial
    best_trial = study.best_trial.params

    # Use the best trial to get the permutation indices
    permutation = [best_trial[f'index_{i}'] for i in range(len(my_array))]

    # Use the best trial to sort the array
    sorted_array = [my_array[i] for i in permutation]

    # Print the results
    print("Original array:", my_array)
    print("Sorted array:", sorted_array)