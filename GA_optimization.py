import numpy as np
import random
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb

BASE = "./features"
FEATURES_KEY = "features"
LABELS_KEY   = "labels"


def blended_crossover(parent1, parent2):
    
    # calculate diffence between parents
    diff = np.abs(parent1 - parent2)

    # define min and max values for the offspring using alpha and beta
    alpha, beta = np.random.rand(), np.random.rand()  # Randomly generate alpha and beta between 0 and 1

    min_values = np.minimum(parent1, parent2) - (alpha * diff)
    max_values = np.maximum(parent1, parent2) + (beta * diff)

    # generate offsprings genes randomly within the defined range
    offspring1 = np.random.uniform(min_values, max_values)
    offspring2 = np.random.uniform(min_values, max_values)
    return offspring1, offspring2

def mutation(offspring, mutation_rate=0.20):
    # Mutate each gene with a certain probability
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            # Randomly change the gene value (for binary, flip the bit)
            offspring[i] = 1 - offspring[i]  # Assuming binary representation
    return offspring

def tournament_selection(population, fitness_scores, tournament_size=3):
    # Randomly select 'tournament_size' individuals from the population
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_individuals = [population[i] for i in selected_indices]
    selected_fitness = [fitness_scores[i] for i in selected_indices]

    # Select the individual with the highest fitness score
    best_index = np.argmax(selected_fitness)
    return selected_individuals[best_index]

def evolve(population, fitness, mutation_rate=0.20, tournament_size=3):
    new_population = []
    for _ in range(len(population) // 2):
        # Select parents using tournament selection
        parent1 = tournament_selection(population, fitness, tournament_size)
        parent2 = tournament_selection(population, fitness, tournament_size)

        # Perform blended crossover
        offspring1, offspring2 = blended_crossover(parent1, parent2)

        # Apply mutation to the offspring
        offspring1 = mutation(offspring1, mutation_rate)
        offspring2 = mutation(offspring2, mutation_rate)

        new_population.extend([offspring1, offspring2])

    return new_population


# def feature_encoding(features):
#     # Encode features with values (0,1)
#     # features with range [0, 0.5) are rejected
#     # features with range [0.5, 1] are accepted
#     encoded = np.where(features >= 0.5, 1, 0)
#     return encoded



# def fitness_function(result, selected_features, total_features, alpha=0.6, beta=0.4):
#     """
#     Calculate the fitness of a solution based on its performance and the number of features selected.

#     Parameters:
#     - result: The performance metric (e.g., accuracy) of the model using the selected features.
#     - selected_features: The number of features selected in the solution.
#     - total_features: The total number of features available in the dataset.
#     - alpha: The weight for the performance metric (default is 0.6).
#     - beta: The weight for the feature selection penalty (default is 0.4).

#     Returns:
#     - fitness: The calculated fitness score for the solution.
#     """
#     # Calculate the feature selection penalty
#     feature_penalty = selected_features / total_features

#     # Calculate the fitness score
#     fitness = alpha * result - beta * feature_penalty

#     return fitness


class FeatureSelector:

    ALPHA = 0.6
    BETA = 0.4

    def __init__(self, pop_size = 100, epochs = 100, mutation_prob = 0.2, n_tournament = 3, cv = 5):
        self.pop_size = pop_size
        self.epochs = epochs
        self.mutation_prob = mutation_prob
        self.n_tournament = n_tournament
        self.cv = cv

        self.selection_mask = None
        self.best_score = None
        self.history = []
    
    @staticmethod
    def decode(chromosome):
        # [0, .5) -> 0 (not selected)
        # [.5, 1] -> 1 (selected)

        rejected = (chromosome >= 0) & (chromosome < 0.5)
        selected = (chromosome >= 0.5) & (chromosome <= 1)

        return selected & ~rejected
    
    def fitness_function(self, chromosome, X, y):
        total_features = len(chromosome)
        mask = self.decode(chromosome)
        selected_features = np.sum(mask)
        print(f"Selected {selected_features} features out of {total_features}")

        if selected_features == 0:
            return 1

        model = lgb.LGBMClassifier(random_state = 42, force_col_wise=True, verbose=-1)

        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)


        try:
            error = 1 - cross_val_score(model, X[:, mask], y, cv=cv, scoring='accuracy').mean()
        except ValueError:
            error = 1
        
        print(f"Error: {error:.4f}")
        
        return self.ALPHA * (1 - error) - self.BETA * (selected_features / total_features)
    
    def fit(self, X, y):

        n_features = X.shape[1]
        pop = np.random.rand(self.pop_size, n_features)  # Randomly initialize population
        fits = np.array([self.fitness_function(chromosome, X, y) for chromosome in pop])

        best_chromosome = pop[np.argmax(fits)]
        best_fit = float(fits.min())
        self.history.append(best_fit)

        for _ in range(self.epochs):
            pop = evolve(pop, fits, self.mutation_prob, self.n_tournament)
            fits = np.array([self.fitness_function(chromosome, X, y) for chromosome in pop])


            current_best_fit = float(fits.min())
            self.history.append(current_best_fit)

            if current_best_fit < best_fit:
                best_fit = current_best_fit
                best_chromosome = pop[np.argmin(fits)]
        
        self.selection_mask = self.decode(best_chromosome)
        self.best_score = best_fit

        n_selected = self.selection_mask.sum()
        print(f"Selected {n_selected} features with fitness score: {best_fit:.4f}")
        return self
    
    def transform(self, X):
        if self.selection_mask is None:
            raise ValueError("Model has not been fitted yet.")
        return X[:, self.selection_mask]

LIGHTGBM_PARAMS_RANGE = {
    "n_estimators"      : (50,   500),
    "max_depth"         : (3,    12),
    "min_child_weight"  : (1,    20),
    "learning_rate"     : (0.01, 0.3),
    "subsample"         : (0.5,  1.0),
    "colsample_bytree"  : (0.5,  1.0),
    "reg_alpha"         : (0.0,  1.0),
    "reg_lambda"        : (0.0,  1.0),
    "num_leaves"        : (20,   150),
    "min_child_samples" : (5,    50),
    "bagging_freq"      : (0,    10),
    "feature_fraction"  : (0.5,  1.0),
    "bagging_fraction"  : (0.5,  1.0),
}

INTEGER_PARAMS = {"n_estimators", "max_depth", "min_child_weight", "num_leaves", "min_child_samples", "bagging_freq"}

class lightgbm_GA_Optimizer:

    ALPHA = 0.6
    BETA = 0.4

    def __init__(self, pop_size = 100, epochs = 100, mutation_prob = 0.2, n_tournament = 3, cv = 5):
        self.pop_size = pop_size
        self.epochs = epochs
        self.mutation_prob = mutation_prob
        self.n_tournament = n_tournament
        self.cv = cv

        self.best_params = None
        self.best_score = None
        self.history = []
    
    @staticmethod
    def decode(chromosome):
        params = {}
        for i, (param, (low, high)) in enumerate(LIGHTGBM_PARAMS_RANGE.items()):
            value = chromosome[i] * (high - low) + low  # Scale to the parameter range
            if param in INTEGER_PARAMS:
                value = int(round(value))  # Round to nearest integer if it's an integer parameter
            params[param] = value
        return params

    def fitness_function(self, chromosome, X, y):
        params = self.decode(chromosome)

        model = lgb.LGBMClassifier(random_state = 42, force_col_wise=True, verbose=-1, **params)

        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        try:
            error = 1 - cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
        except ValueError:
            error = 1
        
        print(f"Error: {error:.4f} with params: {params}")
        
        return self.ALPHA * (1 - error) - self.BETA * (sum(chromosome) / len(chromosome))
    
    def fit(self, X, y):
        n_params = len(LIGHTGBM_PARAMS_RANGE)
        pop = np.random.rand(self.pop_size, n_params)  # Randomly initialize population
        fits = np.array([self.fitness_function(chromosome, X, y) for chromosome in pop])

        best_chromosome = pop[np.argmax(fits)]
        best_fit = float(fits.min())
        self.history.append(best_fit)

        for _ in range(self.epochs):
            pop = evolve(pop, fits, self.mutation_prob, self.n_tournament)
            fits = np.array([self.fitness_function(chromosome, X, y) for chromosome in pop])


            current_best_fit = float(fits.min())
            self.history.append(current_best_fit)

            if current_best_fit < best_fit:
                best_fit = current_best_fit
                best_chromosome = pop[np.argmin(fits)]
        
        self.best_params = self.decode(best_chromosome)
        self.best_score = best_fit

        print(f"Best params: {self.best_params} with fitness score: {best_fit:.4f}")
        return self
    
    def get_model(self):
        if self.best_params is None:
            raise ValueError("Model has not been fitted yet.")
        return lgb.LGBMClassifier(random_state = 42, force_col_wise=True, verbose=-1, **self.best_params)

def load_dataset(BASE_DIR, FEATURES_KEY, LABELS_KEY):
    # Load features and labels

    # Train
    train_features = np.load(os.path.join(BASE_DIR, "train_features.npz"))[FEATURES_KEY]
    train_labels   = np.load(os.path.join(BASE_DIR, "train_features.npz"))[LABELS_KEY]

    # Validation
    val_features   = np.load(os.path.join(BASE_DIR, "val_features.npz"))[FEATURES_KEY]
    val_labels     = np.load(os.path.join(BASE_DIR, "val_features.npz"))[LABELS_KEY]

    # Test
    test_features  = np.load(os.path.join(BASE_DIR, "test_features.npz"))[FEATURES_KEY]
    test_labels    = np.load(os.path.join(BASE_DIR, "test_features.npz"))[LABELS_KEY]

    return train_features, train_labels, val_features, val_labels, test_features, test_labels

def main():
    # Load features and labels
    train_features, train_labels, val_features, val_labels, test_features, test_labels = load_dataset(BASE, FEATURES_KEY, LABELS_KEY)

    feature_selector = FeatureSelector(pop_size=50, epochs=20)
    feature_selector.fit(train_features, train_labels)

    X_train_selected = feature_selector.transform(train_features)
    X_val_selected = feature_selector.transform(val_features)
    X_test_selected = feature_selector.transform(test_features)

    tuner = lightgbm_GA_Optimizer(pop_size=50, epochs=20)
    tuner.fit(X_train_selected, train_labels)

    model = tuner.get_model()
    model.fit(X_train_selected, train_labels)

    for dataset, X, y in [("Validation", X_val_selected, val_labels), ("Test", X_test_selected, test_labels)]:
        score = model.classification_report(y, model.predict(X))
        print(f"{dataset} Classification Report:\n{score}")

if __name__ == "__main__":
    main()
