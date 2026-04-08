import numpy as np
import random
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

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
    offspring1 = np.clip(np.random.uniform(min_values, max_values), 0, 1)
    offspring2 = np.clip(np.random.uniform(min_values, max_values), 0, 1)
    return offspring1, offspring2

def mutation(offspring, mutation_rate=0.20):
    # Mutate each gene with a certain probability
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            # Randomly change the gene value for real-valued genes between 0 and 1
            offspring[i] = np.random.rand()
            
    return offspring

def tournament_selection(population, fitness_scores, tournament_size=3,):
    # Randomly select individuals for the tournament
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = fitness_scores[selected_indices]

    # Select the individual with the best fitness (lowest score)
    winner_index = selected_indices[np.argmin(selected_fitness)]
    return population[winner_index]


def evolve(population, fitness, mutation_rate=0.20, tournament_size=3, n_elites=1):
    
    new_population = []

    # Elitism: Keep the best individuals
    elite_indices = np.argsort(fitness)[:n_elites]
    new_population.extend(population[elite_indices])

    while len(new_population) < len(population):
        # Select parents using tournament selection
        parent1 = tournament_selection(population, fitness, tournament_size)
        parent2 = tournament_selection(population, fitness, tournament_size)

        # Crossover
        offspring1, offspring2 = blended_crossover(parent1, parent2)

        # Mutation
        offspring1 = mutation(offspring1, mutation_rate)
        offspring2 = mutation(offspring2, mutation_rate)

        new_population.append(offspring1)
        if len(new_population) < len(population):
            new_population.append(offspring2)

    return np.array(new_population)


LIGHTGBM_PARAMS_RANGE = {
    "n_estimators"      : (50,   500),
    "max_depth"         : (3,    12),
    "min_child_weight"  : (1,    20),
    "learning_rate"     : (0.01, 0.3),
    "subsample"         : (0.5,  1.0),
    "colsample_bytree"  : (0.5,  1.0),
    "reg_alpha"         : (1.0,  10.0),
    "reg_lambda"        : (1.0,  10.0),
    "num_leaves"        : (20,   150),
    "min_child_samples" : (5,    50),
    "bagging_freq"      : (1,    10),
    "feature_fraction"  : (0.5,  1.0),
    "bagging_fraction"  : (0.5,  1.0),
}

INTEGER_PARAMS = {"n_estimators", "max_depth", "min_child_weight", "num_leaves", "min_child_samples", "bagging_freq"}

# class that combines feature selection and hyperparameter tuning 
class GA_Optimizer:

    ALPHA = 0.6
    BETA = 0.4
    MIN_FEATURES = 3 # Based on the paper

    def __init__(self, pop_size = 100, epochs = 100, mutation_prob = 0.2, n_tournament = 3, cv = 5):
        self.pop_size = pop_size
        self.epochs = epochs
        self.mutation_prob = mutation_prob
        self.n_tournament = n_tournament
        self.cv = cv

        self.best_params = None
        self.selection_mask = None
        self.best_score = None
        self.history = []

    def feature_selection_decode(self, chromosome):
        # [0, .5) -> 0 (not selected)
        # [.5, 1] -> 1 (selected)

        rejected = (chromosome >= 0) & (chromosome < 0.5)
        selected = (chromosome >= 0.5) & (chromosome <= 1)

        return selected & ~rejected

    def hyperparameter_decode(self, chromosome):
        params = {}
        for i, (param, (low, high)) in enumerate(LIGHTGBM_PARAMS_RANGE.items()):     
            value = chromosome[i] * (high - low) + low  # Scale to the parameter range
            if param in INTEGER_PARAMS:
                value = int(round(value))  # Round to nearest integer if it's an integer parameter
                value = np.clip(value, low, high)  # Ensure the value is within the specified range
            else:
                value = float(np.clip(value, low, high))  # Ensure the value is within the specified range
            params[param] = value
        return params
       
    
    def fitness_function(self, chromosome, X, y):
        n_features = X.shape[1]
        feature_chromosome = chromosome[:n_features]
        param_chromosome = chromosome[n_features:]

        # Decode feature selection
        feature_mask = self.feature_selection_decode(feature_chromosome)
        selected_X = X[:, feature_mask]

        if selected_X.shape[1] < self.MIN_FEATURES:
            return 1  # Penalize if no features are selected

        # Decode hyperparameters
        params = self.hyperparameter_decode(param_chromosome)

        model = lgb.LGBMClassifier(random_state = 42, force_col_wise=True, verbose=-1, **params)

        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        try:
            error = 1 - cross_val_score(model, selected_X, y, cv=cv, scoring='accuracy').mean()
        except ValueError:
            error = 1
        
        print(f"Error: {error:.4f} with params: {params} and {feature_mask.sum()} features")
        
        return self.ALPHA * error + self.BETA * (feature_mask.sum() / n_features)
    
    def fit(self, X, y):
        n_features = X.shape[1]
        n_params = len(LIGHTGBM_PARAMS_RANGE)
        total_length = n_features + n_params

        pop = np.random.rand(self.pop_size, total_length)  # Randomly initialize population
        fits = np.array([self.fitness_function(chromosome, X, y) for chromosome in pop])

        best_chromosome = pop[np.argmin(fits)].copy()
        best_fit = float(fits.min())
        self.history.append(best_fit)

        for _ in range(self.epochs):
            pop = evolve(pop, fits, self.mutation_prob, self.n_tournament)
            fits = np.array([self.fitness_function(chromosome, X, y) for chromosome in pop])
            current_best_fit = float(fits.min())
            

            if current_best_fit < best_fit:
                best_fit = current_best_fit
                best_chromosome = pop[np.argmin(fits)].copy()
        
            self.history.append(best_fit)
        
        self.selection_mask = self.feature_selection_decode(best_chromosome[:n_features])
        self.best_params = self.hyperparameter_decode(best_chromosome[n_features:])
        self.best_score = best_fit

        n_selected = self.selection_mask.sum()
        print(f"Selected {n_selected} features with params: {self.best_params} and fitness score: {best_fit:.4f}")
        return self

    def get_model(self):
        if self.best_params is None or self.selection_mask is None:
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
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load features and labels
    train_features, train_labels, val_features, val_labels, test_features, test_labels = load_dataset(BASE, FEATURES_KEY, LABELS_KEY)

    # Initialize and fit the GA optimizer
    ga_optimizer = GA_Optimizer(pop_size=75, epochs=25, mutation_prob=0.2, n_tournament=3, cv=5)
    ga_optimizer.fit(train_features, train_labels)
    
    # Get the best model and selected features
    model = ga_optimizer.get_model()
    model.fit(train_features[:, ga_optimizer.selection_mask], train_labels)
    
    
    

    for dataset, X, y in [("Validation", val_features, val_labels), ("Test", test_features, test_labels)]:

        X = X[:, ga_optimizer.selection_mask]  # Use only the selected features


        # Evaluate the model's performance on the validation and test sets
        y_pred = model.predict(X)

        # Accuracy
        accuracy = np.mean(y_pred == y)
        print(f"{dataset} Accuracy: {accuracy:.4f}")
       
        # Precision
        precision = precision_score(y, y_pred, average='weighted')
        print(f"{dataset} Precision: {precision:.4f}")

        # Recall
        recall = recall_score(y, y_pred, average='weighted')
        print(f"{dataset} Recall: {recall:.4f}")

        # F1 Score
        f1 = f1_score(y, y_pred, average='weighted')
        print(f"{dataset} F1 Score: {f1:.4f}")
        


if __name__ == "__main__":
    main()
