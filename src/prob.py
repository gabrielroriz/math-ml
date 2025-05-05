import numpy as np

# 1 - Given a pre-defined date, what is the value of n such that the probability of having a match is greater than or equal to 0.5?

def simulate(problem_func, n_students=365, n_simulations=1000):
    
    # Initialize the counter of matches at 0
    matches = 0
    
    # Run the simulation for the desired number of times
    for _ in range(n_simulations):
        
        # If there is a match in the classroom add 1 to the counter of matches
        if problem_func(n_students):
            matches += 1
    
    # Return the ratio of number of matches / number of simulations
    return matches/n_simulations


def problem_1(n_students):
    
    # Predefine a specific birthday
    predef_bday = np.random.randint(0, 365)
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Check if predefined bday is among students
    return predef_bday in gen_bdays


import matplotlib.pyplot as plt
import mplcursors

def scatterplot_simulated_probs(simulated_probs, classroom_sizes):
    """
    Plota as probabilidades simuladas vs o tamanho da sala com tooltip interativo.

    Parâmetros:
    - simulated_probs: lista ou array de probabilidades simuladas (valores entre 0 e 1)
    - classroom_sizes: lista ou array com os tamanhos de sala correspondentes
    """
    plt.figure(figsize=(10, 6))

    # Scatterplot com borda branca
    scatter = plt.scatter(
        classroom_sizes,
        simulated_probs,
        label="simulated probabilities",
        edgecolors='white',
        linewidths=0.5
    )

    # Linha horizontal em p = 0.5
    plt.axhline(y=0.5, color='red', linestyle='-', label='p = 0.5')

    # Títulos e rótulos
    plt.title("Probability vs Number of Students")
    plt.xlabel("Classroom Size")
    plt.ylabel("Simulated Probability")
    plt.legend()
    plt.grid(True)

    # Tooltip interativo com mplcursors
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"Classroom size = {classroom_sizes[sel.index]}\nProbability = {simulated_probs[sel.index]:.4f}"
    ))

    plt.show()

def problem_2(n_students):
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Pick one student at random
    rnd_index = np.random.randint(0, len(gen_bdays))
    
    # Get the bday from the selected student
    rnd_bday = gen_bdays[rnd_index]
    
    # Take the bday out of the pool of bdays (otherwise there is always a match)
    remaining_bdays = np.delete(gen_bdays, rnd_index, axis=0)
    
    # Check if another student shares the same bday
    return rnd_bday in remaining_bdays


def problem_3(n_students):
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Get array containing unique bdays
    unique_bdays = np.array(list(set(gen_bdays)))
    
    # Check that both the original and unique arrays have the same length 
    # (if so then no two students share the same bday)
    return len(unique_bdays) != len(gen_bdays)

def problem_4(n_students):
    
    # Generate birthdays for every student in classroom 1
    gen_bdays_1 = np.random.randint(0, 365, (n_students))
    
    # Generate birthdays for every student in classroom 2
    gen_bdays_2 = np.random.randint(0, 365, (n_students))
    
    # Check for any match between both classrooms
    return np.isin(gen_bdays_1, gen_bdays_2).any()

def single_run():
    n = 100
    simulated_prob = simulate(problem_1, n_students=n, n_simulations=10000)
    print(f"The simulated probability of any student to have a bday equal to a predefined value is {simulated_prob} in a classroom with {n} students")

# Generate the simulated probability for every classroom
def get_simulated_probabilities():
    max_classroom_size = 1000
    classroom_sizes = list(range(1, max_classroom_size))

    simulated_probs_1 = [simulate(problem_1, n_students=n, n_simulations=1000) for n in classroom_sizes]
    # Create a scatterplot of simulated probabilities vs classroom size
    scatterplot_simulated_probs(simulated_probs_1, classroom_sizes)


def get_simulated_probabilities_2():
    max_classroom_size = 1000
    classroom_sizes = list(range(1, max_classroom_size))

    # Generate the simulated probability for every classroom
    simulated_probs_2 = [simulate(problem_2, n_students=n) for n in classroom_sizes]

    # Create a scatterplot of simulated probabilities vs classroom size
    scatterplot_simulated_probs(simulated_probs_2, classroom_sizes)

def get_simulated_probabilities_3():
    max_classroom_size = 1000
    classroom_sizes = list(range(1, max_classroom_size))

    # Generate the simulated probability for every classroom
    simulated_probs_3 = [simulate(problem_3, n_students=n, n_simulations=1000) for n in classroom_sizes]

    # Create a scatterplot of simulated probabilities vs classroom size
    scatterplot_simulated_probs(simulated_probs_3, classroom_sizes)

# 3 - Given a classroom with students, what is the value of n such that the probability of having a match is greater than or equal to 0.5 for any two students?

get_simulated_probabilities_3()