"""
This script configures and launches an evolutionary algorithm, using DEAP, to evolve AST poker strategies.

Author: John McCarroll
"""

import operator
import random
import numpy as np
from deap import base, creator, tools, gp
import matplotlib.pyplot as plt
from ai_poker.mvp.poker_env import PokerEnv
from ai_poker.mvp.agents import RandomAgent, RandomAgent2
from ai_poker.genetic.simple_agents import ManiacAgent, StationAgent, SimpleValueAgent
import random
import os
import pickle
import datetime
import inspect
import multiprocessing
import inspect
from deap import gp
from itertools import zip_longest
from ai_poker.genetic.AST_agent import ASTAgent
from ai_poker.genetic.constants import (
    OUTPUT_TYPE,
    ARGUMENT_TYPES,
    FLOAT_CONSTANTS,
    HAND_CLASSES,
    STREETS,
    BETTING_LINES,
    INITIAL_MAX_TREE_HEIGHT,
    INITIAL_MIN_TREE_HEIGHT,
    MAX_TREE_HEIGHT,
    MAX_NODE_COUNT,
    VERBOSE,
    POP_SIZE,
    N_GEN,
    MAX_HANDS,
    EVALUATION_BENCH_SIZE,
    WIN_RATE_FITNESS_WEIGHT,
    NODE_COUNT_FITNESS_WEIGHT,
    SEED,
    SAVE_EVERY_X_GEN,
    VERSION_NUM
)


# DEFINE HELPER FUNCTIONS

# Define a protected division function to avoid ZeroDivisionError
def protected_div(left, right):
    """A protected division operator that returns 1.0 if the denominator is zero."""
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

# Define a conditional operator
def if_then_else(condition, out1, out2):
    """A conditional if-then-else operator."""
    return out1 if condition else out2

def get_win_rate(individual):
    """Returns the first fitness value (the win rate)."""
    if individual.fitness.valid:
        return individual.fitness.values[0]
    return 0 # Return a very low value if invalid

def get_size(individual):
    """Returns the first fitness value (the win rate)."""
    if individual.fitness.valid:
        return individual.fitness.values[1]
    return 0 # Return a very low value if invalid

def run_evaluation(task):
    """
    Wrapper function to run a single evaluation match in a worker process.
    """
    # Unpack the task
    agent1_tree, agent2_tree, max_hands, i, j = task

    def compile_agent(agent_tree_or_class):
        """Helper to compile a DEAP tree or return a class."""
        if inspect.isclass(agent_tree_or_class):
            # It's a benchmark agent like RandomAgent
            return agent_tree_or_class
        elif isinstance(agent_tree_or_class, list):
            # It's a DEAP individual (from pop or fossil_record)
            deap_ind = creator.Individual(agent_tree_or_class)
            return toolbox.compile(expr=deap_ind)
        else:
            # Fallback for unexpected types
            print(f"Warning: Unknown agent type in eval: {type(agent_tree_or_class)}")
            return agent_tree_or_class

    try:
        # 2. Compile logic *inside the worker*
        agent1_logic = compile_agent(agent1_tree)
        agent2_logic = compile_agent(agent2_tree)
        
        # 3. Run the evaluation
        w1, w2, n_hands = toolbox.evaluate(agent1_logic, agent2_logic, max_hands=max_hands)
        
        # 4. Return results with indices to map back
        return (i, j, w1, w2, n_hands)
        
    except Exception as e:
        # Catch errors from bad individuals
        print(f"Error evaluating task ({i} vs {j}): {e}")
        return (i, j, 0, 0, 1) # Return 0 winnings, 1 hand (to avoid divide-by-zero)

def evaluate_agents(agent1_logic, agent2_logic, max_hands=500):
    """
    Simulates a heads-up poker match between two compiled agents.
    Returns the final winnings for each agent.
    """
    
    winnings = [0.0, 0.0]
    num_hands = 0

    # Instantiate heads up poker env
    env = PokerEnv(
        num_players=2,
        num_streets=4,
        blinds=[1,2],
        antes=0,
        raise_sizes='inf',
        num_raises=float('inf'),
        num_suits=4,
        num_ranks=13,
        num_hole_cards=2,
        num_community_cards=[0, 3, 1, 1],
        num_cards_for_hand=5,
        mandatory_num_hole_cards=0,
        start_stack=500,
        low_end_straight=True
    )

    # Instantiate agents
    player1 = agent1_logic
    player2 = agent2_logic
    if inspect.isclass(agent1_logic):
        player1 = agent1_logic(dealer=env.dealer, seat_id=0)
    else:
        player1 = ASTAgent(dealer=env.dealer, seat_id=0, ast=agent1_logic)
        
    if inspect.isclass(agent2_logic):
        player2 = agent2_logic(dealer=env.dealer, seat_id=1)
    else:
        player2 = ASTAgent(dealer=env.dealer, seat_id=1, ast=agent2_logic)

    env.register_agents([player1, player2])
    obs = env.reset()

    done = [False]
    game_over = False

    # Simulate poker game
    while True and num_hands < max_hands:

        bet = env.act(obs)
        obs, rewards, done, info = env.step(bet)

        if all(done):
            game_over = 0 in env.dealer.stacks
            num_hands += 1
            winnings[0] += rewards[0]
            winnings[1] += rewards[1]

            if isinstance(player1, ASTAgent):
                player1.hand_complete(env.dealer.history, rewards)
            if isinstance(player2, ASTAgent):
                player2.hand_complete(env.dealer.history, rewards)
            
            if game_over:
                break
            obs = env.reset()

    return winnings[0], winnings[1], num_hands


# DEAP SET UP

# Create the Primitive Set
# This defines the building blocks (functions and terminals) for our program trees.
pset = gp.PrimitiveSetTyped("main", ARGUMENT_TYPES, OUTPUT_TYPE)

# --- Add Primitives (Operators) ---
# Logical operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# Comparison operators for floats
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(operator.ne, [float, float], bool)

# Comparison operators for strings (hand_class, street)
pset.addPrimitive(operator.eq, [str, str], bool)
pset.addPrimitive(operator.ne, [str, str], bool)

# Arithmetic operators
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float)

# Conditional operators
pset.addPrimitive(if_then_else, [bool, OUTPUT_TYPE, OUTPUT_TYPE], OUTPUT_TYPE)
pset.addPrimitive(if_then_else, [bool, str, str], str, name="if_then_else_str")

# --- Add Terminals (Constants and Inputs) ---
# Add float constants
for val in FLOAT_CONSTANTS:
    pset.addTerminal(val, float)

# Add string constants for hand classifications
for hand_class in HAND_CLASSES:
    pset.addTerminal(hand_class, str)

# Add string constants for streets
for street in STREETS:
    pset.addTerminal(street, str)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

# betting line strings
for line in BETTING_LINES:
    pset.addTerminal(line, str)

# Rename arguments for clarity
pset.renameArguments(
    ARG0='pot_size', 
    ARG1='button', 
    ARG2='stack_size',
    ARG3='opponent_stack_size', 
    ARG4='amount_to_call',
    ARG5='hand_strength', 
    ARG6='hand_class', 
    ARG7='street',

    ARG8='board_highest_card',
    ARG9='board_num_pairs',
    ARG10='board_num_trips', 
    ARG11='board_num_quads',
    ARG12='board_num_same_suits',
    ARG13='board_smallest_3_card_span',

    ARG14='hole_suited',
    ARG15='hole_highest_card',
    ARG16='hole_paired',
    ARG17='hole_flush_draw',
    ARG18='hole_open_straight_draw',
    ARG19='hole_gutshot_straight_draw',

    ARG20='preflop_opponent_line',
    ARG21='flop_opponent_line',
    ARG22='turn_opponent_line',
    ARG23='river_opponent_line',

    ARG24='num_hands',
    ARG25='VPIP',
    ARG26='PFR_PCT',
    ARG27='THREEBET_PCT',
    ARG28='WTSD',
    ARG29='WSD',
    ARG30='WWSF',
    ARG31='AF',
    ARG32='CBET_PCT',
    ARG33='DONK_PCT',
    ARG34='CHECKRAISE_PCT',
)

# --- 2. DEAP Toolbox Setup ---

# Define the fitness criteria: maximize winnings
creator.create("FitnessMax", base.Fitness, weights=(WIN_RATE_FITNESS_WEIGHT, NODE_COUNT_FITNESS_WEIGHT))
# Define the individual: a program tree with the defined fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator:
toolbox.register("expr", gp.genFull, pset=pset, min_=INITIAL_MIN_TREE_HEIGHT, max_=INITIAL_MAX_TREE_HEIGHT)

# Structure initializers:
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Register the evaluation, selection, crossover, and mutation operators
toolbox.register("evaluate", evaluate_agents)
toolbox.register("select", tools.selBest) # Select the best X individuals
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=INITIAL_MIN_TREE_HEIGHT, max_=INITIAL_MAX_TREE_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorate crossover and mutation to prevent code bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))

# Initialize statistics
stats = tools.Statistics(key=get_win_rate)
stats.register("win_rate_avg", np.mean)
stats.register("win_rate_std", np.std)
stats.register("win_rate_min", np.min)
stats.register("win_rate_max", np.max)

stats2 = tools.Statistics(key=get_size)
stats2.register("size_avg", np.mean)
stats2.register("size_std", np.std)
stats2.register("size_min", np.min)
stats2.register("size_max", np.max)

logbook = tools.Logbook()

# DEFINE MAIN EVOLUTIONARY LOOP

def main():
    random.seed(SEED)

    pop = toolbox.population(n=POP_SIZE)
    fossil_record = {}

    print("--- Starting Evolution ---")
    for gen in range(N_GEN):        
        # --- 1. Prepare all evaluation tasks ---
        tasks = []

        # Create bench
        # --- Task Group 2: Benchmark (Pop vs. Bench) ---
        static_bench = [SimpleValueAgent, StationAgent, ManiacAgent, RandomAgent, RandomAgent2]
        num_static_bots = len(static_bench)
        multiplier = EVALUATION_BENCH_SIZE // num_static_bots
        bot_bench = static_bench*multiplier # populate eval bench with static bots

        fossil_bench_dicts = list(fossil_record.values())[-EVALUATION_BENCH_SIZE:]
        fossil_bench = [fossil['individual'] for fossil in fossil_bench_dicts]
        num_fossils = len(fossil_bench)
        num_bots_needed = EVALUATION_BENCH_SIZE - num_fossils
        full_bench = fossil_bench + bot_bench[0:num_bots_needed]
        
        for i in range(len(pop)):
            for j, opponent in enumerate(full_bench):
                # We send the raw tree (pop[i]) and the opponent (class or tree)
                # We use 'None' as the 'j' index to mark it as a bench match
                tasks.append((pop[i], opponent, MAX_HANDS, i, j))
        
        # --- 2. Run all tasks in parallel ---
        results_iterator = pool.imap_unordered(run_evaluation, tasks)
        
        # # --- 3. Process results ---
        winnings_map = {i: 0 for i in range(len(pop))}
        num_hands_map = {i: 0 for i in range(len(pop))}
        
        bench_winnings_map = {i: 0 for i in range(len(full_bench))}
        bench_num_hands_map = {i: 0 for i in range(len(full_bench))}

        # --- 3. Process results from the iterator ---
        for task_count, res in enumerate(results_iterator):
            # This loop receives results as soon as a worker finishes one task
            i, j, w1, w2, n_hands = res
            
            # Update scores for individual i
            winnings_map[i] += w1
            num_hands_map[i] += n_hands
            
            # Update scores for bench players
            if j is not None:
                bench_winnings_map[j] += w2
                bench_num_hands_map[j] += n_hands
                
        # --- 4. Assign fitness (your code, slightly modified) ---
        for i, ind in enumerate(pop):
            node_count = len(ind)
            if num_hands_map[i] > 0:
                ind.fitness.values = (winnings_map[i] / num_hands_map[i], node_count)
            else:
                ind.fitness.values = (0,node_count)
                print("WARNING: individual player had no matches")

        # calc bench win rate
        bench_win_rate_map = {i: {'win_rate': 0.0, 'opponent': None} for i in range(len(full_bench))}
        highest_win_rate = {'win_rate': 0.0, 'opponent_name': None}
        for i, ind in enumerate(full_bench):
            if bench_num_hands_map[i] > 0:
                win_rate = bench_winnings_map[i] / bench_num_hands_map[i]
                bench_win_rate_map[i]['win_rate'] = win_rate
                bench_win_rate_map[i]['opponent'] = ind
                if win_rate > highest_win_rate['win_rate']:
                    highest_win_rate['win_rate'] = win_rate
                    if isinstance(ind, list):
                        if gen >= EVALUATION_BENCH_SIZE:
                            gen_num = i + gen + 1 - EVALUATION_BENCH_SIZE
                        else:
                            gen_num = i
                        opp_name = f'Fossil Record Gen {gen_num}'
                    else:
                        opp_name = str(ind)
                    highest_win_rate['opponent_name'] = opp_name
            else:
                print("WARNING: bench player had no matches")
        
        if VERBOSE:
            print(f'Highest winning bench player: win rate: {highest_win_rate["win_rate"]}, bench player: {highest_win_rate["opponent_name"]}')

        # --- Log statistics ---
        win_rate_stats = stats.compile(pop)
        size_stats = stats2.compile(pop)
        record = {'gen': gen, **win_rate_stats, **size_stats}

        logbook.record(**record)
        print(logbook.stream)

        # --- Save the best of the generation to the fossil record
        best_ind = tools.selBest(pop, 1)[0]
        fossil_record[gen] = {
            'fitness': best_ind.fitness.values[0],
            'individual': best_ind,
            'gen': gen
        }

        # --- Visualize generation's best individual ---
        if VERBOSE:
            print(f'Best Individual of Generation {gen}:')
            print(str(best_ind))

        # --- Selection ---
        num_survivors = POP_SIZE // 4
        survivors = toolbox.select(pop, k=num_survivors)
        
        # --- Create the next generation ---
        offspring1 = [toolbox.clone(ind) for ind in survivors]
        offspring2 = [toolbox.clone(ind) for ind in survivors]
        offspring3 = [toolbox.clone(ind) for ind in survivors]
        
        # Apply crossover
        for child1, child2 in zip_longest(offspring2[::2], offspring2[1::2], fillvalue=None):
            if child1 is None: 
                toolbox.mutate(child2)
                del child2.fitness.values
            elif child2 is None:
                toolbox.mutate(child1)
                del child1.fitness.values
            else:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Introduce new organisms
        offspring3 = [toolbox.individual() for _ in range(len(survivors))]

        # The new population is the survivors and their offspring
        pop[:] = survivors + offspring1 + offspring2 + offspring3

        # Prune trees that are too large BEFORE evaluation
        for i, ind in enumerate(pop):
            if len(ind) > MAX_NODE_COUNT:
                # replace obsese tree with new individual, continue random search
                pop[i] = toolbox.individual()

        # Save
        if gen % SAVE_EVERY_X_GEN == 0:
            # Save fossil record
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_id = f"fossils_v{VERSION_NUM}_gen{gen}_{cur_time}"
            save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.pkl" # Windows file path format
            with open(save_path, 'wb') as f:
                pickle.dump(fossil_record, f)

    print("--- Evolution Finished ---")

    # --- Save, Print, and Plot Results ---
    # Save fossil record
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_id = f"fossils_v{VERSION_NUM}_gen{N_GEN}_{cur_time}"
    save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.pkl" # Windows file path format
    with open(save_path, 'wb') as f:
        pickle.dump(fossil_record, f)

    print("\n--- Fossil Record (Best of Each Generation) ---")
    for gen, data in fossil_record.items():
        print(f"Gen {gen}: Fitness = {data['fitness']:.2f}")
        print(f"  Code: {str(data['individual'])}") # Uncomment to see the evolved code

    # Plotting
    gen_nums = logbook.select("gen")
    avg_fitness = logbook.chapters["fitness"].select("avg")
    max_fitness = logbook.chapters["fitness"].select("max")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen_nums, avg_fitness, "b-", label="Average Fitness")
    line2 = ax1.plot(gen_nums, max_fitness, "r-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Total Winnings (Fitness)")
    ax1.set_title("Poker Agent Fitness Over Generations")
    
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    
    plt.grid(True)
    fig_save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.png" # Windows file path format
    plt.savefig(fig_save_path)


if __name__ == "__main__":
    # Create the pool once, globally.
    # This will use all available CPU cores.
    num_procs = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_procs)
    
    # Make the pool and toolbox globally accessible to the wrapper function
    # Note: This often happens by default, but it's good to be explicit
    # if you run into issues.
    globals()['pool'] = pool
    globals()['toolbox'] = toolbox # Ensure toolbox is global

    try:
        main()
    except KeyboardInterrupt:
        print("Evolution stopped by user.")
    finally:
        # Clean up the worker processes
        pool.close()
        pool.join()
        print("--- Program exited ---")
