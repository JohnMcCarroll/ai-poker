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
import math
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
    VERSION_NUM,
    PROB_IMMIGRATION,
    PROB_CROSSOVER,
    PROB_MUTATION,
    PROB_PRUNE,
    MIN_HANDS,
    HAND_NUM_STEP_SIZE,
    GEN_CURRICULUM_STEP_SIZE,
    LOG,
    EVALUATION_TIMEOUT,
    ELITE_PCT,
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
    return 0 # Return a low value if invalid

def get_size(individual):
    """Returns the first fitness value (the win rate)."""
    if individual.fitness.valid:
        return individual.fitness.values[1]
    return 0 # Return a low value if invalid

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
            if LOG:
                print(f"Warning: Unknown agent type in eval: {type(agent_tree_or_class)}", file=log_file)
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
        if LOG:
            print(f"Error evaluating task ({i} vs {j}): {e}", file=log_file)
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
    while num_hands < max_hands:

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
                # reset stacks and deal next hand
                obs = env.reset(reset_stacks=True)
            else:
                # deal next hand
                obs = env.reset()

    return winnings[0], winnings[1], num_hands

def uniform_prune(individual, max_size):
    """
    Prunes an individual (PrimitiveTree) down to max_size by repeatedly applying
    deap.tools.mutShrink until the size constraint is met.

    Args:
        individual (deap.gp.PrimitiveTree): The tree to be pruned.
        pset (deap.gp.PrimitiveSetTyped): The primitive set used for the tree.
        max_size (int): The maximum allowed size (number of nodes).
        
    Returns:
        deap.gp.PrimitiveTree: The pruned individual.
    """
    
    # Apply mutShrink repeatedly until max_size is reached or no more shrinking is possible.
    while len(individual) > max_size and len(individual) > 1:
        # mutShrink replaces a subtree with one of its children, guaranteeing size reduction.
        # It returns the modified individual (or tuple if k=1, which is the default for mutShrink)
        individual, = gp.mutShrink(individual)
            
    return individual

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
creator.create("FitnessMax", base.Fitness, weights=(WIN_RATE_FITNESS_WEIGHT, 1.0))
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
toolbox.register("prune", uniform_prune, max_size=MAX_NODE_COUNT) 

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

    # Add curriculum: more evaluation for later evolution generations
    hand_num_curriculum = {i:num_hands for i,num_hands in enumerate(range(MIN_HANDS, MAX_HANDS + HAND_NUM_STEP_SIZE, HAND_NUM_STEP_SIZE))}

    pop = toolbox.population(n=POP_SIZE)
    fossil_record = {}

    print("--- Starting Evolution ---")
    if LOG:
        print("--- Starting Evolution ---", file=log_file)

    # Create bench
    # --- Task Group: Benchmark (Pop vs. Bench) ---
    static_bench = [SimpleValueAgent, StationAgent, ManiacAgent, RandomAgent, RandomAgent2]
    num_static_bots = len(static_bench)
    multiplier = EVALUATION_BENCH_SIZE // num_static_bots
    full_bench = static_bench*multiplier # populate eval bench with static bots
    bench_names = [str(i) for i in full_bench]
    bench_gen_nums = [0 for _ in full_bench]
    
    for gen in range(N_GEN):        
        # --- 1. Prepare all evaluation tasks ---
        tasks = []

        curriculum_step = math.floor(gen/GEN_CURRICULUM_STEP_SIZE)
        if curriculum_step > max(hand_num_curriculum.keys()):
            curriculum_step = max(hand_num_curriculum.keys())
        num_hands = hand_num_curriculum[curriculum_step]
        
        for i in range(len(pop)):
            for j, opponent in enumerate(full_bench):
                # We send the raw tree (pop[i]) and the opponent (class or tree)
                # We use 'None' as the 'j' index to mark it as a bench match
                tasks.append((pop[i], opponent, num_hands, i, j))
        
        # # --- 2. Run all tasks in parallel ---
        async_results = []
        for task in tasks:
            # Schedule the job
            res = pool.apply_async(run_evaluation, args=(task,))
            async_results.append(res)

        # # --- 3. Process results ---
        winnings_map = {i: 0 for i in range(len(pop))}
        num_hands_map = {i: 0 for i in range(len(pop))}
        
        bench_winnings_map = {i: 0 for i in range(len(full_bench))}
        bench_num_hands_map = {i: 0 for i in range(len(full_bench))}

        for i, res in enumerate(async_results):
            try:
                # Crucial: This enforces the timeout per job
                result = res.get(timeout=EVALUATION_TIMEOUT) 
                # Process result...
                # This loop receives results as soon as a worker finishes one task
                i, j, w1, w2, n_hands = result
                
                # Update scores for individual i
                winnings_map[i] += w1
                num_hands_map[i] += n_hands
                
                # Update scores for bench players
                if j is not None:
                    bench_winnings_map[j] += w2
                    bench_num_hands_map[j] += n_hands
            except multiprocessing.TimeoutError:
                # Handle the hung worker and continue
                print("Worker timed out!")
                if LOG:
                    print("Worker timed out!", file=log_file)
            except Exception as e:
                # Handle the crashed worker and continue
                print(f"Worker crashed!\n{e}")
                if LOG:
                    print(f"Worker crashed!\n{e}", file=log_file)
        
        # --- 4. Assign fitness  ---
        raw_win_rates = [0]*POP_SIZE
        for i, ind in enumerate(pop):
            if num_hands_map[i] > 0:
                # Fitness values for win rates are in (BB/100 hands)
                raw_win_rates[i] = 50 * winnings_map[i] / num_hands_map[i]
            else:
                raw_win_rates[i] = 0
                print("WARNING: individual player had no matches")
                if LOG:
                    print("WARNING: individual player had no matches", file=log_file)
        
        win_rate_std = np.std(raw_win_rates)
        for i, ind in enumerate(pop):
            raw_win_rate = raw_win_rates[i]
            
            # 1. Calculate Normalized Size: Penalty is 0 for trees size 1 (min)
            # Ensure division by zero is avoided if MAX_NODE_COUNT is small
            node_count = len(ind)
            normalized_size = (node_count - 1) / (MAX_NODE_COUNT - 1)
            
            # 2. Calculate Dynamic Penalty
            # The penalty scales with population variability and individual size.
            size_penalty = NODE_COUNT_FITNESS_WEIGHT * win_rate_std * normalized_size

            ind.fitness.values = (raw_win_rate, size_penalty)

        # calc bench win rate
        bench_win_rate_map = {i: {'win_rate': 0.0, 'opponent': None} for i in range(len(full_bench))}
        highest_win_rate = {'win_rate': 0.0, 'opponent_name': None, 'opponent_index': None}
        lowest_win_rate = {'win_rate': 999999999999999, 'opponent_name': None, 'opponent_index': None}
        for i, ind in enumerate(full_bench):
            if bench_num_hands_map[i] > 0:
                win_rate = 50 * bench_winnings_map[i] / bench_num_hands_map[i]
                bench_win_rate_map[i]['win_rate'] = win_rate
                bench_win_rate_map[i]['opponent'] = ind
                if win_rate > highest_win_rate['win_rate']:
                    highest_win_rate['win_rate'] = win_rate
                    highest_win_rate['opponent_index'] = i
                    highest_win_rate['opponent_name'] = bench_names[i]
                if win_rate < lowest_win_rate['win_rate']:
                    lowest_win_rate['win_rate'] = win_rate
                    lowest_win_rate['opponent_index'] = i
                    lowest_win_rate['opponent_name'] = bench_names[i]
            else:
                print("WARNING: bench player had no matches")
                if LOG:
                    print("WARNING: bench player had no matches", file=log_file)

        if VERBOSE:
            print(f'Best bench player generation {gen}: win rate: {highest_win_rate["win_rate"]}, bench player: {highest_win_rate["opponent_name"]}, tenure: {bench_gen_nums[highest_win_rate["opponent_index"]]}')
            print(f'Worst bench player generation {gen}: win rate: {lowest_win_rate["win_rate"]}, bench player: {lowest_win_rate["opponent_name"]}, tenure: {bench_gen_nums[lowest_win_rate["opponent_index"]]}')
            if LOG:
                print(f'Best bench player generation {gen}: win rate: {highest_win_rate["win_rate"]}, bench player: {highest_win_rate["opponent_name"]}, tenure: {bench_gen_nums[highest_win_rate["opponent_index"]]}', file=log_file)
                print(f'Worst bench player generation {gen}: win rate: {lowest_win_rate["win_rate"]}, bench player: {lowest_win_rate["opponent_name"]}, tenure: {bench_gen_nums[lowest_win_rate["opponent_index"]]}', file=log_file)

        # --- Log statistics ---
        win_rate_stats = stats.compile(pop)
        size_stats = stats2.compile(pop)
        record = {'gen': gen, **win_rate_stats, **size_stats}

        logbook.record(**record)
        gen_stats = logbook.stream
        print(gen_stats)
        if LOG:
            print(gen_stats, file=log_file)

        # --- Save the best of the generation to the fossil record
        best_ind = tools.selBest(pop, 1)[0]
        fossil_record[gen] = {
            'fitness': best_ind.fitness.values[0],
            'individual': best_ind,
            'gen': gen
        }

        # update bench (lowest winner gets replaced)
        full_bench[lowest_win_rate['opponent_index']] = best_ind
        bench_names[lowest_win_rate['opponent_index']] = f'Fossil Gen {gen}'
        for i, tenure in enumerate(bench_gen_nums):
            bench_gen_nums[i] += 1
        bench_gen_nums[lowest_win_rate['opponent_index']] = 0

        print(bench_gen_nums)
        if LOG:
            print(bench_gen_nums, file=log_file)

        # # --- Visualize generation's best individual ---
        if VERBOSE:
            lineage = getattr(best_ind, "lineage", "none")
            print(f'Best Individual of Generation {gen} lineage: {lineage}')
            if LOG:
                print(f'Best Individual of Generation {gen} lineage: {lineage}', file=log_file)

        # # --- Selection ---
        # ----------------------------------------------------------------------------------
        # 1. REPRODUCTION: Standard Generational Model with Elitism and Probabilistic Operators
        # ----------------------------------------------------------------------------------

        # a. Elitism: Copy the best individuals (attempt to guarantee performance never drops)
        # Note: We assume the fitness evaluation for the current 'pop' has just finished.

        num_elites = int(ELITE_PCT*POP_SIZE)
        elites = tools.selBest(pop, num_elites)
        offspring = []
        for elite in elites:
            elite.lineage = "Elite"
            offspring.append(toolbox.clone(elite))

        # b. Calculate the number of individuals to be created via breeding/immigration
        n_to_create = POP_SIZE - len(offspring)

        n_immigrants = int(POP_SIZE * PROB_IMMIGRATION)
        # Ensure we don't exceed the slots we need to fill
        n_breeding = n_to_create - n_immigrants
        if n_breeding < 0:
            n_breeding = 0
            n_immigrants = n_to_create

        # d. Select parents for breeding (n_breeding needed, selected from the entire population)
        parents = toolbox.select(pop, n_breeding)
        parents = list(map(toolbox.clone, parents)) # Clone selected parents

        # e. Apply Crossover and Mutation Probabilistically
        new_children = []
        for child1, child2 in zip_longest(parents[::2], parents[1::2], fillvalue=None):
            
            # --- Crossover ---
            if child2 and random.random() < PROB_CROSSOVER: # Check PCX (e.g., 80%)
                toolbox.mate(child1, child2)
                child1.lineage = "Crossover"
                child2.lineage = "Crossover"
                # enforce max node count
                if len(child1) > MAX_NODE_COUNT:
                    toolbox.prune(child1)
                    child1.lineage = lineage + "Prune"
                if len(child2) > MAX_NODE_COUNT:
                    toolbox.prune(child2)
                    child2.lineage = lineage + "Prune"
                # Invalidate fitness after modification
                del child1.fitness.values
                del child2.fitness.values
            
            # --- Mutation ---
            # Apply mutation to both children with PMUT probability (e.g., 10%)
            if random.random() < PROB_MUTATION:
                toolbox.mutate(child1)
                lineage = getattr(child1, "lineage", "")
                child1.lineage = lineage + "Mutation"
                if len(child1) > MAX_NODE_COUNT:
                    toolbox.prune(child1)
                    child1.lineage = lineage + "Prune"
                if hasattr(child1.fitness, 'values'):
                    del child1.fitness.values
            
            if child2 and random.random() < PROB_MUTATION:
                toolbox.mutate(child2)
                lineage = getattr(child2, "lineage", "")
                child2.lineage = lineage + "Mutation"
                if len(child2) > MAX_NODE_COUNT:
                    toolbox.prune(child2)
                    child2.lineage = lineage + "Prune"
                if hasattr(child2.fitness, 'values'):
                    del child2.fitness.values

            if random.random() < PROB_PRUNE:
                gp.mutShrink(child1)
                lineage = getattr(child1, "lineage", "")
                child1.lineage = lineage + "Prune"
                if hasattr(child1.fitness, 'values'):
                    del child1.fitness.values
            
            if child2 and random.random() < PROB_PRUNE:
                gp.mutShrink(child2)
                lineage = getattr(child2, "lineage", "")
                child2.lineage = lineage + "Prune"
                if hasattr(child2.fitness, 'values'):
                    del child2.fitness.values
                
            new_children.append(child1)
            if child2:
                new_children.append(child2)

        # f. Immigration: Add new random individuals to maintain diversity
        immigrants = [toolbox.individual() for _ in range(n_immigrants)]
        for ind in immigrants:
            ind.lineage = "Immigration"

        # g. Combine all groups to form the new generation
        offspring.extend(new_children)
        offspring.extend(immigrants)

        # Finalize the new population list (truncate if necessary due to odd numbers in breeding)
        pop[:] = offspring[:POP_SIZE]

        # Replace trees that are too large BEFORE evaluation
        for i, ind in enumerate(pop):
            if len(ind) > MAX_NODE_COUNT:
                # replace bloated tree with new individual, continue random search
                # pruning operations should avoid this
                new_ind = toolbox.individual()
                new_ind.lineage = "Immigration"
                pop[i] = new_ind

        # Save
        if gen % SAVE_EVERY_X_GEN == 0:
            # Save fossil record
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_id = f"fossils_v{VERSION_NUM}_gen{gen}_{cur_time}"
            save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.pkl" # Windows file path format
            with open(save_path, 'wb') as f:
                pickle.dump(fossil_record, f)

    print("--- Evolution Finished ---")
    if LOG:
        print("--- Evolution Finished ---", file=log_file)

    # --- Save, Print, and Plot Results ---
    # Save fossil record
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_id = f"fossils_v{VERSION_NUM}_gen{N_GEN}_{cur_time}"
    save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.pkl" # Windows file path format
    with open(save_path, 'wb') as f:
        pickle.dump(fossil_record, f)

    print("\n--- Fossil Record (Best of Each Generation) ---")
    if LOG:
        print("\n--- Fossil Record (Best of Each Generation) ---", file=log_file)
    for gen, data in fossil_record.items():
        print(f"Gen {gen}: Fitness = {data['fitness']:.2f}")
        print(f"  Code: {str(data['individual'])}")
        if LOG:
            print(f"Gen {gen}: Fitness = {data['fitness']:.2f}", file=log_file)
            print(f"  Code: {str(data['individual'])}", file=log_file)

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
    if LOG:
        # Create a unique filename based on the current date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"evolution_log_{timestamp}.txt"
        
        # Check if a log directory exists
        log_dir = "evolution_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_filepath = os.path.join(log_dir, log_filename)
        
        print(f"Logging all output to: {log_filepath}")
        
        log_file = open(log_filepath, 'w')

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
        if LOG:
            print("--- Program exited ---", file=log_file)
