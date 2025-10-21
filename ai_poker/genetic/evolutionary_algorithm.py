import operator
import random
import numpy as np
from deap import base, creator, tools, gp
import matplotlib.pyplot as plt
from ai_poker.mvp.poker_env import PokerEnv
from ai_poker.mvp.agents import RandomAgent, RandomAgent2
from ai_poker.genetic.simple_agents import ManiacAgent, StationAgent, SimpleValueAgent
from clubs_gym.agent.base import BaseAgent
from typing import Any
import clubs
import random

# --- 1. Define Primitives and Terminals ---

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

# Poker actions are mapped to a float:
# < 0.33 -> FOLD
# 0.33 to 0.66 -> CALL
# > 0.66 -> RAISE (the value can be used to scale the raise amount)
OUTPUT_TYPE = float

# The arguments for our evolved function. This defines the "senses" of our agent.
# We will map the gym's observation dictionary to these arguments.
ARGUMENT_TYPES = [
    float, # pot_size
    bool, # button (True for small blind, False for big blind)
    float, # stack_size
    float, # opponent_stack_size
    float, # amount_to_call
    float, # hand_strength (a value from 0.0 to 1.0)
    str,   # hand_class (e.g., 'PAIR', 'FLUSH')
    str,   # street (e.g., 'PREFLOP', 'FLOP')
]

# Create the Primitive Set
# This defines the building blocks (functions and terminals) for our program trees.
pset = gp.PrimitiveSetTyped("main", ARGUMENT_TYPES, OUTPUT_TYPE)

# --- Add Primitives (Operators) ---
# Logical operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
# pset.addPrimitive(operator.xor_, [bool, bool], bool)
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
# Rename arguments for clarity
pset.renameArguments(ARG0='pot_size', ARG1='button', ARG2='stack_size',
                    ARG3='opponent_stack_size', ARG4='amount_to_call',
                    ARG5='hand_strength', ARG6='hand_class', ARG7='street')

# Add float constants
FLOAT_CONSTANTS = [0.0, 0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
for val in FLOAT_CONSTANTS:
    pset.addTerminal(val, float)

# Add string constants for hand classifications
HAND_CLASSES = ['HIGH_CARD', 'PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'STRAIGHT',
                'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND', 'STRAIGHT_FLUSH']
for hand_class in HAND_CLASSES:
    pset.addTerminal(hand_class, str)

# Add string constants for streets
STREETS = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
for street in STREETS:
    pset.addTerminal(street, str)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

# --- 2. DEAP Toolbox Setup ---

# Define the fitness criteria: maximize winnings
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Define the individual: a program tree with the defined fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator:
# We use genHalfAndHalf to create a mix of full trees and growing trees,
# which promotes diversity in the initial population.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

# Structure initializers:
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# --- 3. Fitness Evaluation ---

# Helper function to map gym observations to our function's arguments
HAND_STRENGTH_MAP = {i: strength for i, strength in enumerate(np.linspace(0, 1, len(HAND_CLASSES)))}
STREET_MAP = {0: 'PREFLOP', 1: 'FLOP', 2: 'TURN', 3: 'RIVER'}


class ASTAgent(BaseAgent):
    def __init__(self, dealer: Any, position: int, ast: Any):
        self.dealer = dealer
        self.position = position
        self.ast = ast

    # Value agent bets according to their hand strength
    def act(self, obs: clubs.poker.engine.ObservationDict) -> int:
        # collect ast inputs (pot_size, button, stack_size, opponent_stack_size, amount_to_call, hand_strength, hand_class, street)
        hole_cards = obs['hole_cards']
        community_cards = obs['community_cards']

        pot_size = obs['pot']
        button = self.dealer.button == self.position
        stack_size = obs['stacks'][self.position]
        opponent_stack_size = obs['stacks'][self.position - 1]
        amount_to_call = obs['call']
        hand_strength = self.dealer.evaluator.evaluate(hole_cards, community_cards)
        hands_dict = self.dealer.evaluator.table.hand_dict
        
        if hand_strength < hands_dict['straight flush']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-1]
        elif hand_strength < hands_dict['four of a kind']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-2]
        elif hand_strength < hands_dict['full house']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-3]
        elif hand_strength < hands_dict['flush']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-4]
        elif hand_strength < hands_dict['straight']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-5]
        elif hand_strength < hands_dict['three of a kind']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-6]
        elif hand_strength < hands_dict['two pair']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-7]
        elif hand_strength < hands_dict['pair']['cumulative unsuited']:
            hand_class = HAND_CLASSES[-8]
        else: # High Card
            hand_class = HAND_CLASSES[-9]

        if len(obs['community_cards']) == 0:
            street = STREETS[0]
        elif len(obs['community_cards']) == 3:
            street = STREETS[1]
        elif len(obs['community_cards']) == 4:
            street = STREETS[2]
        else: # River - 5 community cards
            street = STREETS[3]

        # execute AST logic
        action = self.ast(pot_size, button, stack_size, opponent_stack_size, amount_to_call, hand_strength, hand_class, street)
        bet = action * pot_size

        return bet


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
    env.register_agents([ASTAgent(dealer=env.dealer, position=0, ast=agent1_logic), ASTAgent(dealer=env.dealer, position=1, ast=agent2_logic)])
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
            if game_over:
                break
            obs = env.reset()

    return winnings[0], winnings[1], num_hands


# Register the evaluation, selection, crossover, and mutation operators
toolbox.register("evaluate", evaluate_agents)
toolbox.register("select", tools.selBest) # Select the best X individuals
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Decorate crossover and mutation to prevent code bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))       # can we increase limit?
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# --- 4. The Main Evolutionary Loop ---

def main():
    random.seed(42)
    
    POP_SIZE = 50
    N_GEN = 1
    CXPB, MUTPB = 0.7, 0.2
    
    pop = toolbox.population(n=POP_SIZE)
    fossil_record = {}
    
    # Statistics to track
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + mstats.fields

    print("--- Starting Evolution ---")

    for gen in range(N_GEN):
        # --- Evaluate the entire population ---
        # Each individual plays against every other individual (round-robin)
        winnings_map = {i: 0 for i in range(len(pop))}
        num_hands_map = {i: 0 for i in range(len(pop))}
        
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                agent1_logic = toolbox.compile(expr=pop[i])
                agent2_logic = toolbox.compile(expr=pop[j])

                winnings1, winnings2, num_hands = toolbox.evaluate(agent1_logic, agent2_logic)
                
                winnings_map[i] += winnings1
                winnings_map[j] += winnings2
                num_hands_map[i] += num_hands
                num_hands_map[j] += num_hands

        # Assign fitness based on total winnings
        for i, ind in enumerate(pop):
            # # Assign total winnings as fitness
            # ind.fitness.values = (winnings_map[i],)
            # Assign win rate as fitness
            ind.fitness.values = (winnings_map[i] / num_hands_map[i] ,)

        # --- Log statistics ---
        record = mstats.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        print(logbook.stream)

        # --- Save the best of the generation ---
        best_ind = tools.selBest(pop, 1)[0]
        fossil_record[gen] = {
            'fitness': best_ind.fitness.values[0],
            'individual': str(best_ind)
        }

        # --- Selection ---
        num_survivors = POP_SIZE // 2
        survivors = toolbox.select(pop, k=num_survivors)
        
        # --- Create the next generation ---
        offspring = [toolbox.clone(ind) for ind in survivors]
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # The new population is the survivors and their offspring
        pop[:] = survivors + offspring

    print("--- Evolution Finished ---")

    # --- Save, Print, and Plot Results ---
    # Save fossil record
    import os
    import pickle
    import datetime

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_id = f"fossils_v0.1_{cur_time}"
    save_path = f"{os.path.dirname(__file__)}\\fossils\\{save_id}.pkl" # Windows file path format
    with open(save_path, 'wb') as f:
        pickle.dump(fossil_record, f)


    print("\n--- Fossil Record (Best of Each Generation) ---")
    for gen, data in fossil_record.items():
        print(f"Gen {gen}: Fitness = {data['fitness']:.2f}")
        print(f"  Code: {data['individual']}") # Uncomment to see the evolved code

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
    plt.show()


if __name__ == "__main__":
    main()
