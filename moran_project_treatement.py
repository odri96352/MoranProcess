import random
import math
import numpy as np
from random import choice
from random import randint
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def select_individual(total_population):
    '''
    Returns a selected cell type for step one of the Moran process

    Parameters:
                total_population (dictionnary): all the information about all the cell types
    Return:
            output (string): name of selected cell type
    '''
    keys = list(total_population.keys())# name of all the cell types
    fitness_list = [total_population[keys[i]]["fitness"] for i in range(len(keys))] # list of fitness from all cell types
    proba_from_fitness = [ fitness_list[i]/sum(fitness_list) for i in range(len(fitness_list))] # list of probabilities that are proportional to the fitness
    output = np.random.choice(keys, p=proba_from_fitness)
    return output

def give_birth_new_cc(total_population, selected_individual):
    '''
    Returns if the selected indiviual gives birth to a cancer cell or not.

    Parameters:
                total_population (dictionnary): all the information about all the cell types
                selected_individual (string): output of select_individual() function, is a cell type name
    Return:
            output(boolean): TRUE if the selected cell gives birth to a NEW cancer clone
    '''
    select_mutation_rate = total_population[selected_individual]["mutation rate"]
    output = np.random.choice([True, False], p=[select_mutation_rate, 1-select_mutation_rate])
    print("output = "+str(output))
    return output

def new_cc(cc_fitness, cc_mutation_rate):
    '''
    Returns a new cancer clone

    Parameters:

    Return:
            output (dictionnary): all the information regarding the new cancer clone
    '''
    return {"fitness": cc_fitness, "mutation rate": cc_mutation_rate, "population": 1, "cancer detected": False, "remaining treatement generation":0}

def select_individual_to_kill(total_population):
    '''
    Returns a cell type name from which an indiviual will be killed

    Parameters:
                total_population (dictionnary): all the information about all the cell types

    Return:
            output (string): cell type name
    '''
    keys = list(total_population.keys())
    total_cells = sum([total_population[key]["population"] for key in keys])
    weights = [total_population[key]["population"]/total_cells for key in keys]
    return np.random.choice(keys, p=weights)


def barplot(total_population, colors, step):
    '''
    Plots a barplot of each cell type with their population

    Parameters:
                total_population (dictionnary): all the information about all the cell types
                colors (list): list of HEX colors
    '''
    fig, ax = plt.subplots()
    cell_type_names = list(total_population.keys())
    counts = [total_population[cell_type_names[i]]["population"] for i in range(len(cell_type_names))]
    bar_labels = cell_type_names
    bar_colors = colors

    ax.bar(cell_type_names, counts, label=bar_labels, color=bar_colors)

    ax.set_ylabel('Number of cells')
    ax.set_title('Cell Type')
    ax.legend(title='Color legend')
    plt.savefig("/home/azarkua/Documents/2023-2024/biomaths5/barplots/"+str(step)+'.png',dpi=400)
    #plt.show()
    return

import glob
from PIL import Image
def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    frame_one.save("barplotgif.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

def plot_populations(total_population_past_steps, nb_step, cancer_detection_threshold, detect_cancer= False):
    '''
    Plots every cell type's population evolution through every step.

    Parameters:
                total_population_past_steps (dictionnary): List of all population sizes for all the past steps for all the cell types.
                nb_step (int): total number of steps in the Moran process
                cancer_detection_threshold (float): The cancer is detected if a cancer population goes over that threshold
    '''
    plt.figure(figsize=(50,50))
    keys = list(total_population_past_steps.keys())
    x = [ i for i in range(nb_step+1)] # adding 1 to take into account the initial conditions
    if detect_cancer:
        threshold = [cancer_detection_threshold for i in range(nb_step+1)]
        plt.plot(x,threshold, label = "detection threshold", c='red')

    for key in keys:
        y = total_population_past_steps[key]
        plt.plot(x,y, label = str(key))
    plt.legend()
    plt.show()
    return

def create_color_list(total_population):
    '''
    Returns a list of HEX colors. One color for each cell type in the total population.

    Parameters:
                total_population (dictionnary): all the information about all the cell types

    Return:
            colors (list): list of HEX colors
    '''
    colors = []
    for i in range(len(list(total_population.keys()))):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors

def update_population_memory_list(total_population_past_steps):
    '''
    Goes through the dictionnary of population sizes for past steps, and add the previous step's population size for the new step.
    Example, if the population was of 50 for normal cells in the last step, it will append a population of size 50 again for the new step
    for normal cells.


    Parameters:
                total_population_past_steps (dictionnary): List of all population sizes for all the past steps for all the cell types.

    Return:
            total_population_past_steps (dictionnary): dictionnary with updated list
    '''
    keys = list(total_population_past_steps.keys())
    for key in keys:
        last_step_population = total_population_past_steps[key][-1]
        total_population_past_steps[key].append(last_step_population)
    return total_population_past_steps

def cancer_detected(total_population, cancer_detection_threshold):
    '''
    Returns True if a new cancer cell population has gone over the threshold, and returns the list of cell types over the threshold that
    aren't already being subject to the treatement (newly detected cancer cell types only)
    Parameters:
                total_population (dictionnary): all the information about all the cell types
                cancer_detection_threshold (float): The cancer is detected if a cancer population goes over that threshold
    Return:
            output (list): A boolean indicating if a new cancer is detected, and if yes, returns a list of the cell type detected
    '''
    keys = list(total_population.keys())
    detection_boolean = False
    list_of_new_cell_types_detected = []
    for key in keys[1:]:
               # we dont take into account normal cells (keys[0])
        if total_population[key]["population"]> cancer_detection_threshold:
            if total_population[key]["cancer detected"] == False:
                detection_boolean = True
                list_of_new_cell_types_detected.append(key)
    return [detection_boolean,list_of_new_cell_types_detected]

def shanon_diversity(total_population,N):
    '''
    Returns the Shannon Diversity Index of our population
    Parameters:
                total_population (dictionnary): all the information about all the cell types
                N (int): total number of cells
    Return:
            output (float): Shanon diversity index
    '''
    keys = list(total_population.keys())
    pi_list = [(total_population[key]["population"]/N)*math.log(total_population[key]["population"]/N) for key in keys]
    return -sum(pi_list)

def recovery(total_population):
    '''
    Returns True if the only cell type remaining is "normal cells"
    Parameters:
                total_population (dictionnary): all the information about all the cell types
    Return:
            output (boolean): True if the only cell type remaining is "normal cells"
    '''
    keys=list(total_population.keys())
    if len(keys) == 1:
        if str(keys[0]) == "normal cells":
            return True
        else:
            print("ERROR: the only remaining cell type is "+ str(keys[0]))
            return False
    return False

def death(total_population):
    '''
    Returns True if the sum of all cancer cells is superior to the population of normal cells
    Parameters:
                total_population (dictionnary): all the information about all the cell types
    Return:
            output (boolean): True if the sum of all cancer cells is superior to the population of normal cells
    '''
    keys=list(total_population.keys())
    nc_population = total_population["normal cells"]["population"]
    sum_cc_population = sum( [total_population[keys[i]]["population"] for i in range(1, len(keys))] )
                                                                                    # starting at 1 to not take normal cells into account
    if sum_cc_population > nc_population:
            return True
    return False



def main():
    # nc stands for "normal cell"
    # cc stands for "cancer clone"
    N =int(input("Enter total population size:"))
    nc_fitness = 1
    nc_mutation_rate = 0.001
    cc_fitness = 10
    cc_mutation_rate = 0.005
    number_of_cancer_clones_created_so_far = 1
    nb_step = 1000
    cancer_detection_threshold = 0.1 * N
    treatment_efficiency = 0.99
    nb_of_treatement_generation = 4

    # initialisation of total cell population
    total_population = { "normal cells": {"fitness": nc_fitness, "mutation rate": nc_mutation_rate, "population": 0, "cancer detected": False, "remaining treatement generation":0},
                        "cancer clone 1": {"fitness": cc_fitness , "mutation rate": cc_mutation_rate , "population": 0, "cancer detected": False, "remaining treatement generation":0 }
                        }
    total_population["cancer clone 1"]["population"] = int(input("Enter cancer clone 1 population size:"))
    total_population["normal cells"]["population"] = N - total_population["cancer clone 1"]["population"]

    total_population_past_steps = { "normal cells": [ total_population["normal cells"]["population"] ],
                                    "cancer clone 1": [ total_population["cancer clone 1"]["population"] ]}
                                    # dictionnary of list of population size for each cell type at each step
                                    # used for plot purposes
    colors = create_color_list(total_population)

    # Start Moran process
    for step in range(nb_step):
        print("/n")
        print("Step "+str(step)+" of the Moran process")

        # Select an individual
        selected_individual = select_individual(total_population)
        print("selected cell type: "+str(selected_individual))
        if give_birth_new_cc(total_population,selected_individual):
            print("The selected individual gives birth to a new cancer clone")
            # The selected individual gives birth to a new cancer clone
            number_of_cancer_clones_created_so_far += 1
            total_population["cancer clone "+str(number_of_cancer_clones_created_so_far)]=new_cc(cc_fitness, cc_mutation_rate)
            print("New cancer clone "+str(number_of_cancer_clones_created_so_far))
            # Keep previous populations in the memory for plotting purposes
            total_population_past_steps = update_population_memory_list(total_population_past_steps)
            total_population_past_steps["cancer clone "+str(number_of_cancer_clones_created_so_far)]=[0 for i in range(step + 1)]+[1]
                                                                                                    # add a population of 0
                                                                                                    # for each previous step
                                                                                                    # plus the initial condition
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        else:
            print("Increment the frequency of the cell type of the individual that was selected")
            # Simply increment the frequency of the cell type of the individual that was selected
            total_population[selected_individual]["population"] += 1

            # Keep previous populations in the memory for plotting purposes
            total_population_past_steps = update_population_memory_list(total_population_past_steps)
            total_population_past_steps[selected_individual][-1] += 1

        # Choose a random individual and kill it
        killed_individual = select_individual_to_kill(total_population)
        total_population[killed_individual]["population"] -= 1
        print("Kill "+str(killed_individual))


        # Keep previous populations in the memory for plotting purposes
        total_population_past_steps[killed_individual][-1] += -1

        # If the cell type population reaches 0, destroy the cell type
        if total_population[killed_individual]["population"] == 0:
            print("at step "+str(step)+", the population "+str(killed_individual)+" has reached 0.")
            del total_population[killed_individual]

        # Update the status of previously detected cancer cells
        for cell_type in list(total_population.keys()):
            if total_population[cell_type]["cancer detected"] == True:
                # Check if the treatement has ended or not
                if total_population[cell_type]["remaining treatement generation"] > 0:
                    total_population[cell_type]["remaining treatement generation"] -= 1
                else:
                    print("Treatment on cancer cell "+str(cell_type)+" has ended")
                    total_population[cell_type]["cancer detected"] = False
                # Modify the fitness
                total_population[cell_type]["fitness"] *= treatment_efficiency
                if total_population[cell_type]["fitness"] < 1 :
                    # I consider that a fitness lower than 1 means
                    # the treatment has worked and the cancer cells
                    # has no fitness anymore
                    print("Cancer cell "+str(cell_type)+" has a null fitness")
                    total_population[cell_type]["fitness"] = 0
                    total_population[cell_type]["cancer detected"] = False

        #Check if a new cancer is detected
        if cancer_detected(total_population, cancer_detection_threshold)[0]:
            newly_detected_cell_type_list = cancer_detected(total_population, cancer_detection_threshold)[1]
            #Update the status of newly detected cancer total_cells
            for cell_type in newly_detected_cell_type_list:
                total_population[cell_type]["cancer detected"] = True
                total_population[cell_type]["remaining treatement generation"] =  nb_of_treatement_generation
                total_population[cell_type]["fitness"] *= treatment_efficiency
                print("Cancer cell "+str(cell_type)+" is newly detected and now has a fitness of "+str(total_population[cell_type]["fitness"]))

        #barplot(total_population, colors, step)

        if recovery(total_population):
            print("The patient has recovered")
            print("The patient has died of cancer")
            shanon_diversity_index = shanon_diversity(total_population,N)
            print("/////////")
            print("The Shanon diversity index of this Moran process is: "+ str(shanon_diversity_index))
            plot_populations(total_population_past_steps, step+1, cancer_detection_threshold,detect_cancer= True)
            break

        if death(total_population):
            print("The patient has died of cancer")
            shanon_diversity_index = shanon_diversity(total_population,N)
            print("/////////")
            print("The Shanon diversity index of this Moran process is: "+ str(shanon_diversity_index))
            plot_populations(total_population_past_steps, step+1, cancer_detection_threshold,detect_cancer= True)
            break

        if step == nb_step-1 :
            print("The patient has died of natural death")
            shanon_diversity_index = shanon_diversity(total_population,N)
            print("/////////")
            print("The Shanon diversity index of this Moran process is: "+ str(shanon_diversity_index))
            plot_populations(total_population_past_steps, step+1, cancer_detection_threshold,detect_cancer= True)

    #make_gif("/home/azarkua/Documents/2023-2024/biomaths5/barplots/")
    # shanon_diversity_index = shanon_diversity(total_population,N)
    # print("/////////")
    # print("The Shanon diversity index of this Moran process is: "+ str(shanon_diversity_index))
    # plot_populations(total_population_past_steps, nb_step, cancer_detection_threshold,detect_cancer= True)
    return



if __name__ == '__main__':
    main()
