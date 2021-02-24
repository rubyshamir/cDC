import numpy as np

'''
Implementation of the continous Dice Coefficient (https://www.biorxiv.org/content/10.1101/306977v1.full.pdf)
"Continuous Dice Coefficient: a Method for Evaluating Probabilistic Segmentations"
Reuben R Shamir,Yuval Duchin, Jinyoung Kim, Guillermo Sapiro, and Noam Harel

Input:
A - ground-truth or gold-standard binary segmentation (expert labeled data; assumes values are 0 or 1)
B - segmentation probabilistic map (your algorithm's output; assumes values are between 0-1)

Author: Ruby Shamir (feedback is welcome at shamir.ruby at gmail)
'''

def continous_Dice_coefficient(A_binary, B_probability_map):

    AB = A_binary * B_probability_map
    c = np.sum(AB)/max(np.size(AB[AB>0]), 1)
    cDC = 2*(np.sum(AB))/(c*np.sum(A_binary) + np.sum(B_probability_map))

    return cDC

def Dice_coefficient(A_binary, B_binary):

    AB = A_binary * B_binary
    DC = 2*(np.sum(AB))/(np.sum(A_binary) + np.sum(B_binary))

    return DC

def simulate_probablistic_segmentation (start, end):
    x, y = np.meshgrid(np.linspace(start, end, 100), np.linspace(start, end, 100))
    d = np.array(np.sqrt(x * x + y * y))
    mu = 0.0
    sigma = 2.0
    segmentation_result = np.exp(-((d - mu) * (d - mu) / (2.0 * sigma * sigma)))
    segmentation_result[segmentation_result<0.01] = 0
    return segmentation_result


## compare Dice and continous Dice under simulated error #########
if __name__ == '__main__':

    # in this example we simulate a ground truth segmentation (circle) and probabilistic segmentation results (gaussian)
    # the we demonstrate the cDC is less sensitive for shifts in segmentation than Dice Coefficient

    all_cDice = list()
    all_Dice = list()

    start = -10
    end = 10
    segmentation_result = simulate_probablistic_segmentation (start, end)

    ground_truth_simulated = np.ones_like(segmentation_result)
    ground_truth_simulated[segmentation_result < 0.01] = 0

    cDC = continous_Dice_coefficient(ground_truth_simulated, segmentation_result)
    all_cDice.append(cDC)
    binary_segmentation_result = np.zeros_like(segmentation_result)
    binary_segmentation_result[segmentation_result > 0.01] = 1
    DC = Dice_coefficient(ground_truth_simulated, binary_segmentation_result)
    all_Dice.append(DC)
    step = 2
    for shift in range(0, 4):
        segmentation_result = np.hstack((segmentation_result, np.zeros((segmentation_result.shape[0],step))))
        segmentation_result = np.delete(segmentation_result, range(0, step),1)

        cDC = continous_Dice_coefficient(ground_truth_simulated, segmentation_result)
        all_cDice.append(cDC)

        binary_segmentation_result = np.zeros_like(segmentation_result)
        binary_segmentation_result[segmentation_result>0.01] = 1
        # when the input is binary continues Dice Coefficient returns the Dice Coefficient.
        DC = Dice_coefficient(ground_truth_simulated, binary_segmentation_result)
        all_Dice.append(DC)

    all_cDice = [str(round(val,2)) for val in all_cDice]
    all_Dice = [str(round(val, 2)) for val in all_Dice]

    print ('Shift errors of: (mm)')
    print ([str(round(i,2)) for i in range(0, 10, 2)])
    print('Reduced the continues Dice:')
    print (all_cDice)
    print('And the original Dice is:')
    print (all_Dice)

