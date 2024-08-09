# This is the readme for the models associated with the paper:

Qianming Ding, et al. Elimination of reentry spiral waves using adaptive optogenetical illumination based on dynamic learning techniques  (2024) 

## main codes

global_AOI.ipynb generates the article Figures 2 and 3, you can change the alpha in the "Parameters" and the comments "(2) Depolarization currents" or "(3) Hyperpolarization currents" in the "Simulation".

local_AOI.ipynb generates the article Figures 4 and 5, you can change the alpha and  abla_r in the "Parameters" and the comments "(2) Depolarization currents" or "(3) Hyperpolarization currents" in the "Simulation".

discrete_AOI.ipynb generates the article Figures 6 and 7, you can change the alpha,  abla_r and LED_len in the "Parameters" and the comments "(2) Depolarization currents" or "(3) Hyperpolarization currents" in the "Simulation".

ChR2_model.ipynb generates the article Figures 8, you can change the train_len, alpha,  abla_r and LED_len in the "Parameters".

## base codes

channel.py includes four-state Markov model of ChR2.

node.py includes 4 models for Luo-Rudy, Fenton-Karma, Hodgkin–Huxley and FitzHugh-Nagumo.

plt.py includes drawing-related functions

method.py is our core innovation code. It contains the diffusion system generation class, two PS localization methods, synchronization factor, AES & AOI array, and most importantly DLS technique.

## init data

Initial values for the four models. You can use the spiral_generation.ipynb to recreate them.

## .gif data

.gif of article pictures.