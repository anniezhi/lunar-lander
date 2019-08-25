<div align="center">

# Lunar Lander
</div>

<p align="center">
<img src="https://github.com/alanjeffares/lunar-lander/blob/master/lunar_lander.gif"  width="500">
</p>

This project compares three different approaches for beating Lunar Lander using OpenAI's [Gym](https://gym.openai.com/) environment. Firstly, I have fit a range of typical machine learning models to a large dataset of expert players states (e.g. co-ordinartes, velocity & leg status) and actions (e.g. fire left, right, up or do nothing). Secondly, I have fit a CNN to a large dataset of expert players images (screenshots of live play) and actions (e.g. fire left, right, up or do nothing) which are stored as the last digit of the filename. Finally, I have trained a Deep Q-Learning model by allowing it to play for half a million steps of gameplay and rewarding based on its score. 

All three approaches are saved and then analysed in a jupyter notebook. 

Try the game yourself by running `python play.py`!

## File Descriptions
* `LunarLanderStateVectors.csv` and the `data` folder contain the training data for the first two approaches respectively.
* `states_model.ipynb`, `images_model.ipynb` and `deepqlearning_model.ipynb` are the three notebooks in which the different approaches applied and `evaluation.ipynb` is where the resulting performances are compared.
* `task1.mod`, `task2.mod` and `dqn_LunarLander-v2_weights.h5f` are files containing the best performing models weights. 
* `lunar_lander_ml_images_player.py`, `lunar_lander_ml_states_player.py` and `lunar_lander_reinforcement_learning.py` are python files that play their respective models against the lunar lander game. 

## Getting Started
After cloning the repo simply run the following lines in terminal to unzip the (rather large) image data:
```
unzip data/data.zip
cd data; \rm -rf data*
```
Then run any of the `.py` files to watch the different models attempt to beat the game or check out the `ipynb` files to see the different training approaches and an evaluation of their performances. 
