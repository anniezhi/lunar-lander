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


## Code implementations

* [Gram-Schmidt implementation](https://github.com/alanjeffares/elements-of-statistical-learning/blob/master/chapter-3/code/Gram-Schmidt.R) - My implementation of algorithm 3.1

* [Exercise 3.2 simulation experiment](https://github.com/alanjeffares/elements-of-statistical-learning/blob/master/chapter-3/code/exercise_3.2.R) - Simulation experiment for exercise 3.2


