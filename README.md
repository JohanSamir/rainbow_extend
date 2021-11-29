# Lifting veil

# How to run all experiments
1. Install the dependencies with `pip install requirements.txt`
2. Run all experiments with `python3 tmux-runs.py`
3. To test: python3 main.py --env="acrobot" --agent="rainbow" --initial_seed="1" --exp="clip_rewards"


In this work we argue ...

## Quick Start
To use the algorithms proposed in the Revisiting Rainbow paper, you need python3 installed, make sure pip is also up to date.  If you want to run the MinAtar experiments you should install it. To install MinAtar, please check the following paper ([Young et al., 2019][young]) and repository ([github][young_repo]): 

1. Clone the repo: 
```bash
https://github.com/JohanSamir/revisiting_rainbow
```
If you prefer running the algorithms in a virtualenv, you can do the following before step 2:

```bash
python3 -m venv venv
source venv/bin/activate
# Upgrade Pip
pip install --upgrade pip
```

2.  Finally setup the environment and install Revisiting Rainbow's dependencies
```bash
pip install -U pip
pip install -r revisiting_rainbow/requirements.txt
```

## References

[Hado van Hasselt, Arthur Guez, and David Silver. *Deep reinforcement learning with double q-learning*. 
In Proceedings of the Thirthieth AAAI Conference On Artificial Intelligence (AAAI), 2016.][hasselt]

[Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan
Horgan, Bilal Piot, Mohammad Azar, and David Silver. *Rainbow: Combining Improvements in Deep Reinforcement learning*.
In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.][Hessel]

[Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alexander
Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and
Shane Legg. *Noisy networks for exploration*. In Proceedings of the International Conference on
Representation Learning (ICLR 2018), Vancouver (Canada), 2018.][fortunato]

[Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, and Marc G. Bellemare.
*Dopamine: A Research Framework for Deep Reinforcement Learning*, 2018.][castro]

[Kenny Young and Tian Tian. *Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments*, 2019.][young]

[Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas. *Dueling network architectures for deep reinforcement learning*. In Proceedings of the 33rd International
Conference on Machine Learning, volume 48, pages 1995–2003, 2016.][wang]

[Vieillard, N., Pietquin, O., and Geist, M. Munchausen Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS), 2020.][Vieillard]

[fortunato]: https://arxiv.org/abs/1706.10295
[hasselt]: https://arxiv.org/abs/1509.06461
[wang]: https://arxiv.org/abs/1511.06581
[castro]: https://arxiv.org/abs/1812.06110
[Hessel]: https://arxiv.org/abs/1710.02298
[young]: https://arxiv.org/abs/1903.03176
[Vieillard]: https://arxiv.org/abs/2007.14430
[young_repo]: https://github.com/kenjyoung/MinAtar
[arXiv_rev]: https://arxiv.org/abs/2011.14826
[blog]: https://psc-g.github.io/posts/research/rl/revisiting_rainbow/
[video]: https://slideslive.com/38941329/revisiting-rainbow-promoting-more-insightful-and-inclusive-deep-reinforcement-learning-research

## Giving credit
If you use Revisiting Rainbow in your research please cite the following:

Johan S. Obando Ceron, & Pablo Samuel Castro (2020). Revisiting Rainbow: Promoting more insightful and inclusive deep reinforcement learning research. In Deep Reinforcement Learning Workshop, NeurIPS 2020. [*arXiv preprint:* ][arXiv_rev]

In BibTeX format:

```
@inproceedings{obando20revisiting,
  author = {Johan S. Obando Ceron and Pablo Samuel Castro},
  title = {Revisiting Rainbow: Promoting more insightful and inclusive deep reinforcement learning research},
  booktitle = {Deep Reinforcement Learning Workshop, NeurIPS 2020},
  year = 2020
}
```


