# Cluster-Based Generalization in Reinforcement Learning  
### An Analysis on Atari Games using PPO Agents

This repository contains the code and configurations for my Bachelor's thesis:  
**"Cluster-Based Generalization in Reinforcement Learning: An Analysis on Atari Games"**  
submitted in May 2025 at Technical University of Darmstadt

The project investigates how different input representations and training strategies affect generalization in reinforcement learning, using PPO agents trained on three "shoot-em-ups" Atari games.

---

## Project Objective

The goal of this project is to evaluate how well reinforcement learning agents can generalize across similar tasks by:

- Training PPO agents on multiple Atari shooter games (Phoenix, Galaxian, Space Invaders)
- Using three types of input representations:
  - Raw pixels (RGB)
  - Simplified binary representation
  - Structured planes representation
- Comparing three multi-task training strategies:
  - Baseline (individual training)
  - Random Curriculum
  - Block Curriculum

This research analyze generalization performance across game clusters and investigate representational and strategic impacts.

---

## Install dependencies

pip install -r requirements.txt

---

## My Contribution

I developed the full code, designed and conducted the experiments, and analyzed the results as part of my Bachelor’s thesis.
