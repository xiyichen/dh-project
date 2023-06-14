# Leveraging Motion Imitation in Reinforcement Learning for Biped Character

## Branches
Visualizing reference motion: demo
Train Single Motion with Mass curriculum: mass_curriculum
Train Single Motion without Mass curriculum: single_motion
Train One Policy for Multiple Motions with Skill Selector: skill_selector
Train Walking with Random Velocity and Heading: velocity_goal

## Installation
please clone a branch and follow the provided instruction [HOWTO](HOWTO.md) to set up the workflow.

## Demos
### Retargetting from Deepmimic to Bob
![Figure 1 - walk_ref](assets/walk_reference.gif)

Reference - Walking

![Figure 2 - run_ref](assets/run_reference.gif)

Reference - Running

![Figure 3 - jump_ref](assets/jump_reference.gif)

Reference - Jumping

![Figure 4 - punch_ref](assets/punch_reference.gif)

Reference - Punching

![Figure 5 - backflip_ref](assets/backflip_reference.gif)

Reference - Backflip

### Training Results: One Policy per Motion

![Figure 6 - walk](assets/walk.gif)

Training Results - Walk

![Figure 7 - walk_random_v_h](assets/walk_random_v_h.gif)

Training Results - Walk with Random Velocity and Heading

![Figure 8 - run](assets/run.gif)

Training Results - Run

![Figure 9 - jump](assets/jump.gif)

Training Results - Jump
