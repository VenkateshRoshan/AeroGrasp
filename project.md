# AeroGrasp

Train a drone with a mounted robotic hand to autonomously fly to a target, grasp an object, and bring it back — all in PyBullet simulation.

## Pipeline

```
Drone+Hand  -->  Fly to target  -->  Pick object  -->  Return to start
```

## Training Stages

1. **Hover Stabilization** — Train the drone to hold a stable position in the air.
2. **Navigation** — Train the drone to fly to a specific (x, y, z) location and hold.
3. **Attach Hand** — Mount a robotic hand (URDF) on top of the drone.
4. **Joint Fine-Tuning** — Fine-tune the drone+hand system to handle changed dynamics (extra weight, shifted CoM).
5. **Grasp Training** — Train the hand to pick up target objects.
6. **End-to-End** — Fine-tune the full system: fly to target → grasp → return.
7. **Environment Transfer** — Swap in a realistic outdoor environment (grass/farm) and fine-tune.
