# System Diagram

## High-Level

```mermaid
classDiagram
    Grasp_Quality_Metric <|-- Grasp_Dataset: (grasps, labels)
    Grasp_Quality_Metric <|-- NeRF_Dataset: (nerf densities)
    Planner <|-- Grasp_Quality_Metric: neural network
    Evaluation <|-- Planner: planner
 
    class Grasp_Dataset{
      + ACRONYM Dataset: 2-finger grasps => success/fail 
      + DexGraspNet Dataset: 5-finger grasps, all success
      + DexGraspNet Pipeline:  N-finger grasps => success/fail
    }
    class NeRF_Dataset{
      + NeRF Data Collection in Isaac Gym
      + NeRF Training in torch-ngp
    }
    class Grasp_Quality_Metric{
      + Learned via NeRF inputs and Grasp Dataset
      + 2D CNN => 1D CNN => MLP architecture
    }
    class Planner{
      + Cross-Entropy Method to optimize Grasp Quality Metric
    }
    class Evaluation{
      + Isaac Gym environment
      + Pose uncertainty
      + Grasp Controller
    }
```

# Grasping Pipeline

## Mesh + Ferrari-Canny Pipeline

```mermaid
classDiagram
    Grasp_Optimizer <|-- Inputs: mesh
    Grasp_Controller <|-- Grasp_Optimizer: (rays_o*, rays_d*)

    class Grasp_Controller{
      + State-Machine PID Control
    }
    class Grasp_Optimizer{
      + Optimizer: CEM, Dice the Grasp, etc.
      + Metric: Ferrari-Canny
    }
    class Inputs{
      + Ground-Truth Mesh
    }
```

## NeRF + Ferrari-Canny Pipeline

```mermaid
classDiagram
    Grasp_Optimizer <|-- Inputs: nerf
    Grasp_Controller <|-- Grasp_Optimizer: (rays_o*, rays_d*)

    class Grasp_Controller{
      + State-Machine PID Control
    }
    class Grasp_Optimizer{
      + Optimizer: CEM, etc.
      + Metric: Ferrari-Canny
    }
    class Inputs{
      + NeRF
    }
```

## NeRF + Learned Metric Pipeline

```mermaid
classDiagram
    Grasp_Optimizer <|-- Inputs: nerf
    Grasp_Controller <|-- Grasp_Optimizer: (rays_o*, rays_d*)

    class Grasp_Controller{
      + State-Machine PID Control
    }
    class Grasp_Optimizer{
      + Optimizer: CEM, etc.
      + Metric: Learned w/ ACRONYM
    }
    class Inputs{
      + NeRF
    }
```

Additional ablation studies: use ground-truth mesh and NeRF as inputs, use one of each for sampling or metric.

# Learned Metric Network Architecture

```mermaid
classDiagram
    Density_Encoder <|-- Inputs: density cylinders
    Metric_Predictor <|-- Density_Encoder: density embeddings
    Metric_Predictor <|-- Inputs: centroid

    Learned_Metric <|-- Metric_Predictor: grasp success

    class Learned_Metric{
      + Grasp success [0, 1]
    }
    class Metric_Predictor{
      + MLP/Transformer
    }
    class Density_Encoder{
      + CNN
    }
    class Inputs{
      + NeRF density cylinders along rays_o, rays_d
      + NeRF centroid wrt these rays
    }
```

# NeRF Training Pipeline

```mermaid
classDiagram
    Img_Collection <|-- Inputs: mesh + material

    NeRF_Trainer <|-- Img_Collection: imgs, camera_poses

    NeRF <|-- NeRF_Trainer: model_weights

    class NeRF{
      + NeRF of Object
    }
    class NeRF_Trainer{
      + torch-ngp
    }
    class Img_Collection{
      + Isaac Gym Pics from Camera Poses
    }
    class Inputs{
      + Ground-Truth Mesh
      + Material for color & texture
    }
```
