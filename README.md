# 2D Heat Equation Solver using Physics-Informed Neural Network (PINN)

Physics-Informed Neural Networks (PINNs) allow us to solve complex differential equations without the need for labeled data, by embedding physical laws directly into the training process.
By leveraging known physical laws, PINNs enable data-efficient and interpretable learning for solving PDEs.

I trained a neural network using PyTorch to approximate the solution $u(x,y,t)$ of the heat diffusion PDE:

$${du \over dt} = \alpha \left({\partial^2u \over \partial x^2}+{\partial ^2u \over \partial y^2}\right)$$

# Problem Setup

* **Domain**: $(x, y, t) \in [0,1] \times [0,1] \times [0,T]$
* **Initial Condition**: $u(x,y,0)=sin(\pi x)cos(\pi y)$
* **Boundary Condition**: Zero heat flux at boundary (Neumann condition)

$${\partial u \over \partial n}=0$$
* **Diffusion Coefficient**: $\alpha = 0.1$

