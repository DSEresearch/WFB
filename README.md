## Learning Spatiotemporal Dynamics via Wave Function Backpropagation
Deep learning models have achieved strong performance through compositions of linear transformations and nonlinear activations, but their reliance on static representations limits their ability to capture continuous temporal dynamics. This limitation is particularly evident in natural language understanding, where meaning and emotion evolve over time. In this paper, we propose Wave Function Backpropagation (WFB), a novel framework that models neural activations as time-evolving wave functions rather than static vectors. By parameterizing representations with amplitude, wavenumber, angular frequency, and phase, WFB captures spatial and temporal dependencies in a unified form. Unlike conventional backpropagation, WFB incorporates temporal progression directly into the learning process. Experimental results on emotion classification with synthetic temporal augmentation show that WFB improves the modeling of temporal-semantic interactions and outperforms standard feedforward baselines.

<!-- > python .\WFB_GoEmostions.py --output_dir outputs -->
<!--
<img width="584" height="459" alt="laplacian" src="https://github.com/user-attachments/assets/b6303b1b-6f15-4683-a9fe-8c9a18b8315e" />
-->
## Model Comparison at λ = 10⁻⁴

<img width="2000" height="1200" alt="grouped_metrics" src="https://github.com/user-attachments/assets/e7499b00-0e15-4f64-b024-ae90aa0dd70f" />
