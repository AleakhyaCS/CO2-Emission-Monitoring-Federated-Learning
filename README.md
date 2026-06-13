# CO2-Emission-Monitoring-Federated-Learning
This project implements a federated learning framework to classify carbon emission categories across countries while keeping data decentralised. Countries are grouped to act as clients, training a local neural network with global standardisation. Models are aggregated using FedAvg via Flower, ensuring privacy and consistent global performance.

**Key features include:**

Data Preprocessing: Global standardization for consistent scaling across clients.

Federated Learning: Client models trained locally and aggregated using FedAvg.

Model Architecture: Neural network classifier for multi-class emission categories.

Simulation: Supports multiple rounds of federated training with Keras and Flower.

Evaluation: Tracks class distribution, accuracy, and loss on local and global datasets.

Tech Stack: Python, Keras, NumPy, Flower (FL framework), Scikit-learn, SHAP, Streamlit

Data Src: OWID-CO2 dataset from Our World in Data

🚨 Note: The API key is temporarily disabled due to limited credits (will be enabled upon request for demo)
[CO2 Policy App](https://co2-emission-monitoring-federated-learning.streamlit.app/)
