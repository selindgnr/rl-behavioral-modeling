# RL Behavioral Modeling: Exploration-Exploitation in Human Decision-Making

This project models human decision-making behavior in a **multi-armed bandit task** using reinforcement learning. It simulates how people balance *exploration* (trying new options) and *exploitation* (sticking with what works) — a core question in computational neuroscience and behavioral research.

## 🧠 Background

In many real-world decisions, people must choose between options with uncertain rewards. The **multi-armed bandit** is a classic paradigm to study this:
- A participant repeatedly chooses between N slot machines ("arms")
- Each arm has a hidden reward probability
- The goal: maximize total reward over time

How do people solve this? This project fits three computational models to simulated human behavior and compares their performance.

## 📊 Models Implemented

| Model | Description |
|---|---|
| **Random** | Baseline — chooses randomly |
| **Greedy** | Always picks the arm with highest observed reward |
| **Q-Learning (softmax)** | Learns reward values, explores via temperature parameter |

## 🗂️ Project Structure

```
rl-behavioral-modeling/
│
├── bandit_task.ipynb       # Main analysis notebook
├── bandit_task.py          # Core simulation & modeling code (importable)
├── requirements.txt        # Dependencies
└── README.md
```

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/selindgnr/rl-behavioral-modeling.git
cd rl-behavioral-modeling

# Install dependencies
python -m pip install -r requirements.txt

# Open the notebook
python -m notebook bandit_task.ipynb
```

## 📈 Key Results

- Q-learning with softmax exploration outperforms both random and greedy strategies
- Learning rate (α) and temperature (β) parameters interact to shape exploration behavior
- Simulated behavior qualitatively matches patterns observed in human participants

## 🔗 Relevance

This codebase is inspired by work at the **Max Planck Institute for Biological Cybernetics**, where similar models were applied to characterize exploration–exploitation dynamics in human behavioral experiments.

## Dependencies

`numpy` · `matplotlib` · `scipy` · `pandas` · `jupyter`
