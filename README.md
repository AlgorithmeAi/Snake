# Algorithme Ai - Snake 🐍
**Author:** Charles Dana & Algorithme.ai

Snake is an **XAI (Explainable AI)** polynomial-time multiclass oracle. It provides high-accuracy classification while maintaining a full "Audit Trail" for every prediction, allowing you to understand *why* the model reached a specific conclusion through "lookalike" analysis and logical AND statements.

---

### 🚀 Performance
The Snake algorithm achieves a AutoML high accuracy score of **0.795** on the **Titanic Kaggle Dataset** challenge.

### 🔍 Explainability (XAI)
Unlike "black-box" models, Snake provides a full audit of its reasoning. It identifies "lookalikes" from the training set and displays the exact logical conditions that link them to the new data point.

**Example Audit Output:**
> **Predicted outcome:** Class [0] (98.88% probability)  
> **Reasoning:** Datapoint is a lookalike to Passenger #5 (Allen, Mr. William Henry) because:
> * The text field `Sex` does not contain [female]
> * The numeric field `Age` is between [34.0] and [37.0]
> * The numeric field `Pclass` is greater than [2.5]

---

### 🛠 Installation
To install the package in editable mode for local development:
```bash
pip install -e .
```

# 📖 Usage Example: Titanic Submission
The following script trains the model on 100 layers and generates a submission file for Kaggle.
```python
from algorithmeai import Snake

# 1. Initialize and train the Snake Oracle
# target_index=1 corresponds to the 'Survived' column
snake = Snake("titanic/train.csv", target_index=1, n_layers=100)

# 2. Prepare the test population from CSV
population = snake.make_population("titanic/test.csv")

# 3. Generate Submission and Audit results
with open("submission_snake.csv", "w") as f:
    f.write("PassengerId,Survived")
    
    for X in population:
        # Get probability for the positive class (1)
        # snake.get_probability(X) returns a dict or list indexed by class
        probability = snake.get_probability(X)[1]
        
        # Apply the optimized threshold (0.61) for the Titanic challenge
        prediction = 1 if probability > 0.61 else 0
        
        # Ensure PassengerId is written as a clean integer string
        p_id = str(int(float(X["PassengerId"])))
        
        f.write(f"\n{p_id},{prediction}")
        
        # Print XAI Audit trail to console for transparency
        print(snake.get_audit(X))
```
# 📊 Training Logs
During training, Snake provides real-time feedback on its progress and data analysis:
```
# Algorithme.ai : Snake Analysis on Survived a binary problem 0/1
# Algorithme.ai : Occurence Vector {0: 549, 1: 342}
# [Sex] text field | [Age] numeric field | [Pclass] numeric field ...
# Algorithme.ai : Layer 0/100, remainder 23.05s.
# Algorithme.ai : Layer 50/100, remainder 12.16s.
# Algorithme.ai : Layer 99/100, remainder 0.24s.
Safely saved to snakeclassifier.json
```

# 📜 License
This project is licensed under the MIT License.
