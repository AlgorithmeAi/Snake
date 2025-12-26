# Algorithme AI - Snake Oracle
**Author:** Charles Dana & Algorithme.ai

Snake is an **XAI (Explainable AI)** polynomial-time multiclass oracle. It provides high-accuracy classification while maintaining a full "Audit Trail" for every prediction, allowing you to understand *why* the model reached a specific conclusion through "lookalike" analysis and logical AND statements.

---

### 🚀 Performance
The Snake algorithm achieves a high accuracy score of **0.795** on the **Titanic Kaggle Dataset** challenge.

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

### 📖 Usage Example: Titanic Submission
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

### 📊 Training Logs
During training, Snake provides real-time feedback on its progress and data analysis:
```bash
# Algorithme.ai : Snake Analysis on Survived a binary problem 0/1
# Algorithme.ai : Occurence Vector {0: 549, 1: 342}
# [Sex] text field | [Age] numeric field | [Pclass] numeric field ...
# Algorithme.ai : Layer 0/100, remainder 23.05s.
# Algorithme.ai : Layer 50/100, remainder 12.16s.
# Algorithme.ai : Layer 99/100, remainder 0.24s.
Safely saved to snakeclassifier.json
```

### 🔎 Audit feature
Snake can audit each and every one of its multiclass decisions, making it a reliable R.A.G. machine
```bash
### BEGIN AUDIT ###
        ### Datapoint {'Survived': 0, 'PassengerId': 892.0, 'Pclass': 3.0, 'Name': 'Kelly, Mr. James', 'Sex': 'male', 'Age': 34.5, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': '330911', 'Fare': 7.8292, 'Cabin': '', 'Embarked': 'Q'}
        ## Number of lookalikes 1076
        ## Predicted outcome (max proba) [0]
        
# Probability of being equal to class 0 : 98.88475836431226%
# Probability of being equal to class 1 : 1.1152416356877324%

        # Datapoint is a lookalike to #4 of class [0]
        - {'Survived': 0, 'PassengerId': 5.0, 'Pclass': 3.0, 'Name': 'Allen, Mr. William Henry', 'Sex': 'male', 'Age': 35.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': '373450', 'Fare': 8.05, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The text field Sex do not contains [female]
• The numeric field Age is more than [32.5]
• The numeric field Age is less than [39.0]
• The numeric field Pclass is more than [2.5]
• The numeric field Pclass is more than [2.5]
• The text field Sex do not contains [female]
• The numeric field Age is more than [34.5]
• The text field Name do not contains [Juha]
• The numeric field Age is less than [41.5]

        # Datapoint is a lookalike to #4 of class [0]
        - {'Survived': 0, 'PassengerId': 5.0, 'Pclass': 3.0, 'Name': 'Allen, Mr. William Henry', 'Sex': 'male', 'Age': 35.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': '373450', 'Fare': 8.05, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The numeric field Age is more than [34.0]
• The numeric field Age is less than [37.0]
• The numeric field SibSp is less than [0.5]
• The numeric field Pclass is more than [2.5]

        # Datapoint is a lookalike to #4 of class [0]
        - {'Survived': 0, 'PassengerId': 5.0, 'Pclass': 3.0, 'Name': 'Allen, Mr. William Henry', 'Sex': 'male', 'Age': 35.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': '373450', 'Fare': 8.05, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The numeric field Pclass is more than [2.0]
• The numeric field Age is more than [33.0]
• The text field Cabin do not contains [D56]
• The numeric field Age is less than [36.5]
• The text field Sex do not contains [female]

        # Datapoint is a lookalike to #5 of class [0]
        - {'Survived': 0, 'PassengerId': 6.0, 'Pclass': 3.0, 'Name': 'Moran, Mr. James', 'Sex': 'male', 'Age': 0.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': '330877', 'Fare': 8.4583, 'Cabin': '', 'Embarked': 'Q'}
        
        Because of the following AND statement that applies to both
        
• The text field Embarked contains [Q]
• The text field Name contains [Mr.]
• The text field Ticket do not contains [382651]
• The numeric field Fare is less than [15.5396]
• The text field Ticket do not contains [367228]

        # Datapoint is a lookalike to #22 of class [1]
        - {'Survived': 1, 'PassengerId': 23.0, 'Pclass': 3.0, 'Name': 'McGowan, Miss. Anna Annie', 'Sex': 'female', 'Age': 15.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': '330923', 'Fare': 8.0292, 'Cabin': '', 'Embarked': 'Q'}
        
        Because of the following AND statement that applies to both
        
• The text field Embarked contains [Q]
• The numeric field Fare is less than [11.6896]
• The numeric field Age is more than [9.5]
• The numeric field Fare is more than [7.8146]

        # Datapoint is a lookalike to #44 of class [1]
        - {'Survived': 1, 'PassengerId': 45.0, 'Pclass': 3.0, 'Name': 'Devaney, Miss. Margaret Delia', 'Sex': 'female', 'Age': 19.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': '330958', 'Fare': 7.8792, 'Cabin': '', 'Embarked': 'Q'}
        
        Because of the following AND statement that applies to both
        
• The text field Embarked contains [Q]
• The numeric field Fare is less than [11.6896]
• The numeric field Age is more than [9.5]
• The numeric field Fare is more than [7.8146]

        # Datapoint is a lookalike to #67 of class [0]
        - {'Survived': 0, 'PassengerId': 68.0, 'Pclass': 3.0, 'Name': 'Crease, Mr. Ernest James', 'Sex': 'male', 'Age': 19.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': 'S.P. 3464', 'Fare': 8.1583, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The numeric field Fare is less than [10.125]
• The text field Name contains [James]

        # Datapoint is a lookalike to #67 of class [0]
        - {'Survived': 0, 'PassengerId': 68.0, 'Pclass': 3.0, 'Name': 'Crease, Mr. Ernest James', 'Sex': 'male', 'Age': 19.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': 'S.P. 3464', 'Fare': 8.1583, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The numeric field Fare is less than [9.24585]
• The text field Name contains [James]

        # Datapoint is a lookalike to #67 of class [0]
        - {'Survived': 0, 'PassengerId': 68.0, 'Pclass': 3.0, 'Name': 'Crease, Mr. Ernest James', 'Sex': 'male', 'Age': 19.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': 'S.P. 3464', 'Fare': 8.1583, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The numeric field Pclass is more than [2.0]
• The text field Sex do not contains [female]
• The text field Name contains [James]

        # Datapoint is a lookalike to #67 of class [0]
        - {'Survived': 0, 'PassengerId': 68.0, 'Pclass': 3.0, 'Name': 'Crease, Mr. Ernest James', 'Sex': 'male', 'Age': 19.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': 'S.P. 3464', 'Fare': 8.1583, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The text field Name contains [James]
• The numeric field Fare is less than [17.4854]
• The text field Ticket do not contains [33595]

        # Datapoint is a lookalike to #67 of class [0]
        - {'Survived': 0, 'PassengerId': 68.0, 'Pclass': 3.0, 'Name': 'Crease, Mr. Ernest James', 'Sex': 'male', 'Age': 19.0, 'SibSp': 0.0, 'Parch': 0.0, 'Ticket': 'S.P. 3464', 'Fare': 8.1583, 'Cabin': '', 'Embarked': 'S'}
        
        Because of the following AND statement that applies to both
        
• The numeric field Pclass is more than [1.5]
• The numeric field SibSp is less than [1.0]
• The text field Name contains [James]
• The text field Ticket do not contains [C.A.]
...
```

### 📜 License
This project is licensed under the MIT License.
