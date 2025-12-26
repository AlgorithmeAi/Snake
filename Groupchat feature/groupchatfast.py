import sys
import csv
import re
import pandas as pd
from algorithmeai import Snake

def convert_to_csv(input_file):
    """Converts WhatsApp txt export to a CSV format."""
    output_file = 'temp_chat.csv'
    # Pattern to match: [Date Time] Author: Message
    pattern = re.compile(r'^\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\] (.*?): (.*)')
    
    data = []
    current_author, current_date, current_text = None, None, ""

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            match = pattern.match(line)
            if match:
                if current_author:
                    data.append([current_author, current_text.strip(), current_date])
                current_date, current_author, current_text = match.groups()
            elif current_author:
                current_text += " " + line

        if current_author:
            data.append([current_author, current_text.strip(), current_date])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Author', 'Text', 'Date'])
        writer.writerows(data)
    return output_file

def prepare_limited_data(csv_file, train_limit=1000):
    """Splits data into a training set of exactly 1000 samples and puts the rest in backtest."""
    df = pd.read_csv(csv_file)
    # Shuffle the data
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    
    # Ensure we don't try to take more samples than exist
    actual_limit = min(train_limit, len(shuffled_df))
    
    train_df = shuffled_df.iloc[:actual_limit]
    backtest_df = shuffled_df.iloc[actual_limit:]
    
    train_df.to_csv("training.csv", index=0)
    backtest_df.to_csv("backtest.csv", index=0)
    print(f"Prepared training.csv with {len(train_df)} samples and backtest.csv with {len(backtest_df)} samples.")

def run_prediction():
    """Trains the Snake model and evaluates accuracy."""
    snake = Snake("training.csv")
    population = snake.make_population("backtest.csv")
    X_acc = []
    
    for X in population:
        # Check if the predicted author matches the actual author
        X_acc += [X["Author"] == snake.get_prediction(X)]
        accuracy = 100 * sum(X_acc) / len(X_acc)
        print(f"{accuracy:.2f}% accuracy")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Groupchat.py <chat_file.txt>")
    else:
        chat_path = sys.argv[1]
        
        # 1. Convert .txt to .csv
        temp_csv = convert_to_csv(chat_path)
        
        # 2. Split with 1000 train samples
        prepare_limited_data(temp_csv, train_limit=1000)
        
        # 3. Run prediction model
        run_prediction()
