# WhatsApp Group Chat Analysis with Snake AI

Analyze your WhatsApp conversations with **Snake Algorithm**—a multiclass logical classifier that predicts "who said what" based on writing style patterns. Get detailed insights, accuracy metrics, and visual charts about your chat dynamics!

---

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Phase 1: Export Your Chat](#phase-1-export-your-chat)
- [Phase 2: System Setup](#phase-2-system-setup)
- [Phase 3: Run Analysis](#phase-3-run-analysis)
- [Understanding the Results](#understanding-the-results)
- [Output Files](#output-files)

---

## ✨ Features

- 🎯 **Author Prediction**: Identifies who wrote each message based on writing style
- 📊 **Live Statistics**: Real-time accuracy tracking during analysis
- 📈 **ASCII Visualizations**: Bar charts and confusion matrices
- 💡 **Insights**: Discover who's most predictable, mysterious, or distinctive
- 📄 **Detailed Report**: Auto-generated `.txt` file with complete analysis
- 🔮 **Confidence Scores**: See how certain the model is about each prediction

---

## 🚀 Quick Start

```bash
# 1. Export your WhatsApp chat (see Phase 1 below)
# 2. Install dependencies
pip install pandas

# 3. Run analysis
python whatsapp.py your_chat.txt
```

That's it! The script will automatically train a model and generate comprehensive statistics.

---

## 📱 Phase 1: Export Your Chat

To analyze your chat history, you must first export it from WhatsApp to your computer as a `.txt` file.

### On iPhone (iOS)

1. Open the WhatsApp group chat you want to analyze
2. Tap the **Group Name** at the top to open Group Info
3. Scroll to the bottom and tap **Export Chat**
4. Select **Without Media** (ensures smaller file size and better compatibility)
5. Choose **Save to Files** or email/AirDrop the file to your computer
6. Extract the `.zip` file if necessary to find your `.txt` chat file

### On Android

1. Open the WhatsApp group chat
2. Tap the **three dots (⋮)** in the top right corner
3. Tap **More** → **Export chat**
4. Select **Without Media**
5. Send the file to yourself via Email, Google Drive, or USB transfer
6. Save the file on your computer as a `.txt` file

### Expected Format

Your exported chat should look like this:
```
[01/12/2024 14:23:45] Alice: Hey everyone!
[01/12/2024 14:24:12] Bob: What's up?
[01/12/2024 14:25:03] Charlie: Nothing much, you?
```

---

## 🛠️ Phase 2: System Setup

### Required Files

Ensure you have these files in the same folder:

1. **`whatsapp.py`**: Main analysis script (enhanced version with charts)
2. **`algorithmeai.py`**: Core Snake AI logic
3. **`your_chat.txt`**: Your exported WhatsApp file

### Dependencies

The script requires `pandas` for data handling:

```bash
pip install pandas
```

**No other dependencies needed!** Snake AI uses only Python standard library.

---

## 🎮 Phase 3: Run Analysis

### Basic Usage

```bash
python whatsapp.py your_chat.txt
```

### What Happens During Analysis

1. **Conversion** (1-2 seconds)
   - Converts `.txt` export to structured CSV format
   - Parses timestamps, authors, and messages

2. **Data Split** (instant)
   - Shows author distribution
   - Splits into training (75%) and testing (25%) sets
   - Displays dataset statistics

3. **Model Training** (5-30 seconds depending on size)
   - Trains Snake AI model with 25 logical layers
   - Learns unique writing patterns for each author

4. **Live Predictions** (real-time updates)
   - Processes each test message
   - Updates accuracy statistics every 5 messages
   - Displays dynamic table with metrics

5. **Final Report** (automatic)
   - Generates ASCII visualizations
   - Saves detailed `.txt` report
   - Shows sample predictions

---

## 📊 Understanding the Results

### Live Statistics Table

During analysis, you'll see a real-time table:

```
====================================================================================================================
Author               Total    Correct  Accuracy   Avg Conf   Most Confused With            
====================================================================================================================
Alice                45       42       93.33%     87.50%     Bob (2x)
Bob                  38       31       81.58%     72.30%     Charlie (5x)
Charlie              29       20       68.97%     65.10%     Alice (6x)
====================================================================================================================
🎯 OVERALL ACCURACY: 83.04% (112 messages processed)
====================================================================================================================

💡 INSIGHTS:
  🎖️  Most Predictable: Alice (93.3%)
  🎭 Most Mysterious: Charlie (69.0%)
  💪 Most Distinctive Style: Alice (87.5% avg confidence)
```

### Metrics Explained

- **Total**: Number of messages from this author in test set
- **Correct**: Number of correctly predicted messages
- **Accuracy**: Percentage of correct predictions for this author
- **Avg Conf**: Average confidence (probability) when predicting this author
- **Most Confused With**: Which author this person is most often mistaken for

### ASCII Visualizations

#### 📊 Message Distribution
Shows how many messages each person sent:
```
📊 MESSAGE DISTRIBUTION
================================================================================
Alice                ████████████████████████████████ 45 msgs
Bob                  ██████████████████████████ 38 msgs
Charlie              ████████████████████ 29 msgs
```

#### 🎯 Accuracy by Author
Shows prediction accuracy for each person:
```
🎯 ACCURACY BY AUTHOR
================================================================================
Alice                ████████████████████████████████████████████████ 93.3%
Bob                  ██████████████████████████████████████████ 81.6%
Charlie              ██████████████████████████████████ 69.0%
```

#### 💪 Average Confidence
Shows how distinctive each person's style is:
```
💪 AVERAGE CONFIDENCE BY AUTHOR
================================================================================
Alice                ████████████████████████████████████████████ 87.5%
Bob                  ████████████████████████████████████ 72.3%
Charlie              ████████████████████████████████ 65.1%
```

#### 🔀 Confusion Heatmap
Shows which authors get confused with each other:
```
🔀 CONFUSION PATTERNS
================================================================================
(Shows who gets confused with whom - higher = more confusion)

Author          Alice     Bob       Charlie   
------------------------------------------------
Alice              -      ░ ( 2)    . ( 0)    
Bob             ▒ ( 3)      -      ▓ ( 5)    
Charlie         █ ( 6)   ▒ ( 3)       -      
```

**Legend**: 
- `.` = No confusion
- `░` = Low confusion (1-20%)
- `▒` = Medium confusion (20-40%)
- `▓` = High confusion (40-70%)
- `█` = Very high confusion (70%+)

### Insights

The analysis automatically identifies:

- **🎖️ Most Predictable**: Person with the most recognizable style (highest accuracy)
- **🎭 Most Mysterious**: Person hardest to identify (lowest accuracy)
- **💪 Most Distinctive**: Person with unique patterns (highest confidence)

---

## 📁 Output Files

After analysis, you'll find these files:

### Generated During Analysis

1. **`temp_chat.csv`**: Converted chat data
2. **`training.csv`**: Training dataset (75% of messages)
3. **`backtest.csv`**: Testing dataset (25% of messages)
4. **`snakeclassifier.json`**: Trained model (can be reused)

### Analysis Report

**`whatsapp_analysis_YYYYMMDD_HHMMSS.txt`**: Complete analysis report

Contains:
- ✅ Author statistics table
- ✅ All ASCII visualizations
- ✅ Detailed insights with explanations
- ✅ Sample predictions with confidence scores
- ✅ Top 3 predictions for each sample
- ✅ Timestamp and metadata

**Example report excerpt:**
```
============================================================
WHATSAPP CHAT ANALYSIS - SNAKE AI
============================================================

Analysis Date: 2025-12-27 15:45:30
Total Messages Analyzed: 112
Overall Accuracy: 83.04%

============================================================
AUTHOR STATISTICS
============================================================
[... detailed table ...]

[... ASCII charts ...]

============================================================
KEY INSIGHTS
============================================================

Most Predictable Author: Alice (93.3%)
  - Alice has the most recognizable writing style
  - Messages from this author are correctly identified 93.3% of the time

[... more insights ...]

============================================================
SAMPLE PREDICTIONS (Last 5 Messages)
============================================================

Message 1: [CORRECT]
Text: "lol yeah that's so true"
Actual Author: Alice
Predicted Author: Alice
Confidence: 92.3%
Top Predictions: Alice (92.3%), Bob (5.1%), Charlie (2.6%)

[... more samples ...]
```

---

## 🎯 Tips for Best Results

### Good Chat Characteristics

✅ **Multiple authors** (3-10 works best)  
✅ **Sufficient messages** (200+ total recommended)  
✅ **Varied conversations** (different topics, times)  
✅ **Distinct styles** (people with different writing habits)

### What Affects Accuracy

- 📝 **Writing style differences**: More unique = higher accuracy
- 📊 **Message length**: Longer messages are easier to identify
- 🔤 **Vocabulary**: Unique words help distinguish authors
- 😊 **Emoji usage**: Different emoji patterns are very distinctive
- ⏰ **Time patterns**: When people typically message

### Interpretation Guide

| Overall Accuracy | Interpretation |
|-----------------|----------------|
| 80%+ | 🟢 Excellent - Very distinct writing styles |
| 60-80% | 🟡 Good - Notable style differences |
| 40-60% | 🟠 Fair - Some similar styles |
| <40% | 🔴 Low - Very similar writing patterns |

---

## 🔧 Advanced Usage

### Adjust Model Complexity

Edit `whatsapp.py` line ~259 to change layers:

```python
snake = Snake("training.csv", n_layers=25, vocal=False)
#                                      ^^
#                           Change this: 10-100
```

- **Lower (10-15)**: Faster, simpler rules, may underfit
- **Default (25)**: Balanced speed and accuracy
- **Higher (50-100)**: More complex rules, may overfit small datasets

### Customize Train/Test Split

Edit line ~333 to adjust split:

```python
analyzer.prepare_limited_data(temp_csv, train_limit=min(1000, 3 * N // 4))
#                                                            ^^^^^^^
#                                                     75% for training
```

---

## 🐛 Troubleshooting

### "File not found"
- Check the file path and name
- Ensure the `.txt` file is in the same folder as the script

### "No module named 'pandas'"
```bash
pip install pandas
```

### Encoding errors
- Make sure your chat file is UTF-8 encoded
- Try exporting the chat again from WhatsApp

### Low accuracy (<50%)
- Check if authors have distinct writing styles
- Ensure enough messages per author (50+ recommended)
- Try increasing `n_layers` to 50

### Model training is slow
- Reduce `n_layers` to 15
- Reduce `train_limit` to 500

---

## 📚 How It Works

Snake AI uses **logical clause learning**:

1. **Feature extraction**: Analyzes text length, word patterns, character usage, sentence structure
2. **Clause construction**: Builds logical rules (e.g., "if message contains 'lol' AND length < 20 → likely Alice")
3. **Layer stacking**: Creates multiple layers of rules for robust prediction
4. **Lookalike matching**: Finds similar messages in training data for explanations

**Key advantage**: Unlike neural networks, Snake provides **explainable predictions** - you can see WHY it thinks a message is from someone!

---

## 📖 Related Files

- **`algorithmeai.py`**: Core Snake AI implementation
- **`README.md`**: General Snake AI documentation
- **`groupchat.md`**: This file

---

## 📝 License

© Charles Dana, December 2025  
Part of the Algorithme.ai Snake project

---

## 🌟 Example Use Cases

- 👥 **Group dynamics**: See who dominates conversations
- 🎮 **Fun challenges**: Guess who said what, then check results
- 📊 **Writing analysis**: Understand individual communication styles
- 🔍 **Style evolution**: Compare different time periods
- 🎭 **Personality insights**: Discover linguistic patterns

---

**Ready to analyze your chats? Export, run, and discover! 🚀**
