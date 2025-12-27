import sys
import csv
import re
import pandas as pd
from algorithmeai import Snake
from collections import defaultdict
import time

class ChatAnalyzer:
    def __init__(self):
        self.author_stats = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'confidences': [],
            'confused_with': defaultdict(int)
        })
        self.overall_accuracy = 0.0
        self.processed = 0
        
    def convert_to_csv(self, input_file):
        """Converts WhatsApp txt export to a CSV format."""
        output_file = 'temp_chat.csv'
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

    def prepare_limited_data(self, csv_file, train_limit=1000):
        """Splits data into training and backtest sets."""
        df = pd.read_csv(csv_file)
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        actual_limit = min(train_limit, len(shuffled_df))
        
        train_df = shuffled_df.iloc[:actual_limit]
        backtest_df = shuffled_df.iloc[actual_limit:]
        
        train_df.to_csv("training.csv", index=0)
        backtest_df.to_csv("backtest.csv", index=0)
        
        print(f"\n{'='*80}")
        print(f"📊 DATASET SPLIT")
        print(f"{'='*80}")
        print(f"Training samples: {len(train_df)}")
        print(f"Backtest samples: {len(backtest_df)}")
        print(f"Total messages: {len(df)}")
        
        # Show author distribution
        print(f"\n📝 Author Distribution in Training Set:")
        author_counts = train_df['Author'].value_counts()
        for author, count in author_counts.items():
            print(f"  {author}: {count} messages ({100*count/len(train_df):.1f}%)")
        print(f"{'='*80}\n")

    def print_header(self):
        """Print the dynamic table header."""
        print("\n" + "="*120)
        print(f"{'Author':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10} {'Avg Conf':<10} {'Most Confused With':<30}")
        print("="*120)

    def print_stats_row(self, author):
        """Print a single author's statistics row."""
        stats = self.author_stats[author]
        
        # Find most confused author
        if stats['confused_with']:
            most_confused = max(stats['confused_with'].items(), key=lambda x: x[1])
            confused_str = f"{most_confused[0]} ({most_confused[1]}x)"
        else:
            confused_str = "N/A"
        
        print(f"{author:<20} {stats['total']:<8} {stats['correct']:<8} "
              f"{stats['accuracy']:<9.2f}% {stats['avg_confidence']:<9.2f}% {confused_str:<30}")

    def print_overall_stats(self):
        """Print overall statistics."""
        print("="*120)
        print(f"🎯 OVERALL ACCURACY: {self.overall_accuracy:.2f}% ({self.processed} messages processed)")
        print("="*120)

    def update_display(self):
        """Clear screen and redraw the entire table."""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        self.print_header()
        
        # Sort authors by accuracy (descending)
        sorted_authors = sorted(
            self.author_stats.keys(),
            key=lambda a: self.author_stats[a]['accuracy'],
            reverse=True
        )
        
        for author in sorted_authors:
            self.print_stats_row(author)
        
        self.print_overall_stats()
        
        # Show top insights
        self.print_insights()

    def print_insights(self):
        """Print interesting insights from the analysis."""
        if self.processed < 10:
            return
            
        print(f"\n💡 INSIGHTS:")
        
        # Most predictable
        most_predictable = max(
            self.author_stats.items(),
            key=lambda x: x[1]['accuracy'] if x[1]['total'] > 5 else 0
        )
        if most_predictable[1]['total'] > 5:
            print(f"  🎖️  Most Predictable: {most_predictable[0]} ({most_predictable[1]['accuracy']:.1f}%)")
        
        # Least predictable
        least_predictable = min(
            self.author_stats.items(),
            key=lambda x: x[1]['accuracy'] if x[1]['total'] > 5 else 100
        )
        if least_predictable[1]['total'] > 5:
            print(f"  🎭 Most Mysterious: {least_predictable[0]} ({least_predictable[1]['accuracy']:.1f}%)")
        
        # Highest confidence
        highest_conf = max(
            self.author_stats.items(),
            key=lambda x: x[1]['avg_confidence'] if x[1]['total'] > 5 else 0
        )
        if highest_conf[1]['total'] > 5:
            print(f"  💪 Most Distinctive Style: {highest_conf[0]} ({highest_conf[1]['avg_confidence']:.1f}% avg confidence)")
        
        print()

    def create_ascii_bar_chart(self, title, data_dict, max_width=50, value_suffix="%"):
        """Create an ASCII horizontal bar chart."""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * 80)
        
        if not data_dict:
            lines.append("No data available")
            return "\n".join(lines)
        
        max_value = max(data_dict.values()) if data_dict.values() else 1
        if max_value == 0:
            max_value = 1
        
        # Sort by value descending
        sorted_items = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
        
        for name, value in sorted_items:
            bar_length = int((value / max_value) * max_width)
            bar = "█" * bar_length
            spaces = " " * (max_width - bar_length)
            lines.append(f"{name:<20} {bar}{spaces} {value:.1f}{value_suffix}")
        
        return "\n".join(lines)

    def create_confusion_heatmap(self):
        """Create a simple ASCII confusion matrix visualization."""
        lines = []
        lines.append("\n🔀 CONFUSION PATTERNS")
        lines.append("=" * 80)
        lines.append("(Shows who gets confused with whom - higher = more confusion)\n")
        
        authors = sorted(self.author_stats.keys())
        
        if len(authors) < 2:
            lines.append("Not enough authors for confusion analysis")
            return "\n".join(lines)
        
        # Header
        header = "Author          "
        for author in authors:
            header += f"{author[:8]:<10}"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Rows
        for actual_author in authors:
            row = f"{actual_author[:15]:<16}"
            confusions = self.author_stats[actual_author]['confused_with']
            max_confusion = max(confusions.values()) if confusions else 1
            
            for predicted_author in authors:
                if actual_author == predicted_author:
                    cell = "    -     "
                else:
                    confusion_count = confusions.get(predicted_author, 0)
                    if confusion_count == 0:
                        cell = "    .     "
                    else:
                        # Scale: . = 0, ░ = low, ▒ = medium, ▓ = high, █ = very high
                        intensity = confusion_count / max(max_confusion, 1)
                        if intensity < 0.2:
                            symbol = "░"
                        elif intensity < 0.4:
                            symbol = "▒"
                        elif intensity < 0.7:
                            symbol = "▓"
                        else:
                            symbol = "█"
                        cell = f"  {symbol} ({confusion_count:2d})  "
                    row += cell
            lines.append(row)
        
        return "\n".join(lines)

    def create_message_distribution_chart(self):
        """Create ASCII chart of message distribution."""
        message_counts = {author: stats['total'] for author, stats in self.author_stats.items()}
        return self.create_ascii_bar_chart(
            "📊 MESSAGE DISTRIBUTION",
            message_counts,
            max_width=40,
            value_suffix=" msgs"
        )

    def create_accuracy_chart(self):
        """Create ASCII chart of accuracy by author."""
        accuracy_data = {
            author: stats['accuracy'] 
            for author, stats in self.author_stats.items()
            if stats['total'] > 5
        }
        return self.create_ascii_bar_chart(
            "🎯 ACCURACY BY AUTHOR",
            accuracy_data,
            max_width=50,
            value_suffix="%"
        )

    def create_confidence_chart(self):
        """Create ASCII chart of average confidence by author."""
        confidence_data = {
            author: stats['avg_confidence'] 
            for author, stats in self.author_stats.items()
            if stats['total'] > 5
        }
        return self.create_ascii_bar_chart(
            "💪 AVERAGE CONFIDENCE BY AUTHOR",
            confidence_data,
            max_width=50,
            value_suffix="%"
        )

    def run_prediction(self):
        """Trains the Snake model and evaluates with live updates."""
        print("\n🐍 Training Snake AI Model...")
        start_time = time.time()
        snake = Snake("training.csv", n_layers=25, vocal=False)
        train_time = time.time() - start_time
        print(f"✅ Model trained in {train_time:.2f} seconds\n")
        
        print("🔮 Starting predictions...\n")
        time.sleep(1)
        
        population = snake.make_population("backtest.csv")
        total_correct = 0
        
        for i, X in enumerate(population):
            actual_author = X["Author"]
            
            # Get prediction with probability
            probabilities = snake.get_probability(X)
            predicted_author = snake.get_prediction(X)
            confidence = probabilities.get(predicted_author, 0) * 100
            
            is_correct = actual_author == predicted_author
            
            # Update stats
            self.author_stats[actual_author]['total'] += 1
            if is_correct:
                self.author_stats[actual_author]['correct'] += 1
                total_correct += 1
            else:
                self.author_stats[actual_author]['confused_with'][predicted_author] += 1
            
            self.author_stats[actual_author]['confidences'].append(confidence)
            self.author_stats[actual_author]['accuracy'] = (
                100 * self.author_stats[actual_author]['correct'] / 
                self.author_stats[actual_author]['total']
            )
            self.author_stats[actual_author]['avg_confidence'] = (
                sum(self.author_stats[actual_author]['confidences']) / 
                len(self.author_stats[actual_author]['confidences'])
            )
            
            self.processed = i + 1
            self.overall_accuracy = 100 * total_correct / self.processed
            
            # Update display every 5 messages or on last message
            if (i + 1) % 5 == 0 or i == len(population) - 1:
                self.update_display()
                time.sleep(0.05)  # Small delay for readability
        
        # Final summary
        self.print_final_summary(snake, population)

    def print_final_summary(self, snake, population):
        """Print final analysis summary with ASCII charts."""
        print(f"\n{'='*120}")
        print("📈 FINAL ANALYSIS COMPLETE")
        print(f"{'='*120}\n")
        
        # ASCII Charts
        print(self.create_message_distribution_chart())
        print(self.create_accuracy_chart())
        print(self.create_confidence_chart())
        print(self.create_confusion_heatmap())
        
        # Show some example predictions
        print(f"\n{'='*80}")
        print("🔍 Sample Predictions (last 3 messages):\n")
        for X in population[-3:]:
            actual = X["Author"]
            predicted = snake.get_prediction(X)
            proba = snake.get_probability(X)
            text_preview = X["Text"][:60] + "..." if len(X["Text"]) > 60 else X["Text"]
            
            status = "✅" if actual == predicted else "❌"
            print(f"{status} Message: \"{text_preview}\"")
            print(f"   Actual: {actual} | Predicted: {predicted} (confidence: {proba[predicted]*100:.1f}%)")
            print()
        
        print("💾 Model saved as 'snakeclassifier.json'")
        print("📊 Data files: training.csv, backtest.csv, temp_chat.csv")
        print(f"\n{'='*120}\n")
        
        # Save results to file
        self.save_results_to_file(snake, population)

    def save_results_to_file(self, snake, population):
        """Save complete analysis results to a text file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"whatsapp_analysis_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("WHATSAPP CHAT ANALYSIS - SNAKE AI\n")
            f.write("=" * 120 + "\n\n")
            
            # Timestamp
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Messages Analyzed: {self.processed}\n")
            f.write(f"Overall Accuracy: {self.overall_accuracy:.2f}%\n\n")
            
            # Author Statistics Table
            f.write("=" * 120 + "\n")
            f.write("AUTHOR STATISTICS\n")
            f.write("=" * 120 + "\n")
            f.write(f"{'Author':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10} {'Avg Conf':<10} {'Most Confused With':<30}\n")
            f.write("-" * 120 + "\n")
            
            sorted_authors = sorted(
                self.author_stats.keys(),
                key=lambda a: self.author_stats[a]['accuracy'],
                reverse=True
            )
            
            for author in sorted_authors:
                stats = self.author_stats[author]
                if stats['confused_with']:
                    most_confused = max(stats['confused_with'].items(), key=lambda x: x[1])
                    confused_str = f"{most_confused[0]} ({most_confused[1]}x)"
                else:
                    confused_str = "N/A"
                
                f.write(f"{author:<20} {stats['total']:<8} {stats['correct']:<8} "
                       f"{stats['accuracy']:<9.2f}% {stats['avg_confidence']:<9.2f}% {confused_str:<30}\n")
            
            f.write("\n")
            
            # ASCII Charts
            f.write(self.create_message_distribution_chart() + "\n\n")
            f.write(self.create_accuracy_chart() + "\n\n")
            f.write(self.create_confidence_chart() + "\n\n")
            f.write(self.create_confusion_heatmap() + "\n\n")
            
            # Insights
            f.write("=" * 120 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("=" * 120 + "\n\n")
            
            # Most predictable
            most_predictable = max(
                self.author_stats.items(),
                key=lambda x: x[1]['accuracy'] if x[1]['total'] > 5 else 0
            )
            if most_predictable[1]['total'] > 5:
                f.write(f"Most Predictable Author: {most_predictable[0]} ({most_predictable[1]['accuracy']:.1f}%)\n")
                f.write(f"  - {most_predictable[0]} has the most recognizable writing style\n")
                f.write(f"  - Messages from this author are correctly identified {most_predictable[1]['accuracy']:.1f}% of the time\n\n")
            
            # Least predictable
            least_predictable = min(
                self.author_stats.items(),
                key=lambda x: x[1]['accuracy'] if x[1]['total'] > 5 else 100
            )
            if least_predictable[1]['total'] > 5:
                f.write(f"Most Mysterious Author: {least_predictable[0]} ({least_predictable[1]['accuracy']:.1f}%)\n")
                f.write(f"  - {least_predictable[0]}'s style is the hardest to identify\n")
                f.write(f"  - This could indicate a more neutral/common writing style\n\n")
            
            # Highest confidence
            highest_conf = max(
                self.author_stats.items(),
                key=lambda x: x[1]['avg_confidence'] if x[1]['total'] > 5 else 0
            )
            if highest_conf[1]['total'] > 5:
                f.write(f"Most Distinctive Style: {highest_conf[0]} ({highest_conf[1]['avg_confidence']:.1f}% avg confidence)\n")
                f.write(f"  - When the model predicts {highest_conf[0]}, it's very confident\n")
                f.write(f"  - This author has unique linguistic patterns\n\n")
            
            # Sample predictions
            f.write("=" * 120 + "\n")
            f.write("SAMPLE PREDICTIONS (Last 5 Messages)\n")
            f.write("=" * 120 + "\n\n")
            
            for i, X in enumerate(population[-5:], 1):
                actual = X["Author"]
                predicted = snake.get_prediction(X)
                proba = snake.get_probability(X)
                text_preview = X["Text"][:80] + "..." if len(X["Text"]) > 80 else X["Text"]
                
                status = "CORRECT" if actual == predicted else "INCORRECT"
                f.write(f"Message {i}: [{status}]\n")
                f.write(f"Text: \"{text_preview}\"\n")
                f.write(f"Actual Author: {actual}\n")
                f.write(f"Predicted Author: {predicted}\n")
                f.write(f"Confidence: {proba[predicted]*100:.1f}%\n")
                
                # Show top 3 probabilities
                sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]
                f.write(f"Top Predictions: ")
                f.write(", ".join([f"{author} ({prob*100:.1f}%)" for author, prob in sorted_proba]))
                f.write("\n\n")
            
            # Footer
            f.write("=" * 120 + "\n")
            f.write("Analysis completed with Snake AI - Algorithme.ai\n")
            f.write("github.com/AlgorithmeAi/Snake\n")
            f.write("=" * 120 + "\n")
        
        print(f"📄 Results saved to: {filename}")
        return filename

def main():
    if len(sys.argv) < 2:
        print("Usage: python whatsapp.py <chat_file.txt>")
        print("\nExample: python whatsapp.py whatsapp_chat.txt")
        return
    
    chat_path = sys.argv[1]
    
    print(f"\n🚀 WhatsApp Chat Analysis with Snake AI")
    print(f"{'='*120}")
    print(f"📁 Input file: {chat_path}\n")
    
    analyzer = ChatAnalyzer()
    
    try:
        # 1. Convert .txt to .csv
        print("📝 Converting chat export to CSV...")
        temp_csv = analyzer.convert_to_csv(chat_path)
        
        # 2. Load and split data
        tmp_df = pd.read_csv(temp_csv)
        N = tmp_df.shape[0]
        
        analyzer.prepare_limited_data(temp_csv, train_limit=min(1000, 3 * N // 4))
        
        # 3. Run prediction with live updates
        analyzer.run_prediction()
        
    except FileNotFoundError:
        print(f"❌ Error: File '{chat_path}' not found.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
