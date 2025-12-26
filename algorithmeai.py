import json
from random import choice
from time import time

"""
When woring with strings of floating point, handles the mistakes by replacing the value to 0.0 when floating parse error
"""
def floatconversion(txt):
    try:
        result = float(txt)
        return result
    except ValueError:
        return 0.0
        
class Snake():
    def __init__(self, csv_path, n_layers=100, vocal=True, target_index=0, excluded_features_index=[]):
        self.log = """################################################################
#                                                              #
#    Algorithme.ai : Snake         Author : Charles Dana       #
#                                                              #
#    December 2025  -  A multiclass .csv handler O(mn^2)       #
#                                                              #
################################################################
"""
        self.population = 0
        self.header = 0
        self.target = 0
        self.targets = 0
        self.datatypes = 0
        self.clauses = []
        self.lookalikes = 0
        self.n_layers = n_layers
        self.vocal = vocal
        if ".csv" in csv_path:
            self.qprint(f"# Initiated Snake with {self.n_layers} and vocal mode {self.vocal} from csv {csv_path}")
            with open(csv_path, "r") as f:
                header = self.make_bloc_from_line(f.readlines()[0])
            with open(csv_path, "r") as f:
                rows = f.readlines()[1:]
            target_column = header[target_index]
            self.target = target_column
            train_columns = [header[i] for i in range(len(header)) if not i in (excluded_features_index + [target_index])]
            header_index = [target_index] + [i for i in range(len(header)) if not i in (excluded_features_index + [target_index])]
            self.header = [target_column] + train_columns
            self.qprint(f"# Analysis train columns {train_columns}")
            self.qprint(f"# Analysis header {self.header}")
            self.datatypes = []
            targets = [self.make_bloc_from_line(row)[target_index] for row in rows]
            universe = set("".join(targets))
            if sorted(list(set(targets))) == ["0", "1"]:
                self.datatypes = ["B"]
                self.targets = [int(trg) for trg in targets]
                self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a binary problem 0/1")
            elif sorted(list(set(targets))) == ["True", "False"]:
                self.datatypes = ["B"]
                self.targets = [int("T" in trg or "t" in trg) for trg in targets]
                self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a binary problem True/False")
            elif sorted(list(set(targets))) == ["TRUE", "FALSE"]:
                self.datatypes = ["B"]
                self.targets = [int("T" in trg or "t" in trg) for trg in targets]
                self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a binary problem TRUE/FALSE")
            elif [c for c in universe if not c in "0123456789"] == []:
                self.datatypes = ["I"]
                self.targets = [int("0" + trg) for trg in targets]
                unique_targets = sorted(list(set(targets)))
                label = "/".join(unique_targets)
                self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass integers problem {label}")
            elif [c for c in universe if not c in "+-.0123456789e"] == []:
                self.datatypes = ["N"]
                unique_targets = sorted(list(set(targets)))
                label = "/".join(unique_targets)
                self.targets = [floatconversion(trg) for trg in targets]
                self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass floating point problem {label}")
            else:
                unique_targets = sorted(list(set(targets)))
                label = "/".join(unique_targets)
                self.targets = targets
                self.qprint(f"# Algorithme.ai : Snake Analysis on {self.target} a multiclass text field problem {label}")
                self.datatypes = ["T"]
            occurences_vector = {targ : sum((trg == targ for trg in self.targets)) for targ in set(self.targets)}
            self.qprint(f"# Algorithme.ai : Occurence Vector {occurences_vector}")
            for t in range(1, len(self.header)):
                hi = header_index[t]
                dtt = "N"
                values = [self.make_bloc_from_line(row)[hi] for row in rows]
                universe = set("".join(values))
                if [c for c in universe if not c in "+-.0123456789e"] == []:
                    dtt = "N"
                else:
                    dtt = "T"
                if dtt == "N":
                    h = header[hi]
                    self.qprint(f"#\t[{h}] numeric field")
                if dtt == "T":
                    h = header[hi]
                    self.qprint(f"#\t[{h}] text field")
                self.datatypes += [dtt]
            self.qprint(f"# Analysis datatypes {self.datatypes}")
            pp = self.make_population(csv_path, drop=True)
            self.target = self.header[0]
            self.population = pp
            self.lookalikes = {str(l) : [] for l in range(len(self.population))}
            self.clauses = []
            t_0 = time()
            for i in range(self.n_layers):
                self.construct_layer()
                remainder = round((time() - t_0) * (self.n_layers - i) / (i + 1), 2)
                self.qprint(f"# Algorithme.ai : Layer {i}/{self.n_layers}, remainder {remainder}s.")
            self.to_json()
        if ".json" in csv_path:
            self.from_json(filepath)

    def qprint(self, txt):
        if self.vocal:
            print(txt)
        self.log += str(txt) + "\n"
        
    def to_json(self, fout="snakeclassifier.json"):
        snake_classifier = {
            "population" : self.population,
            "header" : self.header,
            "target" : self.target,
            "targets" : self.targets,
            "clauses" : self.clauses,
            "lookalikes" : self.lookalikes,
            "datatypes" : self.datatypes,
            "n_layers" : self.n_layers,
            "vocal" : self.vocal,
            "log" : self.log
        }
        with open(fout, "w") as f:
            f.write(json.dumps(snake_classifier, indent=2))
        self.qprint(f"Safely saved to {fout}")
    
    def from_json(self, filepath="snakeclassifier.json"):
        with open(filepath, "r") as f:
            loaded_module = json.load(f)
        self.population = loaded_module["population"]
        self.header = loaded_module["header"]
        self.target = loaded_module["target"]
        self.targets = loaded_module["targets"]
        self.clauses = loaded_module["clauses"]
        self.lookalikes = loaded_module["lookalikes"]
        self.n_layers = loaded_module["n_layers"]
        self.vocal = loaded_module["vocal"]
        self.datatypes = loaded_module["datatypes"]
        self.log = loaded_module["log"]
        self.qprint(f"# Algorithme.ai : Successful load from {filepath}")
        
    """
    Will parse a .csv line properly, 
    returning an array of string, handling triple quotes elegantly
    """
    def make_bloc_from_line(self, line):
        line = line.replace('\n', '')
        if '"' in line:
            quoted = False
            bloc = []
            txt = ''
            for c in line:
                if c == '"':
                    quoted = not quoted
                else:
                    if c == ',' and not quoted:
                        bloc += [txt]
                        txt = ''
                    else:
                        txt += c
            bloc += [txt]
            return bloc
        return line.split(',')

    """
    Will effectively parse any .csv properly formated by pandas
    Do not parse any document that has not been pandas .to_csv sent to this function
    Could have a fatal error over a poorly passed .csv (can handle special characters)
    """
    def read_csv(self, fname):
        if not '.csv' in fname:
            self.qprint("Algorithme.ai: Please input a .csv file")
            return 0, 0
        with open(fname, "r") as f:
            lines = f.readlines()
        header = self.make_bloc_from_line(lines[0])
        data = [self.make_bloc_from_line(lines[t]) for t in range(1, len(lines))]
        return header, data
    """
    Makes the population available to the user
    """
    def make_population(self, fname, drop=False):
        POPULATION = []
        data_header, data = self.read_csv(fname)
        mapping_table = {h : -1 for h in self.header}
        for h in mapping_table:
            if h in data_header:
                mapping_table[h] = min((t for t in range(len(data_header)) if data_header[t] == h))
        hashes = set()
        for row in data:
            item_hash = ""
            item = {}
            for i in range(len(self.header)):
                h = self.header[i]
                dtt = self.datatypes[i]
                if mapping_table[h] == -1:
                    if dtt in "NIB":
                        item[h] = 0
                    if dtt == "T":
                        item[h] = ""
                else:
                    if dtt in "IB":
                        item[h] = int(row[mapping_table[h]])
                    if dtt in "N":
                        item[h] = floatconversion(row[mapping_table[h]])
                    if dtt == "T":
                        item[h] = str(row[mapping_table[h]])
                if i > 0:
                    item_hash += str(item[h])
            if drop and item_hash in hashes:
                self.qprint(f"# Algorithme.ai : Dropped conflicting row {item}")
            if drop and not item_hash in hashes:
                hashes.add(item_hash)
                POPULATION += [item]
            if not drop:
                POPULATION += [item]
        return POPULATION
    
    """
    Will return
    - For text fields: words to be or not to be included
    - For numeric fields: splits to be greater or not to be greater
    """
    def oppose(self, T, F):
        candidates = [i for i in range(1, len(self.header)) if T[self.header[i]] != F[self.header[i]]]
        index = choice(candidates)
        h = self.header[index]
        if self.datatypes[index] == "T":
            if len(T[self.header[index]]) != len(F[self.header[index]]) and choice(["Do it", "Don't"]) == "Do it":
                return [index, (len(F[h]) + len(T[h])) / 2, len(T[h]) > len(F[h]), "TN"]
            pros = set()
            cons = set()
            for sep in [" ", "/", ":", "-"]:
                for label in T[h].split(sep):
                    pros.add(label.split("\'")[0].split('\"')[0])
                for label in F[h].split(sep):
                    cons.add(label.split("\'")[0].split('\"')[0])
            clean_pros = [label for label in pros if len(label) and len(label) < max(2,len(T[h])) and not label in F[h]]
            clean_cons = [label for label in cons if len(label) and len(label) < max(2,len(F[h])) and not label in T[h]]
            possibilities = [[index, label, False, "T"] for label in clean_pros] + [[index, label, True, "T"] for label in clean_cons]
            if len(possibilities):
                return choice(possibilities)
            else:
                if T[h] != F[h] and not T[h] in F[h]:
                    return [index, T[h], False, "T"]
                if T[h] != F[h] and not F[h] in T[h]:
                    return [index, F[h], True, "T"]
        if self.datatypes[index] == "N":
            return [index, (F[h] + T[h]) / 2, T[h] > F[h], "N"]
    """
    Will return:
    - True if the datapoint satisfies the literal
    - False if the datapoint misses the header value or do not satisfy the literal
    Robust.
    """
    def apply_literal(self, X, literal):
        index = literal[0]
        value = literal[1]
        negat = literal[2]
        datat = literal[3]
        if not self.header[index] in X:
            return False
        if datat == "TN":
            if negat == True:
                return value <= len(X[self.header[index]])
            if negat == False:
                return value > len(X[self.header[index]])
        if datat == "T":
            if negat == True:
                return not value in X[self.header[index]]
            if negat == False:
                return value in X[self.header[index]]
        if datat == "N":
            if negat == True:
                return value <= X[self.header[index]]
            if negat == False:
                return value > X[self.header[index]]
    """
    Applies an or Statement on the literals
    """
    def apply_clause(self, X, clause):
        for literal in clause:
            if self.apply_literal(X, literal) == True:
                return True
        return False
        
    """
    Constructs a minimal clause to discriminate F relative to Ts
    - True on all Ts
    - False on at least F
    - Minimal
    """
    def construct_clause(self, F, Ts):
        clause = [self.oppose(choice(Ts), F)]
        Ts_remainder = [T for T in Ts if self.apply_literal(T, clause[-1]) == False]
        while len(Ts_remainder):
            clause += [self.oppose(choice(Ts_remainder), F)]
            Ts_remainder = [T for T in Ts_remainder if self.apply_literal(T, clause[-1]) == False]
        i = 0
        while i < len(clause):
            sub_clause = [clause[j] for j in range(len(clause)) if i != j]
            minimal_test = False
            for T in Ts:
                if self.apply_clause(T, sub_clause) == False:
                    minimal_test = True
                    break
            if minimal_test:
                i += 1
            if minimal_test == False:
                clause = sub_clause
        return clause
    """
    Constructs a minimal SAT Instance for a target value
    """
    def construct_sat(self, target_value):
        Fs = [self.population[i] for i in range(len(self.population)) if self.targets[i] == target_value]
        Ts = [self.population[i] for i in range(len(self.population)) if self.targets[i] != target_value]
        sat = []
        while len(Fs):
            F = choice(Fs)
            clause = self.construct_clause(F, Ts)
            consequence = [i for i in range(len(self.population)) if self.targets[i] == target_value and self.apply_clause(self.population[i], clause) == False]
            Fs = [F for F in Fs if self.apply_clause(F, clause) == True]
            sat += [[clause, consequence]]
        return sat
    
    """
    Constructs a logical layer of lookalikes
    """
    def construct_layer(self):
        target_values = sorted(list(set(self.targets)))
        for target_value in target_values:
            lookalikes = {str(l) : [] for l in range(len(self.population)) if self.targets[l] == target_value}
            sat = self.construct_sat(target_value)
            for pair in sat:
                self.clauses += [pair[0]]
                for l in pair[1]:
                    lookalikes[str(l)] += [len(self.clauses) - 1]
            for l in lookalikes:
                self.lookalikes[str(l)] += [lookalikes[str(l)]]


    def get_plain_text_assertion(self, condition, l):
        plain_text_assertion = f"""
        # Datapoint is a lookalike to #{l} of class [{self.targets[int(l)]}]
        - {self.population[int(l)]}
        
        Because of the following AND statement that applies to both
        """
        clause = []
        for c in condition:
            clause += self.clauses[c]
        
        for literal in clause:
                index = literal[0]
                value = literal[1]
                negat = literal[2]
                datat = literal[3]
                if datat == "TN":
                    if negat == True:
                        plain_text_assertion += f"\n• The textfield {self.header[index]} has length less than [{value}]"
                    if negat == False:
                        plain_text_assertion += f"\n• The textfield {self.header[index]} has more than [{value}]"
                if datat == "T":
                    if negat == True:
                        plain_text_assertion += f"\n• The text field {self.header[index]} contains [{value}]"
                    if negat == False:
                        plain_text_assertion += f"\n• The text field {self.header[index]} do not contains [{value}]"
                if datat == "N":
                    if negat == True:
                        plain_text_assertion += f"\n• The numeric field {self.header[index]} is less than [{value}]"
                    if negat == False:
                        plain_text_assertion += f"\n• The numeric field {self.header[index]} is more than [{value}]"
        return plain_text_assertion

    def get_audit(self, X):
        lookalikes = self.get_lookalikes(X)
        probability = self.get_probability(X)
        audit = f"""### BEGIN AUDIT ###
        ### Datapoint {X}
        ## Number of lookalikes {len(lookalikes)}
        ## Predicted outcome (max proba) [{self.get_prediction(X)}]
        """
        for target in probability:
            audit += f"\n# Probability of being equal to class {target} : {100*probability[target]}%"
        for triple in lookalikes:
            audit += "\n" + self.get_plain_text_assertion(triple[2], triple[0])
            
        return audit

    """
    Predict the probability vector for a given X
    """
    def get_lookalikes(self, X):
        clause_bool = [self.apply_clause(X, clause) for clause in self.clauses]
        clauses_negated = [i for i in range(len(clause_bool)) if clause_bool[i] == False]
        lookalikes = []
        for l in self.lookalikes:
            for condition in self.lookalikes[l]:
                condition_satisfied = True
                for c_index in condition:
                    if not c_index in clauses_negated:
                        condition_satisfied = False
                        break
                if condition_satisfied:
                    lookalikes += [[int(l), self.targets[int(l)], condition]]
        return lookalikes
    """
    Gives the probability vector associated
    """
    def get_probability(self, X):
        target_values = sorted(list(set(self.targets)))
        lookalikes = self.get_lookalikes(X)
        if len(lookalikes) == 0:
            probability = {target_value : 1/len(target_values) for target_value in target_values}
        else:
            probability = {target_value : sum((triple[1] == target_value for triple in lookalikes)) / len(lookalikes) for target_value in target_values}
        return probability
    """
    Predicts the outcome for a datapoint
    """
    def get_prediction(self, X):
        probability = self.get_probability(X)
        pr_max = max((probability[target] for target in probability))
        prediction = [target for target in probability if probability[target] == pr_max][0]
        return prediction
    
    def get_augmented(self, X):
        Y = X.copy()
        Y["AAI - Lookalikes"] = self.get_lookalikes(X)
        Y["AAI - Probability"] = self.get_probability(X)
        Y["AAI - Prediction"] = self.get_prediction(X)
        return Y
        
            
                    
