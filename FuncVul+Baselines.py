# Added Necessary Libraries

from sklearn .metrics import  precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix
import os
import time
import pandas as pd
import torch
import ast 
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    RagTokenizer, RagSequenceForGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.utils import resample

def get_eval_metrics(eval_pred, threshold=0.5):
    """
        Compute standard classification metrics from model predictions and true labels.

        This function supports both binary and multiclass classification. For binary classification,
        a threshold is applied to the probability of the positive class to determine predicted labels.
        For multiclass classification, the predicted class is the one with the highest probability.

        Parameters:
            eval_pred (tuple): A tuple containing:
                - logits (np.ndarray or torch.Tensor): Raw model outputs before softmax.
                - labels (np.ndarray or torch.Tensor): Ground-truth labels.
            threshold (float, optional): Threshold for positive class prediction in binary classification.
                                         Defaults to 0.5.

        Returns:
            dict: A dictionary with the following keys:
                - 'eval_accuracy': Overall accuracy score.
                - 'eval_precision': Precision score (binary or macro-averaged).
                - 'eval_recall': Recall score (binary or macro-averaged).
                - 'eval_f1': F1 score (binary or macro-averaged).
                - 'eval_false_positives': Count of false positives (binary only, else None).
                - 'eval_false_negatives': Count of false negatives (binary only, else None).
                - 'eval_predictions': List of predicted labels.
                - 'eval_labels': List of true labels.
    """

    logits, labels = eval_pred

    # Convert inputs to tensors if necessary
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Determine if binary classification (i.e., shape[-1] == 2)
    is_binary = True if probabilities.shape[1] == 2 else False

    if is_binary:
        predictions = (probabilities[:, 1] >= threshold).long()
    else:
        predictions = torch.argmax(probabilities, dim=-1)

    # Convert to numpy for metric computation
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    # Compute metrics
    precision = precision_score(y_true, y_pred, average='binary' if is_binary else 'macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary' if is_binary else 'macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary' if is_binary else 'macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

  
    # Compute false positives and false negatives (only for binary classification)
    if is_binary:
        false_positives = int(((predictions == 1) & (labels == 0)).sum().item())
        false_negatives = int(((predictions == 0) & (labels == 1)).sum().item())
    else:
        false_positives = None
        false_negatives = None

    return {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
        'eval_false_positives': false_positives,
        'eval_false_negatives': false_negatives,
    }


def load_dataset(path):
    """
    Loads the JSON dataset for a given name and removes NaN rows.

    Args:
        path (str): Path of the dataset file (excluding path and extension).

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    df = pd.read_json(path, orient='records', lines=True)
    print(f"Loaded dataset with shape: {df.shape} and columns: {df.columns.tolist()}")
    return df.dropna()


def preprocess_feature(df, feature):
    """
    Removes rows with missing values in the specified feature column and resets index.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Feature column to check for NaNs.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # df_clean = df.dropna(subset=[feature]).reset_index(drop=True)
    # print("Cleaned Data Label Counts:\n", df_clean['label'].value_counts())

    data = df.dropna(subset=[feature])

    # Reset the index
    data.reset_index(drop=True, inplace=True)

    return data

def count_newlines(input_string):
    """
    Counts the number of newline characters in the input string.

    Args:
        input_string (str): The input text in which newline characters ('\\n') are to be counted.
                            If the input is not a string, it will be converted to a string.

    Returns:
        int: The number of newline characters found in the string.
    """
    return str(input_string).count("\n")


def flatten_code_chunks(df, feature):
    """
    Converts lists of code lines into a single string, joined by newlines.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Column name containing code snippets or code chunks.

    Returns:
        pd.DataFrame: DataFrame with flattened code strings.
    """
    if feature == "code_chunks":
        df[feature] = ["\n".join(code) for code in df[feature]]
    elif feature == "generic_code_chunks":
        df[feature] = ["\n".join(code) if count_newlines(code) < 2 else code for code in df[feature]]
    return df


class CodeVulnerabilityModel(nn.Module):
    def __init__(self, pretrained_model='microsoft/graphcodebert-base', num_classes=2):
        super(CodeVulnerabilityModel, self).__init__()

        # Load the GraphCodeBERT model for sequence classification using AutoModelForSequenceClassification
        self.graphcodebert = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                                                num_labels=num_classes)

        # CNN Layer
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Fully connected layer for classification
        self.fc1 = nn.Linear(64 * 256, 128)  # Adjust according to the input size after pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get the logits from GraphCodeBERT (classification head)
        outputs = self.graphcodebert.base_model(input_ids, attention_mask=attention_mask)
        # logits = outputs.logits  # Extract logits from the output
        hidden_states = outputs.last_hidden_state  # Extract hidden states


        # Change shape to (batch_size, channels, seq_len) for CNN
        x = hidden_states.permute(0, 2, 1)  # [batch_size, seq_len, 768] -> [batch_size, 768, seq_len]

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Flatten and pass through fully connected layers
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x



def get_model_and_tokenizer(model_name, num_class):
    """
    Returns the appropriate tokenizer and model based on the model name.

    Args:
        model_name (str): Identifier for the model architecture.

    Returns:
        tokenizer, model: Pretrained tokenizer and classification model.

    Raises:
        ValueError: If model name is not recognized.
    """
   
    if model_name == "FuncVul":
        model_path = "microsoft/graphcodebert-base"
    elif model_name == "CodeBERT":
        model_path = "microsoft/codebert-base"
    elif model_name == "CustomVulBERTa":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_class)
        return tokenizer, model
    elif model_name == "BERT":
        model_path = "bert-base-uncased"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_class)
    return tokenizer, model

class CodeDataset():
    """
    A custom PyTorch Dataset for tokenizing and loading code snippets with labels.

    This dataset is used for training and evaluating transformer-based models
    (e.g., CodeBERT, GraphCodeBERT) on code classification tasks.

    Args:
        codes (List[str]): A list of code snippets.
        labels (List[int]): A list of corresponding integer labels (e.g., 0 or 1).
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to process code snippets.
        max_length (int): The maximum sequence length for tokenization (default is 512).
    """
    def __init__(self, codes, labels, tokenizer, max_length=512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the total number of code samples in the dataset."""
        return len(self.codes)

    def __getitem__(self, index):
        """
        Retrieves and tokenizes a single sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - input_ids: Tensor of token ids (shape: [max_length])
                - attention_mask: Tensor indicating padded tokens (shape: [max_length])
                - labels: Tensor of the ground truth label
        """
        code = self.codes[index]
        label = self.labels[index]

        # Tokenize the input code
        encoding = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }



def train_model_batch(model, model_feature, dataloader, vul_loader, optimizer, criterion, device, verbose=True):
    """
    Trains the model for one epoch using the provided dataloader.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to train.

    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of training data.

    optimizer : torch.optim.Optimizer
        Optimizer to use for parameter updates.

    criterion : torch.nn.Module
        Loss function (e.g., CrossEntropyLoss).

    device : torch.device
        Device on which to perform training (CPU or GPU).

    verbose : bool, optional
        Whether to print batch-level training progress.

    Returns:
    --------
    float
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if verbose:
            print(f"Training batch {batch_idx + 1}/{len(dataloader)}")

        # Move inputs and labels to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch[model_feature].to(device)

        # Zero out previous gradientsasfgh,
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        # logits = outputs.logits

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


class CodeVulnerabilityDataset():
    def __init__(self, code_snippets, labels, tokenizer, max_len=512):
        self.code_snippets = code_snippets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.code_snippets)

    def __getitem__(self, item):
        code = self.code_snippets[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def five_fold_performance(dataset, feature, model_name):
    """
    Performs 5-fold cross-validation on the specified dataset using the given model.

    Args:
        data_path (str): Path of Dataset.
        feature (str): Feature column to be used (e.g., 'code_chunks').
        model_name (str): Pretrained model name to use for classification.

    Outputs:
        Saves a CSV file containing fold-wise evaluation metrics and predictions.
    """
   
    data_path = 'Dataset/Dataset_' + str(dataset) + '.json' 
    df = load_dataset(data_path)
    df_clean = preprocess_feature(df, feature)
    print("Data Shape (For Model Build):", df_clean.shape)

    seed = 42

    results = pd.DataFrame(columns=[
        'model', 'K_fold', 'accuracy', 'precision', 'recall', 'f1',
        'FP', 'FN','train_time'
    ])

    print(f"\n{'*' * 90}\nRunning {model_name} on feature '{feature}'\n{'*' * 90}")

    random_df = df_clean.sample(frac=1, random_state=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    label_feature = 'label'
    num_class = max(random_df['label'].values) + 1 # Find the maximum number of classes

    print("Number of Class:", num_class)

    tokenizer, model = get_model_and_tokenizer(model_name, num_class= num_class)

    for fold, (train_idx, test_idx) in enumerate(kf.split(random_df)):
        print(f"\nFold {fold + 1}:")

        train_df = random_df.iloc[train_idx]
        test_df = random_df.iloc[test_idx]
        train_split, val_split = train_test_split(train_df, test_size=0.125, random_state=seed)

        # Preprocess code chunks
        train_split = flatten_code_chunks(train_split, feature)
        val_split = flatten_code_chunks(val_split, feature)
        test_df = flatten_code_chunks(test_df, feature)



        # Prepare datasets
        train_dataset = CodeDataset(train_split[feature].tolist(), train_split[label_feature].tolist(), tokenizer)
        val_dataset = CodeDataset(val_split[feature].tolist(), val_split[label_feature].tolist(), tokenizer)
        test_dataset = CodeDataset(test_df[feature].tolist(), test_df[label_feature].tolist(), tokenizer)

        # Trainer settings
        training_args = TrainingArguments(
            output_dir='./tmp',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=50,
            weight_decay=0.05,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            save_steps=0,
            eval_steps=100,
            load_best_model_at_end=True
        )



        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=get_eval_metrics
        )

        start = time.time()
        trainer.train()
        end = time.time()

        eval_metrics = trainer.evaluate(eval_dataset=test_dataset)

        print(f"Accuracy: {eval_metrics['eval_accuracy']:.4f}, "
              f"Precision: {eval_metrics['eval_precision']:.4f}, "
              f"Recall: {eval_metrics['eval_recall']:.4f}, "
              f"F1: {eval_metrics['eval_f1']:.4f}, "
              f"FP: {eval_metrics['eval_false_positives']}, "
              f"FN: {eval_metrics['eval_false_negatives']}")

        results.loc[len(results)] = [
            model_name, fold,
            eval_metrics['eval_accuracy'],
            eval_metrics['eval_precision'],
            eval_metrics['eval_recall'],
            eval_metrics['eval_f1'],
            eval_metrics['eval_false_positives'],
            eval_metrics['eval_false_negatives'],
            end - start
        ]

    # Save results
    save_dir = f"Results/FuncVul/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}Baseline_{model_name}_Results_{feature}_{dataset}.csv"
    results.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")

def fix_and_parse_list(x):
    if isinstance(x, str):
        # Replace spaces with commas and parse string to list
        x_fixed = x.replace(' ', ', ')
        return ast.literal_eval(x_fixed)
    else:
        # Already a list or other type, return as is
        return x
        
def five_fold_results(dataset, feature, model_name):
    """
    Computes and displays the average and standard deviation of performance metrics
    from five-fold cross-validation results for different models.

    Parameters:
    ----------
    dataset : str
        The name of the dataset used in the evaluation.
    feature : str
        The type of feature used in the evaluation (e.g., 'code_chunks' 0r generic_code_chunks).
    model_name: str . Baseline model name

    Behavior:
    --------
    - Loads the cross-validation result CSV file for the specified dataset and feature.
    - Computes the mean and standard deviation of key metrics (accuracy, precision,
      recall, F1-score, false positives, and false negatives) grouped by model.
    - Prints the transposed average and standard deviation tables for easy comparison.

    Expected CSV format:
    --------------------
    The CSV should contain the following columns:
    ['model', 'accuracy', 'precision', 'recall', 'f1', 'FP', 'FN']

    Example:
    --------
    five_fold_results(dataset = 1 , feature = "code_chunks")
    """

    file_path = f"Results/FuncVul/Baseline_{model_name}_Results_{feature}_{dataset}.csv"
    results = pd.read_csv(file_path)

    print(results)

    # Compute average metrics grouped by model
    avg_results = results[['model', 'accuracy', 'precision', 'recall', 'f1']].groupby(
        'model').mean().round(4).reset_index().T
    print("Average Results:\n", avg_results)

    # Compute standard deviation metrics grouped by model
    std_results = results[['model', 'accuracy', 'precision', 'recall', 'f1']].groupby(
        'model').std().round(4).reset_index().T
    print("\nStandard Deviation Results:\n", std_results)
    
  

if __name__ == '__main__':

    # Select Dataset 
    dataset = 4 # You can select [1 to 6]    
    feature = 'code_chunks' if dataset in [1,2,3] else 'generic_code_chunks'

    #**************************************************************************************************************
    # Run Proposed FuncVul model
    model_name = 'FuncVul'

    five_fold_performance(dataset, feature, model_name=model_name)  

    print("*"*50)
    print("Dataset : ", dataset, "Model : ", model_name, " Results: ")
    five_fold_results(dataset, feature , model_name=model_name)

    print("*"*50)

    #**************************************************************************************************************
    # Run Baselines model ['CodeBERT','CustomVulBERTa','BERT']

    model_name = 'CodeBERT' # You can change the modle name for 'CustomVulBERTa' or 'BERT'

    five_fold_performance(dataset, feature, model_name=model_name)   

    print("*"*50)
    print("Dataset : ", dataset, "Model : ", model_name, " Results: ")
    five_fold_results(dataset, feature , model_name=model_name)

    print("*"*50)

    #**************************************************************************************************************

