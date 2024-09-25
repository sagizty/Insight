import os
import torch
import timm
import torch.nn as nn
from tqdm import tqdm
from timm.layers import SwiGLUPacked
from transformers import GPT2Tokenizer, GPT2Model, ViTModel
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import numpy as np
import re
# ------------------- Dataset Class for VQA -------------------
# Preprocessing function to clean up the questions and answers
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(' +', ' ', text).replace(" ?", "?").strip()
    return text


class VQA_Dataset(Dataset):
    def __init__(self, hf_dataset, tokenizer_name='gpt2',
                 max_seq_length=256, img_size=224, transform=None, answer_to_index=None):

        self.data = hf_dataset
        self.img_size = img_size
        self.max_seq_length = max_seq_length

        # ------------------- Image Preprocessing Function -------------------
        # default_transform is only resize and to tensor
        default_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        # specified slide_feature image Transform can be assigned
        self.transform = transform or default_transform

        # ------------------- Text Preprocessing Function -------------------
        # The GPT-2 model operates on tokenized input, which is essentially converting text into sequences of
        # integers that represent individual tokens (words or subwords).
        self.answer_to_index = answer_to_index
        # Calling tokenizer ensures that the input text is properly formatted and tokenized in the same way
        # the GPT-2 model was trained, which is critical for effective performance.
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2') if tokenizer_name == 'gpt2' else None
        # Padding is used to ensure that all input sequences in a batch are of the same length.
        # EOS (End of sequence model) make the end of a seq to let model know input text is done
        self.tokenizer.pad_token = self.tokenizer.eos_token  # pad with eos, (use eos_token as pad_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # fetch question, answer and image path from the dataset
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']

        # Load the image from Hugging Face dataset
        image = self.data[idx]['image'].convert('RGB')  # Convert CMYK to RGB if needed

        # regardless of greyscale or RGB, Convert to RGB as transformer expects RCG 3 channel input
        img_tensor = self.transform(image)

        # Tokenize the question using GPT-2 tokenizer
        inputs = self.tokenizer(
            question,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length
        )

        # Map the processed answer to an integer index for classification
        if answer in self.answer_to_index:
            answer_idx = self.answer_to_index[answer]
        else:
            # Handle missing or unknown answers by assigning a default valid class
            answer_idx = 0  # Or any valid class index

        return {
            'image': img_tensor,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(answer_idx, dtype=torch.long)
        }


# Custom Collate Function for Batch stacking
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item['labels'] for item in batch])

    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# model and modules
# ------------------- Image Encoder (ViT) -------------------
# todo this will be called from PuzzleAI.ModelBase.ROI_models.Get_ROI_model
# Pre-processed image tensor is passed through the Vision Transformer (ViT), to obtain image embedding (ViT CLS token)
class ImageEncoder(nn.Module):
    def __init__(self, embed_size=768):
        super(ImageEncoder, self).__init__()

        # Pre-trained Vision Transformer (ViT)
        self.Image_Encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.embed_convert = nn.Linear(self.Image_Encoder.embed_dim, embed_size) \
            if self.Image_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, images):
        # Process image through Image_Encoder to get the embeddings
        Image_cls_embedding = self.Image_Encoder(images)  # CLS token output from ViT [B,D]
        return self.embed_convert(Image_cls_embedding)


# ------------------- Text Encoder (GPT-2) -------------------
# todo this will be called from PuzzleAI.ModelBase.Get_Language_model
# After tokenisation, the query (question tokens) is passed through the GPT-2 model,
# generating a sequence of hidden states (intermediate representations of input text after learning)
# The last CLS token from the last hidden state from the sequence is selected as the question's vector representation.
# A dropout layer is applied to the text embeddings to prevent overfitting.

class TextEncoder(nn.Module):
    # this obtains the question embedding (GPT CLS token)
    def __init__(self, tokenizer_name='gpt2', embed_size=768, dropout_rate=0.1):
        super(TextEncoder, self).__init__()
        # Pre-trained GPT-2 (768)
        self.Text_Encoder = GPT2Model.from_pretrained('gpt2') if tokenizer_name == 'gpt2' else None
        self.dropout = nn.Dropout(dropout_rate)

        self.embed_convert = nn.Linear(self.Text_Encoder.embed_dim, embed_size) \
            if self.Text_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, input_ids, attention_mask):
        # Process text through GPT-2 to generate a seq of hidden state
        Text_outputs = self.Text_Encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        Text_cls_embedding = Text_outputs[:, -1, :]  # GPT-2 uses the last token embedding as CLS representation
        Text_cls_embedding = self.dropout(Text_cls_embedding)

        return self.embed_convert(Text_cls_embedding)


# ------------------- Multiple Modality Fusion -------------------
# The text embeddings (query) are passed into the attention mechanism to attend to the image embeddings (key/value).
# The multi-head attention layer computes the attention weights that help the model focus on relevant visual features
# based on the textual query.
# The attended image and text features are concatenated together to form a unified representation of both modalities.
class MultiHeadAttention(nn.Module):
    # In the attention mechanism (both single and multi-head), the core idea is to let the model focus on
    # different parts of the input sequence or different inputs to capture relationships
    # In this Visual Question Answering (VQA) model,
    # the attention mechanism helps the text (the query) focus on the image (to answer the question).
    def __init__(self, embed_size=768, heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        '''
        Query (Q): Represents what you are trying to match or attend to (here it is the text embeddings).
        Key (K): Represents the features to compare against (here it is the image embeddings).
        Value (V): Holds the actual data that will be output after attention is applied (here it is image embeddings).
        The key tells the model where to attend, and the value gives the information for those attended locations.
        '''
        # query, key, value should be [seq_len, batch, embed_size]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)

        # Transpose back to [batch, seq_len, embed_size]
        attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_weights


class MultipleModalityFusion(nn.Module):
    # In the attention mechanism (both single and multi-head), the core idea is to let the model focus on
    # different parts of the input sequence or different inputs to capture relationships
    # In this Visual Question Answering (VQA) model,
    # the attention mechanism helps the text (the query) focus on the image (to answer the question).
    def __init__(self, fusion_method='MHSA', embed_size=768, heads=8, dropout_rate=0.1):
        super(MultipleModalityFusion, self).__init__()
        self.fusion_method = fusion_method
        if self.fusion_method == 'MHSA':
            self.attention = MultiHeadAttention(embed_size=embed_size, heads=heads, dropout_rate=dropout_rate)
        elif self.fusion_method == 'clip':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, text_features, image_features):
        if self.fusion_method == 'MHSA':
            # Attention between image and text
            query = text_features.unsqueeze(1)  # Text features as query
            key_value = image_features.unsqueeze(1)  # Image features as key/value
            attended_features, _ = self.attention(query, key_value, key_value)

            # Combine attended features with text features [B, 2 * embed_size]
            combined_features = torch.cat((attended_features.squeeze(1), text_features), dim=1)
            # The attended_features (the output of the attention mechanism) is combined with the original text_features
            # attended_features.squeeze(1): Removes the extra dimension added by unsqueeze(1) earlier

            return combined_features
        elif self.fusion_method == 'clip':
            raise NotImplementedError
        else:
            raise NotImplementedError


# ------------------- Answer Decoder (VQAbyCLS Classifier) -------------------
class AnswerDecoder_VQAbyCLS(nn.Module):
    '''
    The VQAbyCLS is task design that align and train the multiple modal output in a classification manner

    in the output langurage decoding stage:
    The combined features (which now include both the attended image information and the text representation)
    are passed into the answer decoder, which is a linear classifier predicts
    the final answer by producing logits for each possible answer class.

    The output, logits, is a tensor of size [batch_size, num_classes],
    it represents the raw scores for each possible answer class,
    where num_classes is the total number of possible answer classes

    '''

    def __init__(self, embed_size=768, num_classes=None):
        assert num_classes is not None
        super(AnswerDecoder_VQAbyCLS, self).__init__()
        self.classifier = nn.Linear(embed_size * 2, num_classes)

    def forward(self, combined_features):
        # Classification to predict the answer
        logits = self.classifier(combined_features)
        return logits


# ------------------- Full VQA Model -------------------
class VQAModel_VQAbyCLS(nn.Module):
    def __init__(self, image_encoder, text_encoder, fusion_method='MHSA',
                 num_classes=None, embed_size=768, heads=8, dropout_rate=0.1):
        assert num_classes is not None
        super(VQAModel_VQAbyCLS, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # fusion with clip for future
        self.fusion = MultipleModalityFusion(fusion_method=fusion_method,
                                             embed_size=embed_size, heads=heads, dropout_rate=dropout_rate)

        self.answer_decoder = AnswerDecoder_VQAbyCLS(embed_size=embed_size, num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask):
        # Image encoding
        image_features = self.image_encoder(images)
        # Text encoding
        text_features = self.text_encoder(input_ids, attention_mask)
        # fusion
        combined_features = self.fusion(text_features, image_features)
        # Answer classification [B, 2 * embed_size] -> logits [B, N(num cls)]
        logits = self.answer_decoder(combined_features)

        return logits


# ------------------- Training and Evaluation -------------------

def train_and_validate(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs=10):
    best_val_accuracy = 0.0  # To track the best validation accuracy
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        # Add tqdm to show the progress for each batch
        train_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        for batch in train_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()  # Clear gradients
            # forward
            with torch.cuda.amp.autocast():  # automatic mix precision training
                logits = model(images, input_ids, attention_mask)  # Forward pass
                loss = loss_fn(logits, labels)  # Calculate loss
                total_train_loss += loss.item()
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            _, predicted = torch.max(logits.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update tqdm description with current loss and accuracy
            train_bar.set_postfix(loss=loss.item(), accuracy=correct_train / total_train)

        # Calculate average training loss and accuracy for this epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train if total_train > 0 else 0
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation step at the end of each epoch
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Track the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    return train_losses, train_accuracies, val_losses, val_accuracies


# ------------------- Validation Function (evaluate) -------------------
def evaluate(model, dataloader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss, correct, total = 0, 0, 0

    # Add tqdm to show progress for the validation/test loop
    val_bar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in val_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(images, input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update tqdm description with current loss and accuracy
            val_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == '__main__':
    # Constants
    IMG_SIZE = 224
    MAX_SEQ_LENGTH = 256  # Adjust based on typical question length
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    DROP_RATE = 0.1
    HEADS = 8
    EMBED_SIZE = 768

    tokenizer_name = 'gpt2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------- Prepare the excel dataset -------------------
    # Create DataLoaders for batching and loading the data
    # Download the PathVQA dataset from Hugging Face
    dataset = load_dataset("flaviagiammarino/path-vqa")
    # Preprocess and clean the dataset
    for split in ["train", "validation", "test"]:  # Use "validation" instead of "val"
        dataset[split] = dataset[split].map(lambda example: {
            'question': clean_text(example['question']),
            'answer': clean_text(example['answer'])
        })
    # Check available splits
    print(dataset)

    # Extract all answers from train, validation, and test splits
    all_answers = dataset['train']['answer'] + dataset['validation']['answer'] + dataset['test']['answer']
    # Get unique answers
    unique_answers = np.unique(all_answers)
    # Map each unique answer to an index
    answer_to_index = {ans: idx for idx, ans in enumerate(unique_answers)}
    # Number of classes (unique answers)
    num_classes = len(answer_to_index)
    print(f"Number of unique answers: {num_classes}")

    # ------------------- Create Datasets & DataLoaders -------------------
    train_dataset = VQA_Dataset(hf_dataset=dataset['train'], answer_to_index=answer_to_index)
    val_dataset = VQA_Dataset(hf_dataset=dataset['validation'], answer_to_index=answer_to_index)
    test_dataset = VQA_Dataset(hf_dataset=dataset['test'], answer_to_index=answer_to_index)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                                  num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn,
                                num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn,
                                 num_workers=2)

    # ------------------- Build VQA model and task-------------------
    # Initialize model
    image_encoder = ImageEncoder(embed_size=EMBED_SIZE)
    text_encoder = TextEncoder(embed_size=EMBED_SIZE, dropout_rate=DROP_RATE)
    model = VQAModel_VQAbyCLS(image_encoder, text_encoder, fusion_method='MHSA',
                              embed_size=EMBED_SIZE, heads=HEADS, dropout_rate=DROP_RATE,
                              num_classes=num_classes)
    model = torch.compile(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss()

    # ------------------- Training Loop-------------------
    train_losses, train_accuracies, val_losses, val_accuracies = train_and_validate(
        model, train_dataloader, val_dataloader, optimizer, loss_fn, device, epochs=EPOCHS
    )

    # ------------------- Test Code -------------------
    # Evaluate the model on the test dataset
    test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)

    # Print out test results
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
