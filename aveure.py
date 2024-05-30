
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
import spacy
import os



#!python -m spacy download ca_core_news_md
nlp = spacy.load("ca_core_news_md")

dataset = load_dataset('projecte-aina/sts-ca')

bert_tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
bert_model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

roberta_tokenizer = RobertaTokenizer.from_pretrained('projecte-aina/roberta-base-ca-v2')
roberta_model = RobertaModel.from_pretrained('projecte-aina/roberta-base-ca-v2')


# One-Hot Encoding
def get_one_hot_embeddings(sentences, vocab_size=1000):
    vectorizer = TfidfVectorizer(max_features=vocab_size)
    vectors = vectorizer.fit_transform(sentences).toarray()
    return torch.tensor(vectors, dtype=torch.float)

# Word2Vec
word2vec_model = Word2Vec(sentences=[sent.split() for sent in dataset['train']['sentence1'] + dataset['train']['sentence2']], vector_size=100, min_count=1)
def get_word2vec_embeddings(sentences):
    vectors = [torch.tensor([word2vec_model.wv[word] for word in sent.split() if word in word2vec_model.wv], dtype=torch.float) for sent in sentences]
    vectors = [vec.mean(dim=0) if len(vec) > 0 else torch.zeros(100) for vec in vectors]
    return torch.stack(vectors)

# Weighted Word2Vec (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(dataset['train']['sentence1'] + dataset['train']['sentence2'])
def get_weighted_word2vec_embeddings(sentences):
    vectors = []
    for sent in sentences:
        weights = torch.tensor(tfidf_vectorizer.transform([sent]).toarray()[0], dtype=torch.float)
        words = sent.split()
        vec = torch.zeros(100)
        weight_sum = 0
        for i, word in enumerate(words):
            if word in word2vec_model.wv:
                vec += weights[i] * torch.tensor(word2vec_model.wv[word])
                weight_sum += weights[i]
        vectors.append(vec / weight_sum if weight_sum > 0 else vec)
    return torch.stack(vectors)

# spaCy
def get_spacy_embeddings(sentences):
    vectors = [torch.tensor(nlp(sent).vector, dtype=torch.float) for sent in sentences]
    return torch.stack(vectors)

# BERT
def get_bert_embeddings(sentences):
    embeddings = []
    for sentence in sentences:
        inputs = bert_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze())
    return torch.stack(embeddings)

# RoBERTa
def get_roberta_embeddings(sentences):
    embeddings = []
    for sentence in sentences:
        inputs = roberta_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
        outputs = roberta_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze())
    return torch.stack(embeddings)

# Preprocessar el dataset
def preprocess_dataset(dataset, get_embeddings):
    processed_data = []
    for example in dataset:
        sent1_embedding = get_embeddings([example['sentence1']])[0]
        sent2_embedding = get_embeddings([example['sentence2']])[0]
        label = torch.tensor(example['label'], dtype=torch.float)
        processed_data.append((torch.cat((sent1_embedding, sent2_embedding)), label))
    return processed_data


# Model similitud
class SimilarityModel(nn.Module):
    def __init__(self, input_dim):
        super(SimilarityModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)
    
    
'''def create_dataloader(data, batch_size=32):
    return DataLoader(data, batch_size=batch_size, shuffle=True)'''

def create_dataloader(data, batch_size=32):
    # Obtener la longitud máxima de las oraciones en el conjunto de datos
    max_length = max(len(sentence) for sentence, _ in data)
    # Rellenar o truncar las entradas para que tengan la misma longitud
    padded_inputs = [torch.cat((sentence, torch.zeros(max_length - len(sentence)))) for sentence, _ in data]
    # Stack y convertir a TensorDataset
    dataset = TensorDataset(pad_sequence(padded_inputs, batch_first=True), torch.stack([label for _, label in data]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Función para entrenar el modelo
def train_model(train_loader, input_dim, num_epochs=5):
    model = SimilarityModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return model, criterion


# Evaluar el model
def evaluate_model(model, dev_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in dev_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dev_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')



# Evaluar diferentes embeddings
embedding_methods = {
    'One-Hot': get_one_hot_embeddings,
    'Word2Vec': get_word2vec_embeddings,
    'Weighted Word2Vec': get_weighted_word2vec_embeddings,
    'spaCy': get_spacy_embeddings,
    'BERT': get_bert_embeddings,
    'RoBERTa': get_roberta_embeddings,
}

for name, get_embeddings in embedding_methods.items():
    print(f"Evaluating {name} embeddings")
    
    # Preprocesar los datos
    train_data = preprocess_dataset(dataset['train'], get_embeddings)
    dev_data = preprocess_dataset(dataset['validation'], get_embeddings)
    
    # Crear DataLoaders
    train_loader = create_dataloader(train_data)
    dev_loader = create_dataloader(dev_data)
    
    # Entrenar el modelo
    # input_dim = len(train_data[0][0])
    
    input_dim = train_data[0][0].shape[0]
    
    model, criterion = train_model(train_loader, input_dim)
    
    # Evaluar el modelo
    evaluate_model(model, dev_loader, criterion)
