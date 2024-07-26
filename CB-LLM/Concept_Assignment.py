import re
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# Clean up unnecessary characters and extra spaces in the text
def clean_text(texts):

    cleaned_texts = []
    for text in texts:
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        cleaned_texts.append(text.strip())

    return cleaned_texts

# Clean up unnecessary characters and extra spaces in the concept
def clean_concepts(concepts):

    pattern = r"<example>(.*?)</example>"
    extracted_concepts = []

    for concept in concepts:
        extracted_concepts.extend(re.findall(pattern, concept))

    return extracted_concepts

# Binarization of conceptual allocation results based on thresholds
def thresholding(concept_assignment):

    for i in range(concept_assignment.shape[0]):
        concept_assignment[i] = np.where(concept_assignment[i]>= 0.515648365020752, 1, 0)

    return concept_assignment

# Calculate the cosine similarity between text vectors and concept vectors
def cos_similarity(concepts_vectors, text_vectors):

    num_texts = len(text_vectors)
    num_concepts = len(concepts_vectors)
    scores = np.zeros((num_texts, num_concepts))

    for i in range(num_texts):
        for j in range(num_concepts):
             similarity = np.dot(text_vectors[i],concepts_vectors[j])/(norm(text_vectors[i])*norm(concepts_vectors[j]))
             scores[i,j] = similarity
    # Binarization using thresholding
    scores = thresholding(scores)

    return scores

def get_concepts(concepts,labels):
    text_num = labels.shape[0]
    selected_concepts = []
    concepts = np.array(concepts)                              
    for i in range(len(labels)):
        c = concepts[labels[i] == 1]
        selected_concepts.append(c)
    
    return selected_concepts
    
# Assign concepts to text
# Concepts and texts are lists
def assign_concepts(concepts,texts):

    concepts = clean_concepts(concepts)
    texts = clean_text(texts)
    # Load UAE pre-trained models
    UAE_model = SentenceTransformer("WhereIsAI/UAE-Large-V1").cuda()

    UAE_concepts_vectors = UAE_model.encode(concepts)
    UAE_text_vectors = UAE_model.encode(texts)
    concept_assignment = cos_similarity(UAE_concepts_vectors, UAE_text_vectors)
    selected_concepts = get_concepts(concepts, concept_assignment)
    return concept_assignment, selected_concepts

