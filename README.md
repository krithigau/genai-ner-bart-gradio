## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Named Entity Recognition (NER) is a crucial task in Natural Language Processing (NLP) that involves identifying and classifying entities such as names, locations, dates, and other proper nouns in a text. Traditional NER models often fail to accurately recognize entities when applied to specialized domains or datasets. By fine-tuning a pre-trained model like BART (Bidirectional and Auto-Regressive Transformers), it is possible to create a more effective model for NER that performs well in specific domains. This project aims to develop a prototype application for NER by fine-tuning a BART model and deploying the application using the Gradio framework, which allows users to interact with the system and evaluate its performance.

### DESIGN STEPS:

#### STEP 1:
Fine-Tune BART Model: Use a dataset with labeled entities (e.g., CoNLL-03 or custom dataset) to fine-tune the BART model.

#### STEP 2:
Model Evaluation: Test the fine-tuned model on unseen data to evaluate its performance (precision, recall, F1-score).

#### STEP 3:
Create Gradio Interface: Build a simple web-based interface using Gradio where users can input text and view the NER results.

#### STEP 4:
Deploy the Prototype: Deploy the NER application to a local or cloud-based platform for user interaction.

#### STEP 5:
Evaluate User Feedback: Collect user feedback to assess the accuracy and usability of the system.

### PROGRAM:
```py
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline
import gradio as gr

model = BartForConditionalGeneration.from_pretrained("path_to_fine_tuned_model")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def named_entity_recognition(text):
    entities = ner_pipeline(text)
    results = [(entity['word'], entity['entity']) for entity in entities]
    return results

iface = gr.Interface(fn=named_entity_recognition, inputs="text", outputs="json")

iface.launch()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/8d9dee95-d108-41e5-9f92-e8c8c368c272)


### RESULT:

The application will prompt the user to input text, and the NER system will output a list of identified entities along with their corresponding labels (e.g., PERSON, LOCATION, DATE). The Gradio interface will make it easy for users to interact with the model and evaluate its performance on different texts.
