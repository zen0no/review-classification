import torch

from django.shortcuts import render
from django.http import HttpResponse

from semantic_classifier.utils import load_classifier_model
from spacy.lang.en import English 

from . import forms


nlp = English()



model, vocab = load_classifier_model()
model.eval()

def _preprocess_input_text(text: str):
    tokens = torch.LongTensor([vocab[token.text] for token in nlp.tokenizer(text)])
    outputs = model(torch.unsqueeze(tokens,0))

    return torch.argmax(torch.squeeze(outputs)).item() + 1 


def index(requests):
    context = dict()

    if requests.method == 'POST':
        form = forms.ReviewForm(requests.POST)

        if form.is_valid():
            text = form.cleaned_data['review']
            rating = _preprocess_input_text(text)
            context['rating'] = str(rating)
            context['semantic'] = 'Negative' if rating < 6 else 'Positive'
    else:
        form = forms.ReviewForm()
    context['form'] = form
    return render(requests, 'index.html', context=context)