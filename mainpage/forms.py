from django import forms

class ReviewForm(forms.Form):
    review  = forms.CharField(widget=forms.Textarea)