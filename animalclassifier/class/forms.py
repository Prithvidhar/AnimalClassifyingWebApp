from django import forms

class AnimalForm(forms.Form):
    image = forms.ImageField()