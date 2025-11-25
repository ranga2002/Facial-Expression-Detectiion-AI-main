from django import forms

GENDER_CHOICES = [
    ('male', 'Male'),
    ('female', 'Female'),
    ('other', 'Other'),
]

LIKERT_CHOICES = [
    ('1', 'Strongly Disagree'),
    ('2', 'Disagree'),
    ('3', 'Neutral'),
    ('4', 'Agree'),
    ('5', 'Strongly Agree'),
]

class ImageUploadForm(forms.Form):
    name = forms.CharField(
        label="Your Name",
        max_length=50,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )

    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        label="Gender",
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    age = forms.IntegerField(
        label="Age",
        widget=forms.NumberInput(attrs={
            'class': 'form-range w-100',
            'type': 'range',
            'min': 10,
            'max': 100,
            'step': 1,
            'oninput': 'document.getElementById("ageValue").textContent = this.value'
        })
    )

    question1 = forms.ChoiceField(
        label="I've been feeling emotionally stable lately.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )
    question2 = forms.ChoiceField(
        label="I've experienced anxiety or sadness recently.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )
    question3 = forms.ChoiceField(
        label="I feel I have support from people around me.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )
    question4 = forms.ChoiceField(
        label="I'm able to concentrate and stay focused.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )
    question5 = forms.ChoiceField(
        label="I've been sleeping well and waking rested.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )
    question6 = forms.ChoiceField(
        label="I feel tense, irritable, or on edge.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )
    question7 = forms.ChoiceField(
        label="I still enjoy hobbies or time with others.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )
    question8 = forms.ChoiceField(
        label="I feel hopeful about the near future.",
        choices=LIKERT_CHOICES,
        widget=forms.RadioSelect
    )

    comment = forms.CharField(
        required=False,
        label="Any other thoughts you'd like to share?",
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control'})
    )

    image = forms.ImageField(
        required=False,
        label="Upload a Face Image",
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )
