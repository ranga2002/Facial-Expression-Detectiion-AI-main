import random

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

# 50-question bank with reverse-scored flags.
QUESTION_BANK = [
    ("I've been feeling emotionally stable lately.", False),
    ("I've experienced anxiety or sadness recently.", True),
    ("I feel I have support from people around me.", False),
    ("I'm able to concentrate and stay focused.", False),
    ("I've been sleeping well and waking rested.", False),
    ("I feel tense, irritable, or on edge.", True),
    ("I still enjoy hobbies or time with others.", False),
    ("I feel hopeful about the near future.", False),
    ("Small tasks feel overwhelming to me.", True),
    ("I have energy to get through my day.", False),
    ("I feel disconnected from people I care about.", True),
    ("I find it hard to relax or unwind.", True),
    ("I’ve been taking breaks that actually help me reset.", False),
    ("I often feel on autopilot or numb.", True),
    ("I’m satisfied with how I’m spending my time.", False),
    ("My appetite has noticeably changed.", True),
    ("I’ve been kind to myself when I make mistakes.", False),
    ("I dread the day ahead when I wake up.", True),
    ("I can solve problems without feeling stuck.", False),
    ("I’ve been avoiding people or plans I usually enjoy.", True),
    ("I feel proud of something I did this week.", False),
    ("I’m worrying more than usual about the future.", True),
    ("I feel physically tense (jaw, shoulders, stomach).", True),
    ("I’m getting outside or moving my body regularly.", False),
    ("I feel like I’m letting others down.", True),
    ("I can name something I’m grateful for today.", False),
    ("I’ve noticed my patience running thin.", True),
    ("I’m able to sleep through the night.", False),
    ("I feel motivated to start tasks.", False),
    ("I’ve had moments of panic or racing thoughts.", True),
    ("I’m comfortable asking for help when I need it.", False),
    ("I feel lonely even when I’m around others.", True),
    ("I’m drinking enough water and eating regular meals.", False),
    ("I snap at people more than I mean to.", True),
    ("I’m managing stress in healthy ways.", False),
    ("I feel like my mood is out of my control.", True),
    ("I’m able to focus on one thing at a time.", False),
    ("I’ve been replaying negative moments over and over.", True),
    ("I make time for things that bring me joy.", False),
    ("I feel safe in my environment.", False),
    ("I’m experiencing frequent headaches or stomachaches.", True),
    ("I can express my feelings to someone I trust.", False),
    ("I’m comparing myself harshly to others.", True),
    ("I look forward to parts of my day.", False),
    ("I feel misunderstood most of the time.", True),
    ("I’m keeping a steady daily routine.", False),
    ("I feel guilty for resting or taking breaks.", True),
    ("I’m handling unexpected changes well.", False),
    ("I’ve been withdrawing from conversations.", True),
    ("I notice moments of calm during the day.", False),
    ("I’ve felt like giving up this week.", True),
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

    def __init__(self, *args, selected_questions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_questions = selected_questions or self.sample_questions()
        self.reverse_flags = [q[1] for q in self.selected_questions]
        # Dynamically attach question fields with the sampled labels.
        for idx, (text, _) in enumerate(self.selected_questions, start=1):
            field_name = f"question{idx}"
            self.fields[field_name] = forms.ChoiceField(
                label=text,
                choices=LIKERT_CHOICES,
                widget=forms.RadioSelect
            )
        # Preserve field order: basic info, questions, then comment/image.
        ordered = ["name", "gender", "age"] + [f"question{i}" for i in range(1, len(self.selected_questions) + 1)] + ["comment", "image"]
        self.order_fields(ordered)

    @staticmethod
    def sample_questions(count: int = 10):
        available = QUESTION_BANK[:]
        sample_size = min(count, len(available))
        return random.sample(available, sample_size)

    def get_reverse_flags(self):
        return list(self.reverse_flags)
