from django.db import models


class WellbeingSurvey(models.Model):
    GENDER_CHOICES = [
        ("male", "Male"),
        ("female", "Female"),
        ("other", "Other"),
    ]

    name = models.CharField(max_length=50)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age = models.PositiveIntegerField()
    answers = models.JSONField(default=list)
    comment = models.TextField(blank=True)

    prediction = models.CharField(max_length=64, blank=True)
    suggestion = models.CharField(max_length=255, blank=True)
    model_used = models.CharField(max_length=128, blank=True)
    detector = models.CharField(max_length=64, blank=True)
    top_predictions = models.JSONField(default=list)

    score = models.IntegerField(default=0)
    percent_score = models.IntegerField(default=0)
    score_max = models.IntegerField(default=0)

    stream_summary = models.JSONField(default=dict)
    client_session_key = models.CharField(max_length=64, blank=True, db_index=True)
    image_path = models.CharField(max_length=255, blank=True)
    image_url = models.CharField(max_length=255, blank=True)

    openai_response = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.prediction or 'No prediction'})"


class EmotionStreamEntry(models.Model):
    session_key = models.CharField(max_length=64, db_index=True)
    survey = models.ForeignKey(
        WellbeingSurvey,
        related_name="stream_entries",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    label = models.CharField(max_length=64, default="Unknown")
    confidence = models.FloatField(default=0.0)
    top = models.JSONField(default=list)
    captured_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["session_key", "-captured_at"]),
        ]
        ordering = ["-captured_at"]

    def __str__(self):
        return f"{self.label} @ {round(self.confidence, 2)}"
