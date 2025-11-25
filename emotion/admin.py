from django.contrib import admin

from .models import EmotionStreamEntry, WellbeingSurvey


@admin.register(WellbeingSurvey)
class WellbeingSurveyAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "gender",
        "age",
        "prediction",
        "percent_score",
        "model_used",
        "created_at",
    )
    list_filter = ("gender", "model_used", "detector", "created_at")
    search_fields = ("name", "comment", "prediction")
    readonly_fields = ("created_at",)


@admin.register(EmotionStreamEntry)
class EmotionStreamEntryAdmin(admin.ModelAdmin):
    list_display = ("label", "confidence", "session_key", "survey", "captured_at")
    list_filter = ("label", "captured_at")
    search_fields = ("session_key", "label")
    readonly_fields = ("captured_at",)
