from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="WellbeingSurvey",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=50)),
                ("gender", models.CharField(choices=[("male", "Male"), ("female", "Female"), ("other", "Other")], max_length=10)),
                ("age", models.PositiveIntegerField()),
                ("answers", models.JSONField(default=list)),
                ("comment", models.TextField(blank=True)),
                ("prediction", models.CharField(blank=True, max_length=64)),
                ("suggestion", models.CharField(blank=True, max_length=255)),
                ("model_used", models.CharField(blank=True, max_length=128)),
                ("detector", models.CharField(blank=True, max_length=64)),
                ("top_predictions", models.JSONField(default=list)),
                ("score", models.IntegerField(default=0)),
                ("percent_score", models.IntegerField(default=0)),
                ("score_max", models.IntegerField(default=0)),
                ("stream_summary", models.JSONField(default=dict)),
                ("client_session_key", models.CharField(blank=True, db_index=True, max_length=64)),
                ("image_path", models.CharField(blank=True, max_length=255)),
                ("image_url", models.CharField(blank=True, max_length=255)),
                ("openai_response", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="EmotionStreamEntry",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("session_key", models.CharField(db_index=True, max_length=64)),
                ("label", models.CharField(default="Unknown", max_length=64)),
                ("confidence", models.FloatField(default=0.0)),
                ("top", models.JSONField(default=list)),
                ("captured_at", models.DateTimeField(auto_now_add=True)),
                ("survey", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="stream_entries", to="emotion.wellbeingsurvey")),
            ],
            options={
                "ordering": ["-captured_at"],
            },
        ),
        migrations.AddIndex(
            model_name="emotionstreamentry",
            index=models.Index(fields=["session_key", "-captured_at"], name="emotion_emo_session_f21435_idx"),
        ),
    ]
