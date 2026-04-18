# grader/models.py

from django.db import models


class PokemonCard(models.Model):
    name = models.CharField(max_length=200)
    set_name = models.CharField(max_length=200)
    card_number = models.CharField(max_length=20)
    year = models.IntegerField(null=True, blank=True)

    class Meta:
        unique_together = ("name", "set_name", "card_number")

    def __str__(self):
        return f"{self.name} ({self.set_name} #{self.card_number})"


class GradedCardSample(models.Model):
    """A single labeled training image."""
    GRADE_CHOICES = [(i, f"PSA {i}") for i in range(1, 11)]

    card = models.ForeignKey(
        PokemonCard, on_delete=models.CASCADE,
        null=True, blank=True, related_name="samples"
    )
    image = models.ImageField(upload_to="training_cards/")
    grade = models.IntegerField(choices=GRADE_CHOICES)
    source = models.CharField(
        max_length=50,
        choices=[("ebay", "eBay"), ("psa", "PSA Registry"),
                 ("kaggle", "Kaggle"), ("manual", "Manual")],
        default="manual"
    )
    cert_number = models.CharField(max_length=50, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["grade"]

    def __str__(self):
        return f"PSA {self.grade} - {self.card}"


class GradingResult(models.Model):
    """Results from user-submitted cards."""
    submitted_image = models.ImageField(upload_to="submissions/")
    predicted_grade = models.FloatField()  # Raw float (e.g. 8.7)
    rounded_grade = models.IntegerField()  # PSA scale 1-10
    confidence = models.FloatField()  # 0-100%

    centering_score = models.FloatField(null=True)
    surface_score = models.FloatField(null=True)
    corner_score = models.FloatField(null=True)
    edge_score = models.FloatField(null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)