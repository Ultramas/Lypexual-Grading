# grader/views.py

import os
import tempfile
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from .models import GradingResult, GradedCardSample
from .ml.predict import grade_card


class GradeCardView(APIView):
    """POST an image → get a PSA grade prediction."""
    parser_classes = [MultiPartParser]

    def post(self, request):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Save to temp file for OpenCV processing
        suffix = os.path.splitext(image_file.name)[-1]
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=settings.MEDIA_ROOT
        ) as tmp:
            for chunk in image_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            result = grade_card(tmp_path)

            # Save to DB
            grading = GradingResult.objects.create(
                submitted_image=image_file,
                predicted_grade=result["adjusted_grade"],
                rounded_grade=result["rounded_grade"],
                confidence=result["confidence_pct"],
                centering_score=result["sub_scores"]["centering"],
                ip_address=request.META.get("REMOTE_ADDR")
            )

            return Response({
                "id": grading.id,
                **result
            })
        finally:
            os.unlink(tmp_path)


class TrainingDataView(APIView):
    """GET training data stats per grade."""

    def get(self, request):
        stats = {}
        for grade in range(1, 11):
            count = GradedCardSample.objects.filter(grade=grade).count()
            stats[f"PSA {grade}"] = count
        return Response({
            "total_samples": sum(stats.values()),
            "per_grade": stats,
            "recommended_minimum": "500 images per grade for decent accuracy"
        })