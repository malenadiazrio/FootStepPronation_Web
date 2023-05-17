from django.db import models
from django.shortcuts import render
from django.conf import settings

from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel,
)
from wagtail.models import Page
from wagtail.fields import RichTextField


# Create your models here.
class VideoPage(Page):
    """Video Page."""

    template = "cam_app/video2.html"

    max_count = 2

    name_title = models.CharField(max_length=100, blank=True, null=True)
    name_subtitle = RichTextField(features=["bold", "italic"], blank=True)

    content_panels = Page.content_panels + [
        MultiFieldPanel(
            [
                FieldPanel("name_title"),
                FieldPanel("name_subtitle"),

            ],
            heading="Page Options",
        ),
    ]
    


    def serve(self, request):
        return  render(request, "cam_app/video2.html", {'page': self})
