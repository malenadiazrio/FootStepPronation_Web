from django.db import models
from django.shortcuts import render
from django.conf import settings
from django import forms

from modelcluster.fields import ParentalKey

from wagtail.admin.panels import (
    FieldPanel,
    MultiFieldPanel,
)
from wagtail.models import Page
from wagtail.fields import RichTextField
from django.core.files.storage import default_storage
from model_integration.yoloact_custom import eval_yoloact
from pathlib import Path


import os, uuid, glob, cv2

str_uuid = uuid.uuid4()  # The UUID for image uploading

def reset():
    files_result = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/Result/*.*')), recursive=True)
    files_upload = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/uploadedPics/*.*')), recursive=True)
    files_temp = glob.glob(str(Path(f'{settings.MEDIA_ROOT}/videoFrames/*.*')), recursive=True)
    files = []
    # if len(files_result) != 0:
    #     files.extend(files_result)
    if len(files_upload) != 0:
        files.extend(files_upload)
    if len(files_temp) != 0:
        files.extend(files_temp)
    if len(files) != 0:
        for f in files:
            try:
                if (not (f.endswith(".txt"))):
                    os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        file_li = [Path(f'{settings.MEDIA_ROOT}/Result/Result.txt'),
                   Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'),
                   Path(f'{settings.MEDIA_ROOT}/Result/stats.txt')]
        for p in file_li:
            file = open(Path(p), "r+")
            file.truncate(0)
            file.close()

def test(filename):
    print("Analyzing the video {}".format(filename))
    eval_yoloact(filename)
    return os.listdir(os.path.join(settings.MEDIA_ROOT, "Result"))

# Create your models here.
class ImagePage(Page):
    """Image Page."""

    template = "cam_app2/image.html"

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


    def reset_context(self, request):
        context = super().get_context(request)
        context["my_uploaded_file_names"]= []
        context["my_result_file_names"]=[]
        context["my_staticSet_names"]= []
        context["my_lines"]= []
        return context

    def serve(self, request):
        emptyButtonFlag = False
        if request.POST.get('start')=="":
            context = self.reset_context(request)
            print(request.POST.get('start'))
            print("Start selected")
            fileroot = os.path.join(settings.MEDIA_ROOT, 'uploadedPics')
            res_f_root = os.path.join(settings.MEDIA_ROOT, 'Result')
            with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'), 'r') as f:
                image_files = f.readlines()
            r_file_paths = test(image_files[0])
            if len(r_file_paths)>=0:
                for r_file in r_file_paths:
                    ###
                    # For each file, read, pass to model, do something, save it #
                    ###
                    if r_file.split('.')[-1] == 'txt':
                        continue
                    r_media_filepath = Path(f"{settings.MEDIA_URL}Result/{r_file}")
                    context["my_uploaded_file_names"].append(str(f'{str(image_files[0])}'))
                    context["my_result_file_names"].append(str(f'{str(r_media_filepath)}'))
            return render(request, "cam_app2/image.html", context)

        if (request.FILES and emptyButtonFlag == False):
            print("reached here files")
            reset()
            context = self.reset_context(request)
            context["my_uploaded_file_names"] = []
            for file_obj in request.FILES.getlist("file_data"):
                uuidStr = uuid.uuid4()
                filename = f"{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}"
                with default_storage.open(Path(f"uploadedPics/{filename}"), 'wb+') as destination:
                    for chunk in file_obj.chunks():
                        destination.write(chunk)
                filename = Path(f"{settings.MEDIA_URL}uploadedPics/{file_obj.name.split('.')[0]}_{uuidStr}.{file_obj.name.split('.')[-1]}")
                with open(Path(f'{settings.MEDIA_ROOT}/uploadedPics/img_list.txt'), 'a') as f:
                    f.write(str(filename))
                    f.write("\n")
                context["my_uploaded_file_names"].append(str(f'{str(filename)}'))

            return render(request, "cam_app2/image.html", context)
        context = self.reset_context(request)
        reset()
        return render(request, "cam_app2/image.html", {'page': self})
    