from django.conf import settings
from django.urls import include, path
from django.contrib import admin

from wagtail.admin import urls as wagtailadmin_urls
from wagtail import urls as wagtail_urls
from wagtail.documents import urls as wagtaildocs_urls

from search import views as search_views
from cam_app import views, camera
from cam_app2 import views as v2
from django.http import StreamingHttpResponse
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve


urlpatterns = [

    path('django-admin/', admin.site.urls),

    path('admin/', include(wagtailadmin_urls)),
    path('documents/', include(wagtaildocs_urls)),

    path('search/', search_views.search, name='search'),

    path('scanner_video/', views.ScannerVideoView.as_view(), name='scanner_video'),
    path('img/', v2.ImageView.as_view(), name='img'),
    path('no_video/', views.NoVideoView.as_view(), name='no_video'),
    path('camera_feed/', lambda r: StreamingHttpResponse(camera.generate_frames(camera.VideoCamera(), False),
                                                     content_type='multipart/x-mixed-replace; boundary=frame;')),
    path('camera_feed_AI/', lambda r: StreamingHttpResponse(camera.generate_frames(camera.VideoCamera(), True),
                                                     content_type='multipart/x-mixed-replace; boundary=frame;')),
    path('', include(wagtail_urls)),

]


if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
