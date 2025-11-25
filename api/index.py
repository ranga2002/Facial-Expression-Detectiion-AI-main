import os

from django.core.wsgi import get_wsgi_application

# Ensure Django can locate its settings module when the Vercel function starts.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fer_django_app.settings")

# Expose the WSGI application as the Vercel handler.
application = get_wsgi_application()
