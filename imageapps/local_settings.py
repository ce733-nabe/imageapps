import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

#SECRET_KEY = 'django-insecure-xr3s@yo!3ipeuculyn$o=58o$mv9#a2eq4ch(@6c(8@$uo55jx'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

ALLOWED_HOSTS = []

DEBUG = True
