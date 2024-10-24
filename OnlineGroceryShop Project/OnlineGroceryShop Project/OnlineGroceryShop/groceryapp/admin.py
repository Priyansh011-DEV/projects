from django.contrib import admin

from groceryapp.models import Student
from .models import *




admin.site.register(Student)
admin.site.register(Carousel)
admin.site.register(Category)
admin.site.register(Product)
admin.site.register(UserProfile)
admin.site.register(Cart)
admin.site.register(Booking)
admin.site.register(Feedback)
