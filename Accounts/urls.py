from django.urls import path
from Accounts.views import * 


urlpatterns = [
    path('signup/',signUp,name='signup'),
    path('login/',signIn,name='login'),
    path('logout/',logoutUser,name='logout'),
]
