<div align="center">

<h2 align="center">How to build your own Web App with a machine learning model</h2>
<h3> <strong>Option 1:</strong> Setting up Wagtail from scratch</h3>

</div>

## Steps:
1.	Create a folder wherever you want the web app to be stored. 
2.	Open terminal in that folder OR change directory to that folder.
    ```sh
    mkdir my_project
    cd my_project
    ```
3.	Make sure to use `python 3.9` or higher version.
4. Create a python virtual environment named `venv`:
    ```sh
    python -m venv venv
    ```
5. Activate the virtual environment `venv`:

    Linux/MacOS:
    ```sh
    source venv/bin/activate
    ```
    Windows:
    ```sh
    venv\Scripts\activate
    ```
6. Once Virtual Environment is active, your terminal will look like the following:

    Linux/MacOS:
    ```sh
    (venv) apple... $ `or` %
    ```
    Windows:
    ```sh
    (venv) C:\.... >
    ```
7. Upgrade pip:
    ```sh
    pip install --upgrade pip
    ```
8. Install the libraries now (make sure venv is activated):
    
    Create a file named `requirements.txt` with the following contents:
    ```sh
    django
    django-allauth
    django-extensions
    django-filter
    django-modelcluster
    django-taggit
    django-treebeard
    djangorestframework
    wagtail
    matplotlib
    numpy
    opencv-python
    Pillow
    requests
    uuid
    ```
    Install the libraries now:
    ```sh
    pip install -r requirements.txt
    ```
    This shall install all the required libraries.

9. Once this is setup, use the following command to start your wagtail site. This will create a folder named mysite inside the current folder.

    ```sh
    wagtail start mysite
    ```

10. Change directory to the newly created `mysite` folder:

    ```sh
    cd mysite
    ```

11. Run the following commands to start the basic app:

    ```sh
    python manage.py makemigrations
    ```
    ```sh
    python manage.py migrate
    ```
    ```sh
    python manage.py runserver
    ```
    
    > **Note:** These steps start the server after making all the updates. Everytime you change anything in the project, you need to run these 3 commands for the change to reflect on the actual web app, and these are the steps where you may get errors when you have made a mistake or error in the code for the app.

12. Open the link shown on the terminal:
    ```sh
    http://localhost:8000
    ```

    If you see the wagtail page, you're all set.

    Else: Check and follow: [Wagtail Getting Started Tutorial](https://docs.wagtail.org/en/stable/getting_started/tutorial.html)

13. Press `Ctrl+C` in terminal to stop and close the server.

14. Now you need to create a superuser so that you can get into the admin section of the CMS page:
    ```sh
    python manage.py createsuperuser
    ```
    > This will prompt you to create a new superuser account with full permissions. Note the password text won’t be visible when typed, for security reasons.

15. Check if everything is set:
    Do step 11, 12 and then go to:
    ```sh
    http://localhost:8000/admin
    ```
    >This admin page will let you manage the content of the web pages you **create** and dynamically add some things like cards and images for your generic content based pages.

    >Please note that this is **not** the place where you make pages which are AI enabled. Moreover, for content pages, you need to **setup** page types in the code.


## Set up front end pages: 
> explore wagtail documentation and learn wagtail playlist for more details on this!

1. Let's setup some common/content pages for the web app now:

    This is the flex page which is a page template you can use to create multiple instances for content based pages.
    ```sh
    python manage.py startapp flex
    ```
    This is to create a group of blocks which you'd use in your flex page as options to build the page, e.g. rich text box etc...
    ```sh
    python manage.py startapp streams 
    ```
    This is to create the navigation menus
    ```sh
    python manage.py startapp menus
    ```
    
2. Let's setup some pages which would have AI in them, here each page will act as independent app.

    This will be used for deploying AI on a real-time video from camera:
    ```sh
    python manage.py startapp cam_app 
    ```
    This will be used for deploying AI on image uploads from the system:
    ```sh
    python manage.py startapp cam_app_2
    ```

> There is a lot of work to be done! This is just the basic page setup for now!

## Setup Each App:

### Option 1: Copy Paste from my app!

1. Set up streams
    * Setup model.py
    * Setup views.py
    * Setup templates (found inside `mysite>mysite>templates`)

2. Set up menus
    * Setup model.py
    * Setup views.py
    * Setup templates (found inside `mysite>mysite>templates`)

3. Set up flex page
    * Setup model.py
    * Setup views.py
    * Setup templates (found inside `mysite>mysite>templates`)
    > Use CMS (admin) to create and add content to the pages

4. Set up cam_app
    * Setup camera.py `here goes the real time inference code with model`
    * Setup model.py
    * Setup views.py
    * Setup database_operations.py
    * Setup database file
    * Setup templates (found inside `mysite>mysite>templates`)

5. Set up cam_app2
    * Setup model.py `here goes the code for the model and file handling`
    * Setup views.py
    * Setup database_operations.py
    * Setup database file
    * Setup templates (found inside `mysite>mysite>templates`)

6. All set to go! (Make sure to update and modify all the bits to work according to your requirements)

### Option 2: Use Wagtail Videos
1. Use Learn Wagtail Tutorial Videos
2. Use Django for Data Science Videos
3. Use Wagtail, Django Documentation
4. Use Various online resources
5. Make your own system yourself and refer my code for ideas!


# Done!