<div align="center">

<h2 align="center">How to build your own Web App with a machine learning model</h2>
<h3> <strong>Option 2:</strong> Building up on the existing app</h3>
<p>This option is strongly recommended for the students in 42028 Deep Learning and CNNs subject of UTS</p>
</div>

## Steps to setup this app:
1. Create a folder at a desired location in your computer/laptop for this project and change directory to that folder.
    ```sh
    mkdir my_project
    cd my_project
    ```
2. Follow instructions for `Getting Started` from [ReadME](https://github.com/GitarthVaishnav/Basic_Web_App/blob/master/README.md)
3. Follow instructions for `Installation` from [ReadME](https://github.com/GitarthVaishnav/Basic_Web_App/blob/master/README.md)
4. Follow instructions for `Usage` from [ReadME](https://github.com/GitarthVaishnav/Basic_Web_App/blob/master/README.md)
5. You have the working app now!

## How to modify:
The following is the information about each sub folder and files under them:
0. settings (the `mysite` folder referenced below is inside the parent `mysite` folder)
    * mysite/base.py (contains main settings for the app)
    * mysite/dev.py (contains dev level settings for the app)
    * mysite/production.py (contains production level settings for the app)
    * urls.py (contains urls for the app)
    * wsgi.py

1. streams (basic building blocks of content based pages)
    * model.py
    * views.py
    * templates (found inside `mysite>mysite>templates`)

2. menus (basic navigation menu)
    * model.py
    * views.py
    * templates (found inside `mysite>mysite>templates`)

3. flex (flexible - content page template, this uses streams)
    * model.py
    * views.py
    * templates (found inside `mysite>mysite>templates`)
    > Use CMS (admin) to create flex page instances and add content to the created pages

4. Set up cam_app (app for deploying AI on the real time video stream)
    * camera.py (`here is the real time inference code with model`)
    * model.py (contains page related settings)
    * views.py (contains some more page related settings)
    * database_operations.py (database related functions)
    * templates (found inside `mysite>mysite>templates`)

5. Set up cam_app2 (app for deploying AI on the uploaded images)
    * model.py (`here is the code for the model and file handling along with page related settings`)
    * views.py (contains some more page related settings)
    * database_operations.py (database related functions)
    * templates (found inside `mysite>mysite>templates`)