# Animal Image Classifier Web Application
![image](https://github.com/Prithvidhar/AnimalClassifyingWebApp/assets/62158749/a4326312-16f0-49dc-8409-4562c7c3d8ef)
This application allows users to upload images of wild animals and have my image classifier guess what animal the user's image contains
## Installation
1. To use this application, it is recommended to clone this repository to your local machine. Dowloading the repository as a zip file also works, make sure you extract the files to your prefered directory.
2. Once the code is on your system, open the terminal application (Command Prompt for Windows users) and go into the web application's directory using the command below
3. `cd 'path to web app'`
4. This application requires Django and PyTorch to be installed on your system. It is recommended that you use a package distribution service like [Anaconda](https://anaconda.org/) to install these packages. Follow the tutorial on the site to set up an anaconda virtual environment.
5. If not installed, use these commands to install Django and PyTorch: `conda install -c conda-forge django` and `conda install -c pytorch pytorch`
6. Now run the web server using this command: `python manage.py runserver`
7. After a few checks, a link should appear. Clicking the link will launch the application on your system's default web browser.
## Usage
To use the app, simply click on the **Choose File** button and select an image to upload. Once selected, clicked on the **Upload** button to get a prediction.
This can done any number of times, so feel free to have fun categorizing animals!

### Technologies used
- [PyTorch](https://pytorch.org/)
- [Django](https://www.djangoproject.com/)
- [Bootstrap 5](https://getbootstrap.com/docs/5.0/getting-started/introduction/)
