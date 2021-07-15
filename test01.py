# can check to see if pip is installed
# python3 -m pip --version


# To upgrade the pip module,
# python3 -m pip install --user -U pip


# Creating an Isolated Environment
# python3 -m pip install --user -U virtualenv


# Now you can create an isolated Python environment by typing this:
# cd $ML_PATH
# python3 -m virtualenv my_env


# Now every time you want to activate this environment,
# cd $ML_PATH
# source my_env/bin/activate

# To deactivate this environment, type deactivate.  

# Now you can install all the required modules and their dependencies 
# using this simple pip command

# python3 -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn

# If you created a virtualenv, you need to register it to Jupyter and give it a name:
# python3 -m ipykernel install --user --name=python3
# jupyter notebook

# A Jupyter server is now running in your terminal, listening to port 8888. 
# You can visit this server by opening your web browser to http://localhost:8888/