################################################# PACKAGES ####################################################################
from subprocess import check_call, CalledProcessError, run as runSubprocess
from os.path import exists
from os import getenv
#from getpass import getpass


################################################# METHODS ##############################################################################3
def ensure_pipenv_installed():
    try:
        check_call(['pipenv', '--version'])
        print('pipenv is installed\n')
    except CalledProcessError:
        print('pipenv not found. Install pipenv...')
        check_call(['pip', 'install', 'pipenv'])

def manage_and_use_env():
    if not exists('Pipfile'):
        print('Pipfile not exist. Initializing pipenv environment...\n')
        check_call(['pipenv', 'install'])
    else:

        print('Pipfile exists. Environment ready.\n')


#Function to install a single package using pipenv
def install_package_with_pipenv(package):
    try:
        check_call(['pipenv', 'install', package])
    except CalledProcessError:
        try:
            runSubprocess(f'pip install {package}', shell=True, check=True)
            
            runSubprocess(f'pipenv install {package}', shell=True, check=True)
        except Exception:
            print('\nNot possible\n')

#Function to install all packages from a requirements.txt file using pipveng
def install_packages_from_file_with_pipenv(file):
    check_call(['pipenv', 'install', '-r', f'{file}.txt'])

def run_script(file):
    try:
        runSubprocess(['pipenv', 'run', 'python', f'{file}.py'])
    except CalledProcessError as e:
        print(f'An error occurred: {e.stderr.decode()}')



def upload_docker():
    #username = input('Enter your Docker username: ')
    username = getenv('DOCKER_USERNAME', default='default_username')
    pwd = getenv('DOCKER_PASSWORD', default='default_password')
    #pwd = getpass('Enter your Docker password: ')

    try:
        print('\nLogging...\n')
        runSubprocess(['docker', 'login', '--username', username, '--password', pwd], check=True)

        dockerfile_contents = f"""
#Use the official image of Python
FROM python:3.11.0-slim

#Establised your work directory
WORKDIR /app

#Install pipenv
RUN pip install pipenv

#Copy our Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock /app/

#Installing depends in the system
RUN pipenv install --system --deploy

#Install Jupyter
RUN pip install jupyter ipykernel

#Copy all the files
COPY . /app

#Expose the port 8888
EXPOSE 8888

ENV NAME PipEnvironment

CMD pipenv run python pipenDockerGit.py
    """
        image_name = input('Enter the name of your image: ')

        print('\nWriting Dockerfile\n')
        with open('Dockerfile', 'w') as file:
            file.write(dockerfile_contents)
            file.close()
        print('\nBuilding image...\n')
        runSubprocess(f'docker build -t {image_name} .', shell=True, check=True)
        print('\nImage built.\n')
        runSubprocess(f'docker push {image_name}', shell=True, check=True)
        print('\nImage uploaded to DockerHub.\n')


    except CalledProcessError as cp:
        print(f'CalledProcessError: {cp.stderr}')
    except Exception as e:
        print(f'Exception: {e}')

def upload_github():
    try:
        print('\nemail\n')
        email = getenv("GITHUB_EMAIL", default='default_email')
        runSubprocess(f'git config --global user.email "{email}"',
                      shell=True, check=True)
        print('\nname')
        username=getenv("GITHUB_USERNAME", default='default_username')
        runSubprocess(f'git config --global user.name "{username}"',
                      shell=True, check=True)
        runSubprocess('git init', shell=True, check=True)
        print('\nInitializing Github & git status\n')
        runSubprocess('git status', shell=True, check=True)
        print('\ngit add .\n')
        runSubprocess('git add .', shell=True, check=True)
        print('\ngit commit\n')
        commit=input('Enter commit message: ')
        runSubprocess('git commit -m "test commit"', shell=True, check=True)
        print('\ngit branch\n')
        runSubprocess('git branch -M main', shell=True, check=True)
        first_commit = '3'
        while first_commit not in ['Y', 'y', 'N', 'n']:
            first_commit = input('If your first commit? [Y/N]: ')
            if first_commit not in ['Y', 'y', 'N', 'n']:
                print('\nInvalid option\n')
        if first_commit == ['Y', 'y']:
            repo = input('Enter your repository name: ')
            my_git = f'https://github.com/pyCampaDB/{repo}.git'
            runSubprocess(f'git remote add origin {my_git}',
                                 shell=True, check=True, capture_output=True)
            print('\nremote add origin\n')
        else:
            runSubprocess('git pull origin main', shell=True, check=True)
            print('\npull\n')
        #
        print('\npushing...\n')
        runSubprocess(f'git push -u origin main', shell=True, check=True)
        print('\nProject uploaded to GitHub\n')
    except CalledProcessError as cp:
        print(f'\nCalledProcessError: {cp.stderr}\n')
    except Exception as e:
        print(f'Exeption: {e}')


def run():
    ensure_pipenv_installed()
    manage_and_use_env()
    option = '3'
    while option not in ['1', '2']:
        option = input('\n1. Run script'
                       '\n2. Settings pipenv'
                       '\nEnter your choice: ')
        if option not in ['1', '2']:
            print('\ninvalid option\n')
    if option == '2':
        menu = '1'
        while menu in ['1', '2']:
            menu = input('\n1. Install an only package'
                         '\n2. Install all packages written in the file'
                         '\n(Other). Exit setting pipenv and run script'
                         '\nEnter your choice: ')
            if menu=='1':
                package = input('\nEnter package name: ')
                install_package_with_pipenv(package)
            elif menu=='2':
                file = input('\nEnter the file name: ')
                install_packages_from_file_with_pipenv(file)
            else:
                break
    from dotenv import load_dotenv
    load_dotenv()
    file = input('\nEnter file name: ')
    run_script(file)

    docker_option = '9'
    while docker_option not in ['Y', 'y', 'N', 'n']:
        docker_option = input('Do you want to upload this project to Docker? [Y/N]: ')
        if docker_option not in ['Y', 'y', 'N', 'n']:
            print('\nInvalid option\n')
    if docker_option in ['Y', 'y']:
        upload_docker()
    else:
        print('\nDocker pass...\n')

    git_option = '9'
    while git_option not in ['Y', 'y', 'N', 'n']:
        git_option = input('Do you want to upload this project to GitHub? [Y/N]: ')
        if git_option not in ['Y', 'y', 'N', 'n']:
            print('\nInvalid option\n')
    if git_option in ['Y', 'y']:
        upload_github()
    else:
        print('\nGit pass...\n')

############################################# MAIN ##########################################################################
if __name__ == '__main__':
    run()


