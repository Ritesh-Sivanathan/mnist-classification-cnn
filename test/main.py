import os

print(os.getcwd())
os.chdir('code')
os.chdir(os.path.join('..', 'data'))
print(os.getcwd())