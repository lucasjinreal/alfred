import platform
import os


def mkdir(path):
    e = os.path.exists(path)
    if not e:
        os.makedirs(path)
        return True
    else:
        return False


def mkfile(filePath):
    pipfile = "[global]\ntrusted-host=mirrors.aliyun.com\nindex-url=http://mirrors.aliyun.com/pypi/simple/"
    if os.path.exists(filePath):
        if str(input("File exist!Cover?(Y/N))")).upper() == 'N':
            print("Not Cover.")
            return
    with open(filePath, 'w') as fp:
        fp.write(pipfile)
    print("Write finish.")


def change_pypi_source():
    systype = platform.system()
    print("System type: " + systype)
    if systype == "Windows":
        path = os.path.join(os.getenv('HOMEPATH'), 'pip')
        mkdir(path)
        mkfile(os.path.join(path, 'pip.ini'))
    elif systype == "Linux" or systype == "Darwin":
        path = os.path.join(os.path.expandvars('$HOME'), ".pip")
        mkdir(path)
        mkfile(os.path.join(path, 'pip.conf'))
    else:
        print("System type: " + systype + " Not Support!")
