import platform
import sys
def get_chiya_path():
    if platform.system() == 'Windows':
        path = sys.path[0].split('\\')[:-1]
        # path.append('chiya')
        path = '\\'.join(path)
    else:
        path = sys.path[0].split('/')[:-1]
        path.append(['chiya'])
        path = '/'.join(path)
    return path