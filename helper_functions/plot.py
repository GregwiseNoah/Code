import matplotlib.pyplot as plt


def background_selector(flag):
    if flag == 'dark':
        plt.style.use('dark_background')
    elif flag == 'light':
        plt.style.use('default')
        #plt.style.use('seaborn-v0_8-pastel')