import matplotlib.pyplot as plt

def plot(x,y,title,xlabel,ylabel,show=True):
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.stem(x,y)
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show() 