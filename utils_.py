import numpy as np 
import matplotlib.pyplot as plt
import _pickle as cPickle
import itertools




def plot_confusion_matrix(cm, class_list,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground True Activities')
    plt.xlabel('Predicted Activities')

def plotCNNStatistics(statistics_path, test):

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
 
    #print(statistics_dict)
    bal_alpha = 0.3
    test_alpha = 1.0
    bal_map = np.array([statistics['Trainloss'] for statistics in statistics_dict['Trainloss']])    # (N, classes_num)
    test_map = np.array([statistics['Testloss'] for statistics in statistics_dict['Testloss']])    # (N, classes_num)
    test_f1 = np.array([statistics['test_f1'] for statistics in statistics_dict['test_f1']])    # (N, classes_num)

    # val_map = np.array([statistics['val_f1'] for statistics in statistics_dict['val_f1']])

    #print(bal_map)
    #print(test_map)
    line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
    line, = ax.plot(test_map, color='r', alpha=test_alpha)
    # line, = ax.plot(val_map, color='g', alpha=test_alpha)

    lines.append(line)


    ax.set_ylim(0, 1.)
    #ax.set_xlim(0, len(iterations))
    #ax.xaxis.set_ticks(np.arange(0, len(iterations), 25))
    #ax.xaxis.set_ticklabels(np.arange(0, max_plot_iteration, 50000))
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(labels=['Training Loss','Testing Loss'], loc=2)
    plt.title('{}'.format(test.name))

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(test_f1, color='r', alpha=test_alpha)
    ax.set_ylim(0,1.)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    plt.ylabel('Test Average Fscore')
    plt.title('{}'.format(test.name))
