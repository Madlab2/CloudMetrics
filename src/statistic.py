import numpy as np
import laspy
import classes, helper, metric
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick

def cloud_statistic(type, pc_path_1, pc_path_2=None, pc_path_3=None, name=None):
    output = "Number of Points per relevant Class in Dataset " + str(name) + ":\n"

    pc_1 = laspy.read(pc_path_1)
    points_1 = pc_1.points
    
    points_1_class_wise = metric.class_split_pc(points_1, type=type)
    
    class_point_counts = []
    for class_points in points_1_class_wise:
        class_count = len(class_points)
        class_point_counts.append(class_count)

    # if we pass a second dataset in (e.g. train 2)
    if pc_path_2 is not None:
        pc_2 = laspy.read(pc_path_2)
        points_2 = pc_2.points
        points_2_class_wise = metric.class_split_pc(points_2, type=type)
        
        class_idx = 0
        for class_points in points_2_class_wise:
            class_count = len(class_points)
            class_point_counts[class_idx] += class_count
            class_idx += 1
    
    # if we pass a third dataset in (e.g. train 3)
    if pc_path_3 is not None:
        pc_3 = laspy.read(pc_path_3)
        points_3 = pc_3.points
        points_3_class_wise = metric.class_split_pc(points_3, type=type)
        
        class_idx = 0
        for class_points in points_3_class_wise:
            class_count = len(class_points)
            class_point_counts[class_idx] += class_count
            class_idx += 1
    
    if type == 'real':
        classes_dict = classes.CLASSES_FOR_M3C2_REAL
    else:
        classes_dict = classes.CLASSES_FOR_M3C2_SYNTH

    idx = 0
    keys = list(classes_dict.keys())
    assert(len(keys) == len(class_point_counts))

    for point_count in class_point_counts:
        class_key = keys[idx]
        output += f"\t#Points in class {classes_dict[class_key]}:\t{point_count} | Relative share within this dataset: {round(100 * point_count / np.sum(class_point_counts), 2)}%\n"
        idx += 1

    return output, class_point_counts


def compute_data_distribution(type, path_train_1, path_train_2, path_train_3, path_valid, path_test, display_bar_percentage=True):
    train_output_string, train_point_counts = cloud_statistic(type, path_train_1, path_train_2, path_train_3, 'Train')
    valid_output_string, valid_point_counts = cloud_statistic(type, path_valid, name='Valid')
    test_output_string, test_point_counts = cloud_statistic(type, path_test, name='Test')

    output = train_output_string + "\n\n" + valid_output_string + "\n\n" + test_output_string + "\n\n"

    train_point_counts = np.array(train_point_counts)
    valid_point_counts = np.array(valid_point_counts)
    test_point_counts = np.array(test_point_counts)
    all_point_counts = train_point_counts + valid_point_counts + test_point_counts

    # points of one class in train/valid/test over all points of this class
    relativ_point_counts_train = 100 * train_point_counts / all_point_counts
    relativ_point_counts_valid = 100 * valid_point_counts / all_point_counts
    relativ_point_counts_test = 100 * test_point_counts / all_point_counts

    # number of points of one class in all datasets combined over total number of points in all datasets
    class_shares_total = 100 * all_point_counts / np.sum(all_point_counts)

    output += "Distribution of each class between Train, Validatio, Test:\n"
    output2 = "\nTotal Share of each class over all datasets:\n"
    keys = list(classes.CLASSES_FOR_M3C2_REAL.keys())

    idx = 0
    for idx in range(len(all_point_counts)):
        class_key = keys[idx]
        class_name = classes.CLASSES_FOR_M3C2_REAL[class_key]
        output2 += f"\tClass {class_name}: {round(class_shares_total[idx], 2)}%\n"
        output += f"\tClass {class_name}: Train {round(relativ_point_counts_train[idx], 2)}%, Validation: {round(relativ_point_counts_valid[idx], 2)}%, Test: {round(relativ_point_counts_test[idx], 2)}%\n"
        idx += 1
    
    output += output2

    output_file_path = "../results/statistics_results.txt"
    with open(output_file_path, "w") as file:
        file.write(output)

    # plot and save image
    x_names = list(classes.CLASSES_FOR_M3C2_REAL.values())
    width = 0.4
    bar_plot_inputs = dict()
    
    bar_plot_inputs['Train'] = class_shares_total * relativ_point_counts_train / 100
    bar_plot_inputs['Validation'] = class_shares_total * relativ_point_counts_valid / 100
    bar_plot_inputs['Test'] = class_shares_total * relativ_point_counts_test / 100

    fig, ax = plt.subplots(figsize=(9, 12))
    bottom = np.zeros(len(x_names))
    relative_point_counts = [relativ_point_counts_train, relativ_point_counts_valid, relativ_point_counts_test]
    set_counter = 0
    for names, shares in bar_plot_inputs.items():
        p = ax.bar(x_names, shares, width, label=names, bottom=bottom)
        bottom += shares
        if display_bar_percentage == True:
            # Add percentage labels to each section of the bar  
            for i, rect in enumerate(p):
                x = rect.get_x() + rect.get_width() / 2
                y = rect.get_y() + rect.get_height() / 2
                # hack to avoid numbers overlaying with small bars
                if y < (set_counter + 1):
                    y = (set_counter + 1)
                percentage = relative_point_counts[set_counter][i] / 100
                ax.text(x, y, f"{percentage:.1%}", ha='center', va='top', color='black', fontsize=10)
            
            set_counter += 1

    
    ax.set_title("Classes, their share [%] over all data with relative train, valid, test shares")
    ax.set_ylabel("Absolute Percentage w.r.t entire data")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.legend(loc="upper right")

    plt.savefig("../results/shares.png", dpi=400)

    return output