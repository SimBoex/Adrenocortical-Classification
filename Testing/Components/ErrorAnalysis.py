import numpy as np

'''
This returns the list of the ID of patients that are TRUE NEGATIVE
'''
def IDs_TN(predictions, y_test, id_test):
    ## Checking the TN
    index = (predictions == 0) & (y_test == 0)
    IDs_TN = list(id_test[index])
    
'''
This returns the list of the ID of patients that are FALSE NEGATIVE
'''
def IDs_FN(predictions, y_test, id_test):
    # Checking the FalseNegative
    index = (predictions == 0) & (y_test == 1)
    IDs_FN = list(id_test[index])
    return IDs_FN 
    
'''
This returns the list of the ID of patients that are FALSE POSITIVE
'''   
def IDs_FP(predictions,y_test, id_test):
    ## Checking the FalsePositive
    index = (predictions == 1) & (y_test == 0)
    IDs_FP = list(id_test[index])
    return IDs_FP
    
    
'''
This returns the list of true subtypes given a list of IDs
'''
def SubTypes(pred_ids, id_test, morf_codificata):
    matches = id_test.isin(pred_ids)
    indices = np.where(matches)[0]
    corresponding_values = morf_codificata.iloc[indices]
    corresponding_values
    
    
    
#######################################

import matplotlib.pyplot as plt
import pandas as pd

'''
This returns a TABLE where there is an X on the corrisponding true subtypes of the FN

INPUT: 
    morf_codificata is an ordered list of subtypes
    FoundFN_IDs is a list of ids that are FN
    ids is the list of all the id of patients
'''
def TableSubTypesFN(morf_codificata, FoundFN_IDs, ids,size):
    # Create a DataFrame for the plot
    unique_values = morf_codificata.unique()
    sorted_idsFN = sorted(FoundFN_IDs, key=lambda x: int(x.split('_')[1]))
    table_df = pd.DataFrame({'patient':sorted_idsFN  })

    # Add columns for each unique value in morf_codificata_test and initialize with empty strings
    for value in sorted(map(int,unique_values)):
        table_df[value] = ''

    # Populate the DataFrame with 'x' for matches
    for patient in FoundFN_IDs:
        if patient in ids.values:
            morf_value = morf_codificata[ids == patient].values[0]  
            if morf_value in unique_values:
                table_df.loc[table_df['patient'] == patient, morf_value] = 'x'
            else:
                print("there is a problem!")
        else:
            print("There is a problem!")
            
    if size == "Small":
        dim = (12, 8)
    else:
        dim = (18, 14)

    # Plotting the table without row labels
    fig, ax = plt.subplots(figsize=dim)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
    plt.title("Multi FN Table")

    description = "This table shows the distribution of false negatives among different patient categories." + "\n" + "Knowing that: {2,3,4} are  the classes associated with the label 1"+"\n"+ "(so the FN can be only from these classes )"
    fig.text(0.5, 0.7, description, ha='center', va='center', wrap=True, fontsize=12)  

    plt.show()
    
    
'''
    This returns a TABLE where there is an X on the corrisponding true subtypes of the FP
INPUT: 
    morf_codificata is an ordered list of subtypes
    FoundFN_IDs is a list of ids that are FN
    ids is the list of all the id of patients
'''
def TableSubTypesFP(morf_codificata, FoundFP_IDs, ids, size):
    # Create a DataFrame for the plot
    unique_values = morf_codificata.unique()
    sorted_idsFP = sorted(FoundFP_IDs, key=lambda x: int(x.split('_')[1]))

    table_df = pd.DataFrame({'patient': sorted_idsFP})

    # Add columns for each unique value in morf_codificata_test and initialize with empty strings
    for value in sorted(map(int,unique_values)):
        table_df[value] = ''

    # Populate the DataFrame with 'x' for matches
    for patient in FoundFP_IDs:
        if patient in ids.values:
            morf_value = morf_codificata[ids == patient].values[0]  # Assuming each id_test is unique
            if morf_value in unique_values:
                table_df.loc[table_df['patient'] == patient, morf_value] = 'x'
            else:
                print("there is a problem!")
        else:
            print("There is a problem!")

    if size == "Small":
        dim = (12, 8)
    else:
        dim = (18, 14)
    
    # Plotting the table without row labels
    fig, ax = plt.subplots(figsize=dim)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
    plt.title("Multi FP Table")

    description = "This table shows the distribution of false positives among different patient categories." + "\n" + "Knowing that: {0,1,5} are  the classes associated with the label 0"+"\n"+ "(so the FP can be only from these classes )"
    fig.text(0.5, 0.8, description, ha='center', va='center', wrap=True, fontsize=12)  # Adjust the y-coordinate and fontsize as needed

    plt.show()
    
    
    
    
'''
INPUT: PREDICTIONS, LABELS, AND SUBTYPES
OUTPUT: NONE (SHOWING THE STACKED HISTOGRAM OF ERRORS)
'''
def StackedHist(predictions, y_test,morf_codificata_test):
    # Calculate indices for correct predictions, false positives, and false negatives
    cp_index = (predictions == y_test)
    fp_index = (predictions == 1) & (y_test == 0)
    fn_index = (predictions == 0) & (y_test == 1)

    # Filter labels for CP, FP, and FN
    cp_labels = pd.Series(morf_codificata_test[cp_index])
    fp_labels = pd.Series(morf_codificata_test[fp_index])
    fn_labels = pd.Series(morf_codificata_test[fn_index])

    # Get value counts for CP, FP, and FN
    cp_counts = cp_labels.value_counts().sort_index()
    fp_counts = fp_labels.value_counts().sort_index()
    fn_counts = fn_labels.value_counts().sort_index()

    # Ensure all labels are present in each series
    all_labels = sorted(set(cp_counts.index) | set(fp_counts.index) | set(fn_counts.index))
    cp_counts = cp_counts.reindex(all_labels, fill_value=0)
    fp_counts = fp_counts.reindex(all_labels, fill_value=0)
    fn_counts = fn_counts.reindex(all_labels, fill_value=0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))  # Width: 12 inches, Height: 10 inches

    # Stack CP, FP, and FN
    bars_cp = ax.bar(all_labels, cp_counts, label='Correct Predictions', color='green')
    bars_fp = ax.bar(all_labels, fp_counts, bottom=cp_counts, label='False Positives', color='orange')
    bars_fn = ax.bar(all_labels, fn_counts, bottom=cp_counts + fp_counts, label='False Negatives', color='blue')

    # Annotate each segment with its count
    def annotate_bars(bars, prev_heights=None, fontsize=10):
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            y_position = height / 2 if prev_heights is None else prev_heights[idx] + height + 2
            if height > 0:
                ax.annotate(f'{height}', (bar.get_x() + bar.get_width() / 2, y_position), ha='center', va='center', fontsize=fontsize,color='black')

    annotate_bars(bars_cp, fontsize=17)  # Increase font size for annotations
    annotate_bars(bars_fp, cp_counts.values, fontsize=17)
    annotate_bars(bars_fn, (cp_counts + fp_counts).values, fontsize=17)

    # Annotate each bin with the total count# Annotate each bin with the total count
    max_total  = 100
    total_offset = 0.1 * max_total  # You can adjust this value to change the space between the bar and the total count
    for idx, label in enumerate(all_labels):
        total = cp_counts[label] + fp_counts[label] + fn_counts[label]
        ax.annotate(f'Total: {total}', (idx, total + total_offset), ha='center', fontsize=12, color='black')  # Move total count further from the bar

    # Adjust y-limit based on the maximum total count
    max_total = max(cp_counts + fp_counts + fn_counts)
    ax.set_ylim([0, max_total + 0.1 * max_total])

    # Adding labels and title
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Stacked Histogram of Predictions')

    # Show legend and plot
    plt.legend()
    plt.tight_layout()
    plt.show()

        


def StackedHist_Hu(predictions, y_test,morf_codificata_test):
    # Calculate indices for correct predictions, false positives, and false negatives
    cp_index = (predictions == y_test)
    fp_index = (predictions == 1) & (y_test == 0)
    fn_index = (predictions == 0) & (y_test == 1)

    # Filter labels for CP, FP, and FN
    cp_labels = pd.Series(morf_codificata_test[cp_index])
    fp_labels = pd.Series(morf_codificata_test[fp_index])
    fn_labels = pd.Series(morf_codificata_test[fn_index])

    # Get value counts for CP, FP, and FN
    cp_counts = cp_labels.value_counts().sort_index()
    fp_counts = fp_labels.value_counts().sort_index()
    fn_counts = fn_labels.value_counts().sort_index()



    # Ensure all labels are present in each series
    all_labels = sorted(set(cp_counts.index) | set(fp_counts.index) | set(fn_counts.index))
    cp_counts = cp_counts.reindex(all_labels, fill_value=0)
    fp_counts = fp_counts.reindex(all_labels, fill_value=0)
    fn_counts = fn_counts.reindex(all_labels, fill_value=0)



    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 12))

    # Stack CP, FP, and FN, aligning bars with the x-axis using a range based on the number of labels
    bars_cp = ax.bar(np.arange(len(all_labels)), cp_counts, label='Correct Predictions', color='green')
    bars_fp = ax.bar(np.arange(len(all_labels)), fp_counts, bottom=cp_counts, label='False Positives', color='orange')
    bars_fn = ax.bar(np.arange(len(all_labels)), fn_counts, bottom=cp_counts + fp_counts, label='False Negatives', color='blue')

    # Annotate each segment with its count
    def annotate_bars(bars, prev_heights=None, fontsize=10):
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            y_position = height / 2 if prev_heights is None else prev_heights[idx] + height + 1
            if height > 0:
                ax.annotate(f'{height}', (bar.get_x() + bar.get_width() / 2, y_position), ha='center', va='center', fontsize=fontsize, color='black')

    annotate_bars(bars_cp, fontsize=17)
    annotate_bars(bars_fp, cp_counts.values, fontsize=17)
    annotate_bars(bars_fn, (cp_counts + fp_counts).values, fontsize=17)

    # Annotate each bin with the total count
    max_total = max(cp_counts + fp_counts + fn_counts)
    total_offset = 0.1 * max_total
    for idx, label in enumerate(all_labels):
        total = cp_counts[label] + fp_counts[label] + fn_counts[label]
        ax.annotate(f'Total: {total}', (idx, total + total_offset), ha='center', fontsize=12, color='black')

    max_total = max(cp_counts + fp_counts + fn_counts) + 2

    # Adjust y-limit based on the maximum total count
    ax.set_ylim([0, max_total + 0.1 * max_total])

    # Set the x-ticks to align with the bars and label them accordingly
    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_xticklabels(all_labels)

    # Adding labels and title
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Stacked Histogram of Predictions')

    # Show legend and plot
    plt.legend()
    plt.tight_layout()
    plt.show()



def Histogram_MultiClass(predictions, y_test ):
    index = (predictions != y_test)
    filtered_labels = y_test[index]
    label_counts = filtered_labels.value_counts().sort_index(axis=0)
    total_counts = y_test.value_counts()
    ax = label_counts.plot(kind='bar', figsize=(10, 8))
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Histogram of Label Counts for Mismatches')

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    totals_summary = "\n".join([f"Label {label}: {total} total" for label, total in total_counts.items()])
    props = dict(boxstyle='round', alpha=0.5, facecolor='wheat')
    ax.text(1.05, 0.95, totals_summary, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()
    
    
    
    
def stackedHistSec(predictions, y_test, morf_codificata_test):
    # Calculate indices for correct predictions, false positives, and false negatives
    cp_index = (predictions == y_test)
    fp_index = (predictions == 1) & (y_test == 0)
    fn_index = (predictions == 0) & (y_test == 1)

    # Filter labels for CP, FP, and FN
    cp_labels = pd.Series(morf_codificata_test[cp_index])
    fp_labels = pd.Series(morf_codificata_test[fp_index])
    fn_labels = pd.Series(morf_codificata_test[fn_index])

    # Get value counts for CP, FP, and FN
    cp_counts = cp_labels.value_counts().sort_index()
    fp_counts = fp_labels.value_counts().sort_index()
    fn_counts = fn_labels.value_counts().sort_index()

    # Ensure all labels are present in each series
    all_labels = sorted(set(cp_counts.index) | set(fp_counts.index) | set(fn_counts.index))
    cp_counts = cp_counts.reindex(all_labels, fill_value=0)
    fp_counts = fp_counts.reindex(all_labels, fill_value=0)
    fn_counts = fn_counts.reindex(all_labels, fill_value=0)


    fig, ax = plt.subplots(figsize=(12, 10))  

    # Stack CP, FP, and FN
    bars_cp = ax.bar(all_labels, cp_counts, label='Correct Predictions', color='green')
    bars_fp = ax.bar(all_labels, fp_counts, bottom=cp_counts, label='False Positives', color='orange')
    bars_fn = ax.bar(all_labels, fn_counts, bottom=cp_counts + fp_counts, label='False Negatives', color='blue')

    # Annotate each segment with its count
    def annotate_bars(bars, prev_heights=None, fontsize=10):
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            y_position = height / 2 if prev_heights is None else prev_heights[idx] + height + 2
            if height > 0:
                ax.annotate(f'{height}', (bar.get_x() + bar.get_width() / 2, y_position), ha='center', va='center', fontsize=fontsize,color='black')

    annotate_bars(bars_cp, fontsize=17)  
    annotate_bars(bars_fp, cp_counts.values, fontsize=17)
    annotate_bars(bars_fn, (cp_counts + fp_counts).values, fontsize=17)

    max_total  = 100
    total_offset = 0.1 * max_total  
    for idx, label in enumerate(all_labels):
        if label == 5:
            idx= idx + 1
        total = cp_counts[label] + fp_counts[label] + fn_counts[label]
        ax.annotate(f'Total: {total}', (idx, total + total_offset), ha='center', fontsize=12, color='black')  

    max_total = max(cp_counts + fp_counts + fn_counts)
    ax.set_ylim([0, max_total + 0.1 * max_total])

    # Adding labels and title
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Stacked Histogram of Predictions')

    # Show legend and plot
    plt.legend()
    plt.tight_layout()
    plt.show()
