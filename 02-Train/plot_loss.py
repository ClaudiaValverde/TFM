import os
import re
import sys
import argparse
import matplotlib.pyplot as plt

def parse_log_folder(folder_path):
    # Regex pattern to extract loss from filenames
    #pattern = re.compile(r"2025-02-25_*_loss_([\d\.]+)\.pth")
    #pattern = re.compile(r"(2025-02-(28))_(\d{2}-\d{2}-\d{2}).*loss_([\d\.]+)\.pth")
    pattern = re.compile(r"(lr_2025-(?:05-17))_(\d{2}-\d{2}-\d{2}).*loss_([\d\.]+)\.pth")


    loss_list = []
    file_data = []
    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            print(match)
            print(match.group(1))
            print(match.group(2))
            print(match.group(3))
            date = match.group(1)
            time = match.group(2)
            loss = float(match.group(3))
            #print(f"File: {filename}, Loss: {loss}")
            '''
            if loss > 2:
                print(loss)
            '''
            loss_list.append(loss)
            file_data.append((date, time, loss, filename))
    print(len(loss_list))

    file_data.sort()
    print(file_data)
    loss_list = [loss for _,_,loss,_ in file_data]

    print(f"Total files matched: {len(loss_list)}")
    return loss_list


def parse_log_folder2(folder_path):
    # Regex pattern to extract loss from filenames
    #pattern = re.compile(r"2025-02-25_*_loss_([\d\.]+)\.pth")
    #pattern = re.compile(r"(2025-02-(28))_(\d{2}-\d{2}-\d{2}).*loss_([\d\.]+)\.pth")
    #pattern = re.compile(r"(lr_2025-(?:05-17))_(\d{2}-\d{2}-\d{2}).*loss_([\d\.]+)\.pth")
    pattern = re.compile(r"checkpoint_ep(\d+)_loss(\d+\.\d+)\.pth")


    loss_list = []
    file_data = []
    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            print(match)
            print(match.group(1))
            print(match.group(2))
            #print(match.group(3))
            epoch = int(match.group(1))
            loss = float(match.group(2))
            #print(f"File: {filename}, Loss: {loss}")
            '''
            if loss > 2:
                print(loss)
            '''
            loss_list.append(loss)
            file_data.append((epoch, loss, filename))
    print(len(loss_list))

    file_data.sort()
    print(file_data)
    loss_list = [loss for _,loss,_ in file_data]

    print(f"Total files matched: {len(loss_list)}")
    return loss_list

def parse_log_file(folder_path):
    # Regex pattern to extract loss from filenames
    #pattern = re.compile(r"2025-02-25_*_loss_([\d\.]+)\.pth")
    #pattern = re.compile(r"(2025-02-(28))_(\d{2}-\d{2}-\d{2}).*loss_([\d\.]+)\.pth")
    pattern = re.compile(r"Epoch\s+(\d{3})\s+\|\s+loss\s+(\d\.\d{4})")

    loss_list = []
    file_data = []
    # Loop through files in the folder
    with open(folder_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            match = pattern.search(line)
            if match:
                print(match.group(1))
                print(match.group(2))
                #print(match.group(3))
                epoch = match.group(1)
                loss = float(match.group(2))

                loss_list.append(loss)
                file_data.append((epoch, loss, line))
    print(len(loss_list))

    file_data.sort()
    print(file_data)
    loss_list = [loss for _,loss,_ in file_data]

    print(f"Total files matched: {len(loss_list)}")
    print(loss_list)
    return loss_list


def plot_metrics(loss_list, save_dir, file_name): # VQ_loss_list, coords_loss_list,
    # Generate a simple sequence for the x-axis
    epochs = list(range(1, len(loss_list) + 1))

    plt.figure(figsize=(12, 5))

    #plt.subplot(2, 2, 1)
    plt.plot(loss_list, 'darkcyan', label=file_name)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, file_name+'.png')
    try:
        plt.savefig(plot_path)
        plt.close()
        print(f"Training metrics plot saved to '{plot_path}'")
    except Exception as e:
        print(f"Error saving plot to '{plot_path}': {e}")


def plot_metrics_several(loss_dict, save_dir, file_name): # VQ_loss_list, coords_loss_list,
    plt.figure(figsize=(12, 6))
    '''
    for label, loss_list in loss_dict.items():
        epochs = list(range(1, len(loss_list)+1))
        plt.plot(epochs, loss_list, label=label)
    '''
    colors = ['#127478', '#2598d9', '#D55E00', '#9436c9', '#E69F00', '#009E73', '#0072B2'][:len(loss_dict)]  # Example default palette

    for (i, (label, loss_list)) in enumerate(loss_dict.items()):
        #epochs = list(range(1, len(loss_list) + 1))
        epochs = list(range(1,31))
        plt.plot(epochs, loss_list[:30], label=label, color=colors[i % len(colors)])  # Use color from list
    
    plt.title('Training Losses over Epochs for all selected models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


    # Force axes to start at 0
    plt.xlim(left=0)
    #plt.ylim(bottom=0)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, file_name + '.png')
    try:
        plt.savefig(plot_path)
        plt.close()
        print(f"Training metrics plot saved to '{plot_path}'")
    except Exception as e:
        print(f"Error saving plot to '{plot_path}': {e}")

def plot_split_comparisons(loss_dict, save_dir, file_name):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Extract lists
    epochs = list(range(1, 31))
    esm2_virus = loss_dict['ESM2-prot only with virus'][:30]
    esm2_no_virus = loss_dict['ESM2 without virus'][:30]
    esm2_pocket = loss_dict['ESM2-pocket'][:30]
    saprot_pocket = loss_dict['SaProt-pocket'][:30]
    tensordti = loss_dict['TensorDTI'][:30]

    # --- Common style ---
    esm2_color = '#127478'
    comparison_colors = ['#D55E00', '#c1cc2d', '#9436c9']

    # --- Left Plot: ESM2 with vs. without virus ---
    axs[0].plot(epochs, esm2_virus, label='ESM2-prot only (with virus)', color=esm2_color, linestyle='-')
    axs[0].plot(epochs, esm2_no_virus, label='ESM2-prot only (no virus)', color=esm2_color, linestyle='--')

    axs[0].set_title('Effect of Virus Inclusion on ESM2')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(left=0)

    # --- Right Plot: Virus-inclusive models vs ESM2 (reference) ---
    axs[1].plot(epochs, esm2_virus, label='ESM2-prot only (with virus)', color=esm2_color, linestyle='-')
    axs[1].plot(epochs, esm2_pocket, label='ESM2-pocket', color=comparison_colors[0])
    axs[1].plot(epochs, saprot_pocket, label='SaProt-pocket', color=comparison_colors[1])
    axs[1].plot(epochs, tensordti, label='TensorDTI', color=comparison_colors[2])

    axs[1].set_title('Selected Models vs ESM2 Reference')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(left=0)

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, file_name + '.png')
    try:
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Comparison plots saved to '{plot_path}'")
    except Exception as e:
        print(f"Error saving plot to '{plot_path}': {e}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='Info is a filename or a doc? "Folder" or "File" or "Several')
    parser.add_argument('--log_folder', type=str, help='Path to the training log folder.')
    parser.add_argument('--save_dir', type=str, default='plots')
    parser.add_argument('--out_dir', type=str, default='Path for output plots .png')
    args = parser.parse_args()

    log_file_path = args.log_folder
    save_directory = args.save_dir
    file_name = args.out_dir

    '''
    if not os.path.isfolder(log_file_path):
        print(f"Error: The file '{log_file_path}' does not exist.")
        sys.exit(1)
    '''

    if args.type == 'Folder':

        losses = parse_log_folder2(log_file_path)
    
    if args.type == 'File':
        losses = parse_log_file(log_file_path)
    
    if args.type == 'Several':
        
        
        log_file_path2 = 'checkpoints/weighted_add_ESM2_1_0'
        log_file_path3 = 'checkpoints/weighted_add_ESM2_1_0_novirus'
        log_file_path4 = 'checkpoints/concat_add_ESM2_pocket'
        log_file_path1 = 'checkpoints/concat_add_saprot_pocket'
        log_file_path5 = 'checkpoints/concat_tensorDTI'
        #log_file_path6 = 'checkpoints/weighted_add_tensorDTIlast_512_1_0'

        #losses_0_1 = parse_log_folder2(log_file_path1) 'weighted_add: 0-1': losses_0_1,
        losses_esm2 = parse_log_folder2(log_file_path2)
        losses_esm2pocket = parse_log_folder2(log_file_path4)
        losses_saprot = parse_log_folder2(log_file_path1)
        losses_tensor = parse_log_folder2(log_file_path5)
        losses_novirus = parse_log_folder2(log_file_path3)
        #losses_concat4 = parse_log_folder2(log_file_path6)

        losses_to_plot = {'ESM2 without virus':losses_novirus, 'ESM2-prot only with virus':losses_esm2, 'ESM2-pocket':losses_esm2pocket ,'SaProt-pocket':losses_saprot, 'TensorDTI':losses_tensor}
        
        '''
        plot_metrics_several(
        loss_dict=losses_to_plot,
        save_dir = save_directory,
        file_name = file_name
        )
        '''
        plot_split_comparisons(
        loss_dict=losses_to_plot,
        save_dir = save_directory,
        file_name = file_name
        )



    '''
    if not epochs:
        print("No epoch data found in the log file.")
        sys.exit(1)

    sorted_data = sorted(zip(epochs, losses, perplexities), key=lambda x: x[0])
    epochs, losses, perplexities = zip(*sorted_data)

    print("Parsed Data:")
    print(f"Epochs       : {epochs}")
    print(f"Average Loss : {losses}")
    print(f"Average PPL  : {perplexities}")
    print(f"Average VQ loss  : {VQ_losses}")
    print(f"Average Coords loss  : {Coords_Losses}")
    
    print("Parsed Data:")
    print(f"Epochs       : {len(epochs)}")
    print(f"Average Loss : {len(losses)}")
    print(f"Average PPL  : {len(perplexities)}")
    print(f"Average VQ loss  : {len(VQ_losses)}")
    print(f"Average Coords loss  : {len(Coords_Losses)}")

    '''

    '''
    plot_metrics(
        loss_list=list(losses),
        save_dir = save_directory,
        file_name = file_name
    )
    '''


if __name__ == "__main__":
    main()
