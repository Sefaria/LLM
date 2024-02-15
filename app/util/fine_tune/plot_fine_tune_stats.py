import csv
import matplotlib.pyplot as plt
import typer


def plot_data(stats_csv_path: str, plot_png_path: str):
    # Lists to store data
    steps = []
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_mean_token_accuracy = []

    # Read data from CSV
    with open(stats_csv_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            steps.append(int(row['step']))
            train_loss.append(float(row['train_loss']))
            train_accuracy.append(float(row['train_accuracy']))
            if row['valid_loss'] == '':
                valid_loss.append(None)
            else:
                valid_loss.append(float(row['valid_loss']))
            if row['valid_mean_token_accuracy'] == '':
                valid_mean_token_accuracy.append(None)
            else:
                valid_mean_token_accuracy.append(float(row['valid_mean_token_accuracy']))

    # Plotting
    plt.figure(figsize=(12, 8))

    # Training Loss and Validation Loss
    plt.subplot(2, 2, 1)
    steps_filtered = [s for s, v_l in zip(steps, valid_loss) if v_l is not None]
    train_loss_filtered = [t_l for t_l, v_l in zip(train_loss, valid_loss) if v_l is not None]
    valid_loss_filtered = [acc for acc in valid_loss if acc is not None]
    plt.plot(steps_filtered, train_loss_filtered, label='Training Loss', marker='o')
    plt.plot(steps_filtered, valid_loss_filtered, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    # Training Accuracy and Validation Token Accuracy
    plt.subplot(2, 2, 2)
    # plt.plot(steps, train_accuracy, label='Training Accuracy', marker='o', color='orange')
    # Remove None values from the data
    steps_filtered = [s for s, acc in zip(steps, valid_mean_token_accuracy) if acc is not None]
    accuracy_filtered = [acc for acc in valid_mean_token_accuracy if acc is not None]
    plt.plot(steps_filtered, accuracy_filtered, label='Validation Token Accuracy', marker='o', color='green')
    plt.title('Training Accuracy and Validation Token Accuracy Over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.savefig(plot_png_path)

if __name__ == '__main__':
    typer.run(plot_data)