import matplotlib.pyplot as plt
import yaml
import os

def main():

    # Results folders
    # NOTE: It's on you to make sure params are constant except for feature type, and that feature type matches folder
    results_folders = {
        'd2-orb':  './results/d2-orb-homo-false-nov-16-3-epoch-10-range/',
        'orb':     './results/orb-homo-false-range/',
        'd2-sift': './results/d2-sift-homo-false-nov-16-3-epoch-10-range/',
        'sift':    './results/sift-homo-false-range/',
    }

    # Initialize
    thresholds = [1,2,3,4,5,6,7,8,9,10]
    repeatability = {}
    mma = {}

    # Load results file
    for feature_type, folder in results_folders.items():
        results_path = os.path.join(os.getcwd(), folder, 'results.yaml')

        with open(results_path, 'r') as f: results = yaml.load(f, Loader=yaml.FullLoader)
        # Get repeatability
        repeatability[feature_type] = results['repeatability']
        # Get mma
        mma[feature_type] = results['mma']

    # Plot repeatability
    for feature_type, values in repeatability.items(): plt.plot(thresholds, values, label=feature_type)
    plt.ylabel('Repeatability')
    plt.xlabel('Threshold [px]')
    plt.xticks(ticks=thresholds)
    plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.set_xlim([1,10])
    ax.set_ylim([0.0, 1.0])
    plt.show()
    plt.savefig('./plots/repeatability.png')
    plt.cla()
    

    # Plot MMA
    for feature_type, values in mma.items(): plt.plot(thresholds, values, label=feature_type)
    plt.xlabel('Threshold [px]')
    plt.ylabel('MMA')
    plt.title('Modality Change Only (No Geometric Transformation)')
    plt.xticks(ticks=thresholds)
    plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.set_xlim([1,10])
    ax.set_ylim([0.0, 1.0])
    plt.show()
    plt.savefig('./plots/mma.png')
    plt.cla()

if __name__ == "__main__":
    main()