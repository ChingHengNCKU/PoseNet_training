import argparse
import os
import numpy as np

def main():
    """
    Args:
        name (str):
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='',
                        metavar='str', required=True)
    args = parser.parse_args()
    name = args.name

    errors_list = []
    print('[pme (m), ome (degree)]\n')
    for root, dirs, files in os.walk('results'):
        if name in os.path.basename(root):
            for f in files:
                if 'test_pme.txt' in f:
                    pme = np.loadtxt(os.path.join(root, f))
                elif 'test_ome.txt' in f:
                    ome = np.loadtxt(os.path.join(root, f))
            errors_list.append([pme.item(), ome.item()])
            print(errors_list[-1])
    test_result = np.array(errors_list).mean(axis=0)
    np.savetxt(f'results/{name}.test_result.txt', test_result)
    print(f'\n{list(test_result)}')

if __name__ == '__main__':
    main()
