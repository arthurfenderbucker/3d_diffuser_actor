import os
from subprocess import call
import pickle
from pathlib import Path

import tap


class Arguments(tap.Tap):
    root_dir: Path
    tasks: str = None


def main(root_dir, task):
    variations = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
    seen_variations = {}
    for variation in variations:

        # ignore .DS_Store
        if variation == '.DS_Store':
            continue
        num = int(variation.replace('episode', ''))
        variation = pickle.load(
            open(
                f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_number.pkl',
                'rb'
            )
        )
        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes', exist_ok=True)
        print(variation, num)
        if variation not in seen_variations.keys():
            seen_variations[variation] = [num]
        else:
            seen_variations[variation].append(num)

        if os.path.isfile(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
            data1 = pickle.load(open(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', 'rb'))
            data2 = pickle.load(open(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl', 'rb'))
            assert data1 == data2
        else:
            call(['ln', '-sfn',
                  f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl',
                  f'{root_dir}/{task}/variation{variation}/'])

        ep_id = len(seen_variations[variation]) - 1
        call(['ln', '-sfn',
              "{:s}/{:s}/all_variations/episodes/episode{:d}".format(root_dir, task, num),
              f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'])


if __name__ == '__main__':
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())

    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f] if args.tasks is None else args.tasks.split(',')
    print(tasks)
    for task in tasks:
        main(root_dir, task)
