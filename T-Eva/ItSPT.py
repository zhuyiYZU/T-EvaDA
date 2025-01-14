import subprocess
import shutil
import os

class ItSPT:
    def __init__(self, dataset, div, itera):
        self.dataset = dataset
        self.div = div
        self.itera = itera

        # Define file paths
        self.source_file_train = f'/home/ubuntu/Lizhenglong/T-Eva/datasets/{dataset}/train_all.csv'
        self.target_file_train = f'/home/ubuntu/Lizhenglong/T-Eva/datasets/{dataset}/train.csv'
        self.source_file_test = f'/home/ubuntu/Lizhenglong/T-Eva/datasets/{dataset}/test_all.csv'
        self.target_file_test = f'/home/ubuntu/Lizhenglong/T-Eva/datasets/{dataset}/test.csv'

    def copy_files(self):
        """Copy source train and test files to target locations."""
        shutil.copy(self.source_file_train, self.target_file_train)
        shutil.copy(self.source_file_test, self.target_file_test)
        print('Files copied successfully.')

    def run_command(self, cmd):
        """Run a shell command."""
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing: {cmd}\n{e}")

    def process_1(self):
        """Run the series of commands to process the dataset."""
        self.copy_files()

        # Define the commands
        def generate_commands():
            cmd = f"python ./ItSPT_utils/fewshot.py --result_file ./output_f1.txt --dataset {self.dataset} --template_id 0 --seed 120 --shot 20 --verbalizer manual"
            cmd1 = f"python ./ItSPT_utils/1adjust.py --dataset {self.dataset}"
            cmd2 = f"python ./ItSPT_utils/2div_train.py --dataset {self.dataset} --div {self.div}"
            cmd3 = f"python ./ItSPT_utils/3rep_train.py --dataset {self.dataset} --itera {self.itera} --div {self.div}"
            cmd4 = f"python ./ItSPT_utils/4cover.py --dataset {self.dataset} --itera {self.itera} --div {self.div}"
            return [cmd, cmd1, cmd2, cmd3, cmd4]

        commands = generate_commands()

        for command in commands:
            self.run_command(command)

    def process_2(self):
        """Run optional steps (if needed)."""
        cmd5 = f"python 5together.py --dataset {self.dataset} --itera {self.itera} --div {self.div}"
        cmd6 = f"python 6final_train.py --dataset {self.dataset} --itera {self.itera} --div {self.div}"

        self.run_command(cmd5)
        self.run_command(cmd6)

