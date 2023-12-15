import subprocess
import os


def execute_process_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True)
    except subprocess.CalledProcessError:
        print(f"Error al ejecutar {script_path}")
    except FileNotFoundError:
        print(f"El archivo {script_path} no se encontró.")


SCRIPT_TRAIN = os.path.join('src', "process_train.py")
SCRIPT_TEST = os.path.join('src', "process_test.py")
SCRIPT_PARTITIONS = os.path.join('src', "process_partitions.py")
SCRIPT_POSITIONING = os.path.join('src', "positioning_partitions.py")

if __name__ == "__main__":
    execute_process_script(SCRIPT_TRAIN)
    execute_process_script(SCRIPT_TEST)
    execute_process_script(SCRIPT_PARTITIONS)
    execute_process_script(SCRIPT_POSITIONING)