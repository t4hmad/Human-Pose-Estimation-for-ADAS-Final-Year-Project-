import subprocess
import threading
import os
import psutil

def set_priority(pid, priority):
    if os.name == 'posix':  # Unix-like systems
        os.nice(priority)
    elif os.name == 'nt':  # Windows
        p = psutil.Process(pid)
        p.nice(psutil.HIGH_PRIORITY_CLASS)

def run_script(script_path, priority=False):
    process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if priority:
        set_priority(process.pid, -10)
    
    stdout, stderr = process.communicate()
    print(f"Output of {script_path}:\n{stdout}")
    if stderr:
        print(f"Error in {script_path}:\n{stderr}")

def main():
    # Create threads for each script
    thread1 = threading.Thread(target=run_script, args=("drowsiness_yawn.py", True))  # Give priority to this script
    thread2 = threading.Thread(target=run_script, args=("main.py",))

    # Start the threads
    thread1.start()
    thread2.start()

    # Wait for both threads to complete
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()
