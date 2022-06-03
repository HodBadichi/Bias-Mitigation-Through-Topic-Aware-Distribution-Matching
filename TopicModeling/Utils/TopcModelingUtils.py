from datetime import datetime

current_run_time = None


def getCurrRunTime():
    global current_run_time
    if current_run_time is None:
        current_run_time = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    return current_run_time


def IsAscii(s):
    return all(ord(c) < 128 for c in s)