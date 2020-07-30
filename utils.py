import datetime
import functools
import os


def timestamp(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        print("%s | " % datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"), end="")
        # print("%s | " % time.asctime(), end="")
        return func(*args, **kwargs)
    return decorated


def timer(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        start = datetime.datetime.now()
        res = func(*args, **kwargs)
        end = datetime.datetime.now()
        print("Time costs: " + ("%s" % (end - start)).split('.')[0])
        return res
    return decorated


@timestamp
def tprint(*args, **kwargs):
    print(*args, **kwargs)


def get_free_gpu():
    """
    取闲置的gpu号
    return: list
    """
    command = os.popen("nvidia-smi -q -d PIDS | grep Processes")
    lines = command.read().split("\n")  # 如果显卡上有进程那么这一行只会有一个Processes
    free_gpu = []
    for i in range(len(lines)):
        if "None" in lines[i]:
            free_gpu.append(str(i))
    return free_gpu