# an echo server inhert from the coroutines scheduler 
# just change the task to the scheduler will do 
from queue import *
import multiprocessing 
import threading
import socket as sc 
import select 
import types 

# def task class in ops 
# every mission run in the cor ops run as task 
# target should be a coroutine function 
# call run funciton to run the target coroutin once 
# support function nest(subroutines), so that the coroutines function can be put into class 
class Task(object): 
    # def class global value 
    taskid = 0 
    # def class function 
    def __init__(self, target): 
        # generate task id 
        Task.taskid += 1 

        # get task id 
        self._tid = Task.taskid 

        # set task target 
        self._target = target 

        # set sendval(initial to None set on the eventloop) 
        self._sendval = None 

        # a stack support function nest 
        self._stack = [] 
        
    def __del__(self): 
        pass 

    # def public function 
    def run(self): 
        while(True): 
            try: 
                result = self._target.send(self._sendval) 
                # SystemCall deal outside(deal as nothing happened) 
                if isinstance(result, SystemCall): 
                    return result 

                if isinstance(result, types.GeneratorType): 
                    # push the current function into the stack and run later 
                    self._stack.append(self._target) 
                    self._sendval = None 
                    self._target = result 
                    continue 

                # nothing left to do 
                if not self._stack: 
                    return 

                # continue to run the stack coroutine function 
                self._sendval = None 
                self._target = self_stack.pop() 
            except StopIteration: 
                if not self._stack: 
                    raise 

                self._sendval = None 
                self._target = self._stack.pop() 
                return self._target.send(self._sendval) 

# define a system call class to do the ops transfer function 
class SystemCall(object): 
    def handle(self): 
        pass 

# the scheduler class arrange the whole tasks:like a teacher with a dozen of children 
# what a scheduler provide? 1.SystemCall(New,KillTask),2.WaitUntilTaskExit,3.TaskQueue(Auto Task Dispatch),4.dynamic I/O Wait 
class Scheduler(object): 
    def __init__(self): 
        # the queue to hold the task running sequence 
        self._ready_queue = Queue() 

        # the task catalog 
        self._task_map = {} 

        # add exit waiting 
        self._exit_waiting = {} 
        
        # record read&write dictionary wait for I/O wait 
        self._read_waiting = {} 
        self._write_waiting = {} 

    def __del__(self): 
        pass 

    # private function define 
    # coroutine funcion(timely check the polling) 
    def __iotask(self): 
        while(True): 
            if self._ready_queue.empty(): 
                self.iopoll(None) 
            else: 
                self.iopoll(0) 
            yield 
    # def public functions 
    # add one task to the task queue(sort by the add sequence) 
    def add_task(self, target): 
        # generate the task object 
        task = Task(target) 

        # add task to the catalog 
        self._task_map[task._tid] = task 

        # put the task to the running queue 
        self.schedule(task) 

        # return the task tid(in case needed) 
        return task._tid 

    # wait for exit 
    def wait_for_exit(self, task, wait_tid): 
        # just add the waitid & task to the stack 
        if wait_tid in self._task_map: 
            self._exit_waiting.setdefault(wait_tid, []).append(task) 
            return True 
        else: 
            return False 

    # wait for read 
    def wait_for_read(self, task, fd): 
        self._read_waiting[fd] = task 

    # wait for write 
    def wait_for_write(self, task, fd): 
        self._write_waiting[fd] = task 

    # remove the task from the catlog 
    def exit_task(self, task): 
        print("Task %d terminated" % task._tid) 
        del self._task_map[task._tid] 

        # try to find the waiting task and add to the scheduler 
        for task in self._exit_waiting.pop(task._tid, []): 
            self.schedule(task) 

        # remove iopoll 
        if len(self._task_map) == 1: 
            print("iopoll task ended!") 
            del self._task_map[self._iotask_id] 

    # iopoll (check whether some fd is available) 
    def iopoll(self, timeout): 
        if self._read_waiting or self._write_waiting: 
            # select the available fd 
            r,w,e = select.select(self._read_waiting, self._write_waiting, [], timeout) 

            # deal with read 
            for fd in r: 
                self.schedule(self._read_waiting.pop(fd)) 

            for fd in w: 
                self.schedule(self._write_waiting.pop(fd)) 

    # set the task to the running queue 
    def schedule(self, task): 
        self._ready_queue.put(task) 

    # def the start the sheduleloop 
    def start(self): 
        self._iotask_id = self.add_task(self.__iotask()) 
        while self._task_map: 
            # get one task from the queue 
            task = self._ready_queue.get() 
            try: 
                # run the task 
                result = task.run() 

                # handle the system call during the task 
                # add inner task or del task or task finished 
                if isinstance(result, SystemCall): 
                    # set the environment param to handle the task 
                    result.task = task 
                    result.sched = self result.handle() 
                    continue 

            except StopIteration: 
                # handle the task end 
                self.exit_task(task) 
                continue 
            # reschedule the task(add to the end of the queue) 
            self.schedule(task) 

        print("scheduler ended!") 

# test area: define two coroutines functions 
# get taskid 
class GetTid(SystemCall): 
    def handle(self): 
        self.task._sendval = self.task._tid 
        self.sched.schedule(self.task) 

# create a new task during the task break 
class NewTask(SystemCall): 
    def __init__(self, target): 
        self._target = target

    def handle(self): 
        tid = self.sched.add_task(self._target) 
        self.task._sendval = tid 
        self.sched.schedule(self.task) 

# kill one task by taskid 
class KillTask(SystemCall): 
    def __init__(self, tid): 
        self._tid = tid 

    def handle(self): 
        task = self.sched._task_map.get(self._tid, None) 

        # get the task and return the kill result 
        if task: 
            task._target.close() 
            self.task.sendval = True 
        else: 
            self.task.sendval = False 
            self.sched.schedule(self.task) 

# wait for some task 
class WaitTask(SystemCall): 
    def __init__(self, tid): 
        self._tid = tid 

    def handle(self): 
        result = self.sched.waitforexit(self.task, self._tid) 
        self.task.sendval = result 

        # if wait failed, tell the orig task immediately 
        if not result: 
            self.sched.schedule(self.task) 

# add read&write I/O async(SystemCall) 
class ReadWait(SystemCall): 
    def __init__(self, f): 
        self._file = f 

    def handle(self): 
        fd = self._file.fileno() 
        self.sched.wait_for_read(self.task, fd) 

# write wait 
class WriteWait(SystemCall): 
    def __init__(self, f): 
        self._file = f 

    def handle(self): 
        fd = self._file.fileno() 
        self.sched.wait_for_write(self.task, fd) 

# define a Sockert Wrapper to run the socket functions 
#class Socket(object): 
#    def __init__(sel
#f, sock): self._sock = sock def accept(self): yield ReadWait(self._sock) client, addr = self._soc k.accept() yield Socket(client), addr def send(self, buffer): while(buffer): yield WriteWait(self 
