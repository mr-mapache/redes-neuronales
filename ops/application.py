from threading import Thread
from queue import Queue
from logging import getLogger

from ops.ports import Repository

class Command:
    pass

class Stop(Command):
    pass

class Application(Thread):
    def __init__(self, repository: Repository):
        super().__init__()
        self.repository = repository
        self.queue = Queue()

    def run(self):
        while True:
            try:
                message = self.queue.get()
                if isinstance(message, Stop):
                    break
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        self.queue.put(Stop())