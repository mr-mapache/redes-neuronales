from typing import Any
from typing import Callable
from queue import Queue
from dataclasses import dataclass
from threading import Thread
from logging import getLogger

@dataclass
class Message:
    topic: str
    payload: Any

class Consumer(Thread):
    def __init__(self, handler: Callable[[Message], None]) -> None:
        super().__init__()
        self.queue: Queue[Message] = Queue()
        self.handler = handler
        self.daemon = True
        self.start()

    def consume(self, message: Message) -> None:
        self.queue.put(message)

    def run(self):
        while True:
            message = self.queue.get()
            if message is None:
                break
            self.handler(message.payload)
            self.queue.task_done()
    
    def stop(self) -> None:
        self.consume(None)

class Publisher:
    def __init__(self):
        self.logger = getLogger('ops')
        self.consumers: dict[str, list[Consumer]] = {}

    def subscribe(self, topic: str, consumer: Consumer) -> None:
        self.consumers.setdefault(topic, []).append(consumer)

    def publish(self, topic: str, payload: Any) -> None:
        for consumer in self.consumers.get(topic, []):
            consumer.consume(Message(topic, payload))

    def stop(self) -> None:
        self.logger.info('Stopping publisher')
        for consumers in self.consumers.values():
            for consumer in consumers:
                self.logger.info(f'Stopping consumer {consumer}')
                consumer.stop()
                consumer.join()
        self.consumers.clear()