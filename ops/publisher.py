from logging import getLogger
from typing import Any
from ops.ports import Consumer

class Publisher:
    def __init__(self):
        self.logger = getLogger('ops')
        self.consumers: dict[str, list[Consumer]] = {}

    def subscribe(self, topic: str, consumer: Consumer) -> None:
        self.consumers.setdefault(topic, []).append(consumer)

    def publish(self, topic: str, message: Any):
        for consumer in self.consumers.get(topic, []):
            consumer.consume(message)

    def stop(self) -> None:
        self.logger.info('Stopping publisher')
        for consumers in self.consumers.values():
            for consumer in consumers:
                self.logger.info(f'Stopping consumer {consumer}')
                consumer.stop()