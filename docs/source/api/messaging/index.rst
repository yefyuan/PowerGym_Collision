Messaging
=========

Message broker system for distributed agent communication.

MessageBroker Interface
-----------------------

.. py:class:: heron.messaging.base.MessageBroker

   Abstract base class for message brokers.

   .. py:method:: publish(channel: str, message: dict) -> None
      :abstractmethod:
      :async:

      Publish message to a channel.

      :param channel: Channel name
      :param message: Message payload

   .. py:method:: consume(channel: str) -> dict
      :abstractmethod:
      :async:

      Consume next message from channel.

      :param channel: Channel name
      :returns: Message payload

   .. py:method:: subscribe(channel: str, callback: Callable) -> None
      :abstractmethod:

      Subscribe to channel with callback.

      :param channel: Channel name
      :param callback: Function called for each message

   .. py:method:: create_channel(channel: str) -> None
      :abstractmethod:

      Create a new channel.

      :param channel: Channel name

InMemoryBroker
--------------

.. py:class:: heron.messaging.memory.InMemoryBroker

   In-memory message broker for local simulation.

   **Usage:**

   .. code-block:: python

      from heron.messaging.memory import InMemoryBroker

      broker = InMemoryBroker()

      # Create channels
      broker.create_channel("control")
      broker.create_channel("status")

      # Subscribe with callback
      def handle_control(msg):
          print(f"Received: {msg}")

      broker.subscribe("control", handle_control)

      # Publish message
      await broker.publish("control", {"setpoint": 1.0})

      # Or consume directly
      msg = await broker.consume("status")

ChannelManager
--------------

.. py:class:: heron.messaging.base.ChannelManager

   Utility for managing broker channels.

   .. py:method:: create_agent_channels(agent_ids: list)

      Create channels for a list of agents.

   .. py:method:: get_channel_name(from_id: str, to_id: str) -> str

      Get channel name for agent-to-agent communication.

   **Usage:**

   .. code-block:: python

      from heron.messaging.base import ChannelManager

      manager = ChannelManager(broker)

      # Create channels for all agent pairs
      manager.create_agent_channels(["mg1", "mg2", "mg3"])

      # Get specific channel
      channel = manager.get_channel_name("mg1", "mg2")

Extending for Production
------------------------

Implement custom brokers for production systems:

.. code-block:: python

   from heron.messaging.base import MessageBroker
   import json

   class KafkaBroker(MessageBroker):
       """Kafka-based message broker."""

       def __init__(self, bootstrap_servers: list[str]):
           from kafka import KafkaProducer, KafkaConsumer
           self.producer = KafkaProducer(
               bootstrap_servers=bootstrap_servers,
               value_serializer=lambda v: json.dumps(v).encode()
           )
           self.consumers = {}

       async def publish(self, channel: str, message: dict):
           self.producer.send(channel, message)
           self.producer.flush()

       async def consume(self, channel: str) -> dict:
           if channel not in self.consumers:
               from kafka import KafkaConsumer
               self.consumers[channel] = KafkaConsumer(
                   channel,
                   bootstrap_servers=self.bootstrap_servers,
                   value_deserializer=lambda v: json.loads(v.decode())
               )
           for msg in self.consumers[channel]:
               return msg.value

       def create_channel(self, channel: str):
           # Kafka topics are auto-created or pre-configured
           pass

       def subscribe(self, channel: str, callback):
           # Implement callback-based subscription
           pass

   class RedisBroker(MessageBroker):
       """Redis-based message broker."""

       def __init__(self, host: str = "localhost", port: int = 6379):
           import redis
           self.client = redis.Redis(host=host, port=port)

       async def publish(self, channel: str, message: dict):
           self.client.publish(channel, json.dumps(message))

       async def consume(self, channel: str) -> dict:
           pubsub = self.client.pubsub()
           pubsub.subscribe(channel)
           for msg in pubsub.listen():
               if msg["type"] == "message":
                   return json.loads(msg["data"])

       def create_channel(self, channel: str):
           pass  # Redis channels are dynamic

       def subscribe(self, channel: str, callback):
           pubsub = self.client.pubsub()
           pubsub.subscribe(**{channel: callback})
